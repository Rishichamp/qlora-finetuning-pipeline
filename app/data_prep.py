"""
Dataset preparation for fine-tuning.

Supports three formats:
  - Alpaca:  {instruction, input, output}
  - ChatML:  {conversations: [{from, value}]}
  - Plain:   raw text strings

Includes:
  - Data loading from JSONL files
  - Quality filtering (length, deduplication, artifact removal)
  - Prompt formatting per template
  - Train/eval splitting
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# ── Prompt templates ──────────────────────────────────────────────

ALPACA_TEMPLATE = """Below is an instruction that describes a task{input_section}.

### Instruction:
{instruction}
{input_block}
### Response:
{output}"""

CHATML_TEMPLATE_TURN = "<|im_start|>{role}\n{content}<|im_end|>\n"


def format_alpaca(example: dict) -> str:
    """Format an Alpaca-style example into a prompt string."""
    input_text = example.get("input", "").strip()
    if input_text:
        input_section = ", paired with an input that provides further context"
        input_block   = f"\n### Input:\n{input_text}\n"
    else:
        input_section = ""
        input_block   = ""
    return ALPACA_TEMPLATE.format(
        input_section=input_section,
        instruction=example["instruction"].strip(),
        input_block=input_block,
        output=example.get("output", "").strip(),
    )


def format_chatml(example: dict) -> str:
    """Format a ShareGPT/ChatML-style example into a prompt string."""
    conversations = example.get("conversations", [])
    result = ""
    for msg in conversations:
        role    = msg.get("from", "user")
        content = msg.get("value", "")
        # Normalise role names
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        role = role_map.get(role, role)
        result += CHATML_TEMPLATE_TURN.format(role=role, content=content)
    return result


def format_example(example: dict, fmt: str) -> str:
    """Route to the appropriate formatter."""
    if fmt == "alpaca":
        return format_alpaca(example)
    elif fmt == "chatml":
        return format_chatml(example)
    elif fmt == "plain":
        return example.get("text", example.get("content", ""))
    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'alpaca', 'chatml', or 'plain'.")


# ── Data loading ──────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file (one JSON object per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return examples


# ── Data cleaning ─────────────────────────────────────────────────

BAD_PATTERNS = [
    "as an ai language model",
    "as an ai, i",
    "i cannot and will not",
    "i'm just an ai",
    "it is worth noting that",  # filler
    "certainly! here",
    "of course! i",
    "great question!",
    "i'd be happy to help with that",
]


def quality_filter(
    examples: list[dict],
    fmt: str,
    min_tokens: int = 20,
    max_tokens: int = 1500,
) -> list[dict]:
    """
    Filter dataset for quality:
      1. Length bounds (by approximate word count)
      2. Remove common LLM artifact phrases
      3. Remove HTML and excessive whitespace
      4. Basic deduplication by instruction text
    """
    kept = []
    seen_instructions = set()

    for ex in examples:
        # Get the main text content for filtering
        if fmt == "alpaca":
            content = ex.get("output", "")
            # Dedup key includes instruction AND input — many examples share
            # the same instruction template with different inputs, which
            # are NOT duplicates and must be kept.
            key     = (ex.get("instruction", "")[:80] + "||" + ex.get("input", "")[:80])
        elif fmt == "chatml":
            convs   = ex.get("conversations", [])
            content = " ".join(c.get("value", "") for c in convs if c.get("from") in ("gpt", "assistant"))
            key     = str(convs[:2])[:150]
        else:
            content = ex.get("text", ex.get("content", ""))
            key     = content[:150]

        # 1. Length filter (approximate word count)
        word_count = len(content.split())
        if word_count < min_tokens or word_count > max_tokens:
            continue

        # 2. Artifact filter
        content_lower = content.lower()
        if any(bad in content_lower for bad in BAD_PATTERNS):
            continue

        # 3. Clean HTML and whitespace
        content = re.sub(r"<[^>]+>", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            continue

        # 4. Deduplication by instruction/key
        if key in seen_instructions:
            continue
        seen_instructions.add(key)

        # Update cleaned content
        if fmt == "alpaca":
            ex = {**ex, "output": content}
        kept.append(ex)

    pct = 100 * len(kept) / max(len(examples), 1)
    logger.info(f"Quality filter: kept {len(kept)}/{len(examples)} ({pct:.1f}%)")
    return kept


# ── Dataset builder ───────────────────────────────────────────────

def build_dataset(
    data_path: str,
    fmt: str,
    eval_path: Optional[str] = None,
    eval_split: float = 0.1,
    seed: int = 42,
    min_tokens: int = 20,
    max_tokens: int = 1500,
) -> DatasetDict:
    """
    Full dataset pipeline:
      1. Load from JSONL
      2. Quality filter
      3. Format into prompt strings
      4. Split into train/eval
      5. Return as HuggingFace DatasetDict
    """
    # Load
    raw = load_jsonl(data_path)

    # Filter
    cleaned = quality_filter(raw, fmt, min_tokens, max_tokens)
    if not cleaned:
        raise ValueError("No examples passed the quality filter. Check your data and filter settings.")

    # Format into text
    formatted = []
    for ex in cleaned:
        try:
            text = format_example(ex, fmt)
            if text.strip():
                formatted.append({"text": text})
        except Exception as e:
            logger.warning(f"Formatting failed for example: {e}")

    logger.info(f"Formatted {len(formatted)} examples")

    if eval_path and Path(eval_path).exists():
        # Separate eval file provided
        eval_raw     = load_jsonl(eval_path)
        eval_cleaned = quality_filter(eval_raw, fmt, min_tokens, max_tokens)
        eval_fmt     = [{"text": format_example(ex, fmt)} for ex in eval_cleaned]
        return DatasetDict({
            "train": Dataset.from_list(formatted),
            "eval":  Dataset.from_list(eval_fmt),
        })
    else:
        # Auto-split
        full_ds = Dataset.from_list(formatted)
        split   = full_ds.train_test_split(test_size=eval_split, seed=seed)
        return DatasetDict({"train": split["train"], "eval": split["test"]})


def print_dataset_stats(datasets: DatasetDict) -> None:
    """Print dataset statistics for verification before training."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    for split, ds in datasets.items():
        lengths = [len(ex["text"].split()) for ex in ds]
        avg_len = sum(lengths) / max(len(lengths), 1)
        print(f"{split:10} | {len(ds):6,} examples | avg {avg_len:.0f} words")
        # Show first example preview
        if len(ds) > 0:
            preview = ds[0]["text"][:200].replace("\n", " ")
            print(f"           | Sample: {preview}...")
    print("=" * 50 + "\n")
