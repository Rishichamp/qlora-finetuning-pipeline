"""
Evaluation module — compares base model vs fine-tuned model.

Metrics:
  - Perplexity (lower = better fit to domain)
  - ROUGE (for structured/summarisation tasks)
  - BLEU (for tasks with reference outputs)
  - LLM-as-judge (optional, requires OpenAI API key)

Usage:
    python -m app.evaluation --compare
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch

from app.config import settings
from app.data_prep import load_jsonl, format_example

logger = logging.getLogger(__name__)


# ── Perplexity ────────────────────────────────────────────────────

def compute_perplexity(model, tokeniser, texts: list[str], max_length: int = 1024) -> float:
    """
    Compute perplexity = exp(average cross-entropy loss).
    Lower perplexity = model assigns higher probability to the text
    = better fit to this distribution.
    """
    total_loss, total_tokens = 0.0, 0
    model.eval()

    with torch.no_grad():
        for text in texts:
            inputs = tokeniser(
                text, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            if inputs["input_ids"].shape[1] < 2:
                continue

            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].shape[1]
            total_loss   += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ── Generation ────────────────────────────────────────────────────

def generate_response(model, tokeniser, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response from the model for a given prompt."""
    inputs = tokeniser(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokeniser.eos_token_id,
        )
    full_text = tokeniser.decode(output_ids[0], skip_special_tokens=True)
    # Return only the newly generated portion
    return full_text[len(tokeniser.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


# ── ROUGE / BLEU ──────────────────────────────────────────────────

def compute_rouge_bleu(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE and BLEU scores between predictions and references."""
    try:
        from evaluate import load
        rouge = load("rouge")
        rouge_scores = rouge.compute(predictions=predictions, references=references)
    except Exception as e:
        logger.warning(f"ROUGE unavailable: {e}")
        rouge_scores = {}

    try:
        from evaluate import load
        bleu = load("sacrebleu")
        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
    except Exception as e:
        logger.warning(f"BLEU unavailable: {e}")
        bleu_score = {}

    return {**rouge_scores, "bleu": bleu_score.get("score")}


# ── Before/After comparison ──────────────────────────────────────

def evaluate_on_test_set(
    model, tokeniser, test_examples: list[dict], fmt: str, max_examples: int = 20
) -> dict:
    """
    Run generation + scoring on a held-out test set.

    Returns a dict with:
      - predictions: list of generated responses
      - references: list of expected responses
      - rouge/bleu scores
      - perplexity on reference texts
    """
    test_examples = test_examples[:max_examples]
    predictions, references, prompts = [], [], []

    for ex in test_examples:
        if fmt == "alpaca":
            # Build prompt without the response (for generation)
            prompt_ex = {**ex, "output": ""}
            prompt = format_example(prompt_ex, fmt)
            reference = ex.get("output", "")
        else:
            # For chatml, use everything up to last assistant turn
            prompt = format_example(ex, fmt)
            reference = ""

        generated = generate_response(model, tokeniser, prompt, max_new_tokens=150)
        predictions.append(generated)
        references.append(reference)
        prompts.append(prompt[:100])

    scores = compute_rouge_bleu(predictions, references) if any(references) else {}

    full_texts = [format_example(ex, fmt) for ex in test_examples]
    ppl = compute_perplexity(model, tokeniser, full_texts)

    return {
        "n_examples": len(test_examples),
        "perplexity": round(ppl, 2),
        "scores": scores,
        "samples": [
            {"prompt": p, "predicted": pred[:200], "reference": ref[:200]}
            for p, pred, ref in zip(prompts[:5], predictions[:5], references[:5])
        ],
    }


def compare_models(
    base_model, base_tokeniser,
    finetuned_model, finetuned_tokeniser,
    test_path: str, fmt: str, max_examples: int = 20,
) -> dict:
    """
    Run before/after comparison between base and fine-tuned models.
    """
    test_examples = load_jsonl(test_path)

    logger.info("Evaluating BASE model...")
    base_results = evaluate_on_test_set(base_model, base_tokeniser, test_examples, fmt, max_examples)

    logger.info("Evaluating FINE-TUNED model...")
    ft_results = evaluate_on_test_set(finetuned_model, finetuned_tokeniser, test_examples, fmt, max_examples)

    ppl_delta = ft_results["perplexity"] - base_results["perplexity"]

    return {
        "base":       base_results,
        "finetuned":  ft_results,
        "perplexity_delta": round(ppl_delta, 2),
        "perplexity_improved": ppl_delta < 0,
        "summary": (
            f"Perplexity: {base_results['perplexity']:.2f} -> {ft_results['perplexity']:.2f} "
            f"({'improved' if ppl_delta < 0 else 'regressed'} by {abs(ppl_delta):.2f})"
        ),
    }


if __name__ == "__main__":
    import argparse
    from app.model_setup import load_base_model, load_tokeniser, load_for_inference

    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Compare base vs fine-tuned")
    parser.add_argument("--test-path", default="./data/eval.jsonl")
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.compare:
        print("Loading base model...")
        base_model = load_base_model(quantise=True)
        base_tok   = load_tokeniser()

        print("Loading fine-tuned model...")
        ft_model, ft_tok = load_for_inference(adapter_path=settings.ADAPTER_PATH, quantise=True)

        results = compare_models(
            base_model, base_tok, ft_model, ft_tok,
            test_path=args.test_path, fmt=settings.DATASET_FORMAT,
            max_examples=args.max_examples,
        )
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(results["summary"])
        print(json.dumps(results, indent=2, default=str))
