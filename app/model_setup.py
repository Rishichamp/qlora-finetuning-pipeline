"""
Model loading and LoRA configuration for QLoRA fine-tuning.

Handles:
  - 4-bit NF4 quantisation via bitsandbytes
  - LoRA adapter attachment via PEFT
  - Tokeniser setup (padding, EOS token)
  - Trainable parameter reporting
"""

import logging
from typing import Optional
import torch

from app.config import settings

logger = logging.getLogger(__name__)


def get_bnb_config():
    """Build BitsAndBytesConfig for 4-bit QLoRA."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=settings.USE_4BIT,
        bnb_4bit_use_double_quant=settings.USE_DOUBLE_QUANT,
        bnb_4bit_quant_type=settings.QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model(quantise: bool = True):
    """
    Load the base model with optional 4-bit quantisation.

    Args:
        quantise: if True, apply QLoRA 4-bit quantisation (default).
                  Set False for full fine-tuning or CPU-only inference.
    """
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading base model: {settings.BASE_MODEL}")
    kwargs = dict(
        pretrained_model_name_or_path=settings.BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    if settings.HF_TOKEN:
        kwargs["token"] = settings.HF_TOKEN

    if quantise and settings.USE_4BIT:
        kwargs["quantization_config"] = get_bnb_config()
        logger.info("4-bit NF4 quantisation enabled")
    else:
        kwargs["torch_dtype"] = torch.bfloat16
        logger.info("Loading in BF16 (no quantisation)")

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    # Disable cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.config.pretraining_tp = 1  # tensor parallelism = 1 for single GPU

    return model


def load_tokeniser():
    """Load and configure the tokeniser."""
    from transformers import AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(
        settings.BASE_MODEL,
        trust_remote_code=True,
        token=settings.HF_TOKEN or None,
    )
    # Many LLMs don't have a pad token — use EOS
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
        tokeniser.pad_token_id = tokeniser.eos_token_id

    # Pad on the right for causal LM training
    tokeniser.padding_side = "right"

    logger.info(
        f"Tokeniser loaded: vocab_size={tokeniser.vocab_size}, "
        f"pad_token='{tokeniser.pad_token}'"
    )
    return tokeniser


def apply_lora(model) -> tuple:
    """
    Prepare model for kbit training and apply LoRA adapters.

    Returns: (peft_model, lora_config)
    """
    from peft import (
        LoraConfig, get_peft_model,
        prepare_model_for_kbit_training, TaskType
    )

    # Enable gradient checkpointing + cast layer norms to float32
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_config = LoraConfig(
        r=settings.LORA_R,
        lora_alpha=settings.LORA_ALPHA,
        target_modules=settings.target_modules_list,
        lora_dropout=settings.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

    peft_model = get_peft_model(model, lora_config)
    _print_trainable_params(peft_model)

    return peft_model, lora_config


def _print_trainable_params(model) -> None:
    """Print a summary of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / max(total, 1)

    print(f"\n{'=' * 50}")
    print(f"TRAINABLE PARAMETERS")
    print(f"{'=' * 50}")
    print(f"  Trainable : {trainable:>15,}  ({pct:.4f}%)")
    print(f"  Frozen    : {total - trainable:>15,}")
    print(f"  Total     : {total:>15,}")
    print(f"{'=' * 50}\n")


def load_for_inference(
    adapter_path: Optional[str] = None,
    merged_path: Optional[str] = None,
    quantise: bool = True,
):
    """
    Load model for inference.

    Two modes:
      - adapter_path: load base + LoRA adapter (flexible, smaller disk)
      - merged_path:  load pre-merged model (fastest inference, no PEFT overhead)
    """
    from transformers import AutoModelForCausalLM

    tokeniser = load_tokeniser()

    if merged_path:
        logger.info(f"Loading merged model from {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif adapter_path:
        from peft import PeftModel
        logger.info(f"Loading base + adapter from {adapter_path}")
        base  = load_base_model(quantise=quantise)
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        raise ValueError("Provide either adapter_path or merged_path.")

    model.eval()
    return model, tokeniser


def merge_and_save(adapter_path: str, output_path: str) -> None:
    """
    Merge LoRA weights into base model and save.
    The merged model can be used without PEFT and has zero inference overhead.
    """
    from peft import PeftModel
    import torch

    logger.info("Loading base model for merging (BF16, no quantisation)...")
    base = load_base_model(quantise=False)

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging LoRA weights into base model...")
    # merge_and_unload(): W_merged = W_base + (alpha/r) * B @ A
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    load_tokeniser().save_pretrained(output_path)
    logger.info("Merge complete.")
