"""
Training script for the Domain-Adapted LLaMA Fine-Tuning Pipeline.

Usage:
    python scripts/train.py

Pipeline:
  1. Load and prepare dataset (Alpaca/ChatML/plain format)
  2. Load base model with 4-bit QLoRA quantisation
  3. Apply LoRA adapters
  4. Train with SFTTrainer (response masking + packing)
  5. Save adapter weights

All settings controlled via app/config.py / .env
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.data_prep import build_dataset, print_dataset_stats
from app.model_setup import load_base_model, load_tokeniser, apply_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "=" * 60)
    print("DOMAIN-ADAPTED LLAMA FINE-TUNING PIPELINE")
    print("=" * 60)
    print(f"Base model     : {settings.BASE_MODEL}")
    print(f"Dataset format : {settings.DATASET_FORMAT}")
    print(f"LoRA rank      : {settings.LORA_R} (alpha={settings.LORA_ALPHA})")
    print(f"Target modules : {', '.join(settings.target_modules_list)}")
    print(f"Quantisation   : {'4-bit NF4' if settings.USE_4BIT else 'BF16 (full precision)'}")
    print(f"Effective batch: {settings.effective_batch_size} "
          f"({settings.PER_DEVICE_BATCH} x {settings.GRAD_ACCUM} grad accum)")
    print(f"Learning rate  : {settings.LEARNING_RATE}")
    print(f"Epochs         : {settings.NUM_EPOCHS}")
    print("=" * 60 + "\n")

    # ── Step 1: Dataset ──────────────────────────────────────────
    logger.info("Building dataset...")
    datasets = build_dataset(
        data_path=settings.DATA_PATH,
        fmt=settings.DATASET_FORMAT,
        eval_path=settings.EVAL_DATA_PATH if Path(settings.EVAL_DATA_PATH).exists() else None,
    )
    print_dataset_stats(datasets)

    # ── Step 2: Tokeniser ────────────────────────────────────────
    logger.info("Loading tokeniser...")
    tokeniser = load_tokeniser()

    # ── Step 3: Model ─────────────────────────────────────────────
    logger.info("Loading base model (this may take a few minutes)...")
    model = load_base_model(quantise=settings.USE_4BIT)

    # ── Step 4: Apply LoRA ───────────────────────────────────────
    logger.info("Applying LoRA adapters...")
    model, lora_config = apply_lora(model)

    # ── Step 5: Training arguments ───────────────────────────────
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    training_args = TrainingArguments(
        output_dir=settings.OUTPUT_DIR,
        per_device_train_batch_size=settings.PER_DEVICE_BATCH,
        per_device_eval_batch_size=settings.PER_DEVICE_BATCH,
        gradient_accumulation_steps=settings.GRAD_ACCUM,
        learning_rate=settings.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=settings.WARMUP_RATIO,
        num_train_epochs=settings.NUM_EPOCHS,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit" if settings.USE_4BIT else "adamw_torch",
        save_strategy="steps",
        save_steps=settings.SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=settings.EVAL_STEPS,
        logging_steps=settings.LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],  # set to ["wandb"] or ["tensorboard"] if desired
        save_total_limit=2,
    )

    # ── Step 6: Response masking ─────────────────────────────────
    # Determine response template based on dataset format
    if settings.DATASET_FORMAT == "alpaca":
        response_template = "### Response:\n"
    elif settings.DATASET_FORMAT == "chatml":
        response_template = "<|im_start|>assistant\n"
    else:
        response_template = None  # no masking for plain text

    collator = None
    if response_template:
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokeniser
        )
        logger.info(f"Response masking enabled: '{response_template}'")
    else:
        logger.warning("No response masking — model will train on full text (plain format)")

    # ── Step 7: Build trainer ────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokeniser,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=settings.MAX_SEQ_LENGTH,
        packing=settings.USE_PACKING and response_template is None,
        # Note: packing is disabled when using response masking,
        # since packing concatenates examples and would break masking boundaries
    )

    # ── Step 8: Train ─────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train()

    # ── Step 9: Save adapter ─────────────────────────────────────
    adapter_path = settings.ADAPTER_PATH
    logger.info(f"Saving LoRA adapter to {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokeniser.save_pretrained(adapter_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Adapter saved to: {adapter_path}")
    print(f"Next steps:")
    print(f"  1. Evaluate:  python -m app.evaluation --compare")
    print(f"  2. Merge:     python scripts/merge_model.py")
    print(f"  3. Serve:     python run.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
