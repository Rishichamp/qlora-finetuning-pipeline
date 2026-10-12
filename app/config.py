"""
Configuration for the Domain-Adapted LLaMA Fine-Tuning Pipeline.
All settings controlled via environment variables / .env file.
"""
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    # ── Base model ────────────────────────────────────────────────
    # Supported: meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B,
    #            mistralai/Mistral-7B-v0.3, microsoft/Phi-3-mini-4k-instruct
    BASE_MODEL: str         = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-3B")
    HF_TOKEN: str           = os.getenv("HF_TOKEN", "")

    # ── LoRA config ───────────────────────────────────────────────
    LORA_R: int             = int(os.getenv("LORA_R", "16"))
    LORA_ALPHA: int         = int(os.getenv("LORA_ALPHA", "32"))
    LORA_DROPOUT: float     = float(os.getenv("LORA_DROPOUT", "0.05"))
    # Comma-separated list of modules to target
    LORA_TARGET_MODULES: str = os.getenv(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )

    # ── Quantisation ──────────────────────────────────────────────
    USE_4BIT: bool          = os.getenv("USE_4BIT", "true").lower() == "true"
    USE_DOUBLE_QUANT: bool  = os.getenv("USE_DOUBLE_QUANT", "true").lower() == "true"
    QUANT_TYPE: str         = os.getenv("QUANT_TYPE", "nf4")

    # ── Training ──────────────────────────────────────────────────
    OUTPUT_DIR: str         = os.getenv("OUTPUT_DIR", "./output")
    DATA_PATH: str          = os.getenv("DATA_PATH", "./data/train.jsonl")
    EVAL_DATA_PATH: str     = os.getenv("EVAL_DATA_PATH", "./data/eval.jsonl")

    PER_DEVICE_BATCH: int   = int(os.getenv("PER_DEVICE_BATCH", "4"))
    GRAD_ACCUM: int         = int(os.getenv("GRAD_ACCUM", "4"))
    LEARNING_RATE: float    = float(os.getenv("LEARNING_RATE", "2e-4"))
    NUM_EPOCHS: int         = int(os.getenv("NUM_EPOCHS", "3"))
    WARMUP_RATIO: float     = float(os.getenv("WARMUP_RATIO", "0.05"))
    MAX_SEQ_LENGTH: int     = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
    USE_PACKING: bool       = os.getenv("USE_PACKING", "true").lower() == "true"

    SAVE_STEPS: int         = int(os.getenv("SAVE_STEPS", "200"))
    EVAL_STEPS: int         = int(os.getenv("EVAL_STEPS", "200"))
    LOGGING_STEPS: int      = int(os.getenv("LOGGING_STEPS", "25"))

    # ── Dataset format ────────────────────────────────────────────
    # "alpaca" | "chatml" | "plain"
    DATASET_FORMAT: str     = os.getenv("DATASET_FORMAT", "alpaca")

    # ── Inference / API ───────────────────────────────────────────
    ADAPTER_PATH: str       = os.getenv("ADAPTER_PATH", "./lora_adapter")
    MERGED_PATH: str        = os.getenv("MERGED_PATH", "./merged_model")
    APP_PORT: int           = int(os.getenv("APP_PORT", "8000"))
    APP_HOST: str           = os.getenv("APP_HOST", "0.0.0.0")

    @property
    def target_modules_list(self) -> list[str]:
        return [m.strip() for m in self.LORA_TARGET_MODULES.split(",")]

    @property
    def effective_batch_size(self) -> int:
        return self.PER_DEVICE_BATCH * self.GRAD_ACCUM

settings = Settings()
