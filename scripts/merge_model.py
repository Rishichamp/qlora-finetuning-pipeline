"""
Merge LoRA adapter weights into the base model for production deployment.

After merging:
  - No PEFT dependency needed at inference time
  - Zero adapter overhead (single forward pass through merged weights)
  - Can be quantised to GGUF/AWQ for further deployment optimisation

Usage:
    python scripts/merge_model.py
    python scripts/merge_model.py --adapter ./lora_adapter --output ./merged_model
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.model_setup import merge_and_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", default=settings.ADAPTER_PATH,
                        help="Path to the trained LoRA adapter")
    parser.add_argument("--output", default=settings.MERGED_PATH,
                        help="Output path for the merged model")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error(f"Adapter path does not exist: {adapter_path}")
        logger.error("Run 'python scripts/train.py' first to produce an adapter.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MERGING LORA ADAPTER INTO BASE MODEL")
    print("=" * 60)
    print(f"Base model : {settings.BASE_MODEL}")
    print(f"Adapter    : {args.adapter}")
    print(f"Output     : {args.output}")
    print("=" * 60 + "\n")

    merge_and_save(adapter_path=str(adapter_path), output_path=args.output)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Merged model saved to: {args.output}")
    print("\nNext steps:")
    print("  - Serve directly:  set MERGED_PATH in .env and run 'python run.py'")
    print("  - Quantise to GGUF for CPU deployment (see study guide)")
    print("  - Quantise to AWQ for GPU deployment (see study guide)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
