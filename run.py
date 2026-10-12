"""Entry point for the inference server. Usage: python run.py"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"\n🧠  Domain-Adapted LLaMA Inference Server")
    print(f"   Base model : {settings.BASE_MODEL}")
    print(f"   LoRA r/α   : {settings.LORA_R}/{settings.LORA_ALPHA}")
    print(f"   Docs       : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   If you haven't trained yet:")
    print(f"   1) python scripts/train.py")
    print(f"   2) python scripts/merge_model.py  (optional, for faster inference)")
    print(f"   3) python run.py\n")
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, log_level="info")
