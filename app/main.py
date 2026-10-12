"""
FastAPI inference server for the fine-tuned model.

Endpoints:
  POST /generate    — generate a response for a given instruction/input
  GET  /info        — model and adapter configuration
  GET  /health      — health check

Supports both:
  - Adapter mode: base model + LoRA adapter loaded separately
  - Merged mode:  pre-merged model (faster, no PEFT overhead)

Set MERGED_PATH in .env to use merged mode; otherwise ADAPTER_PATH is used.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.data_prep import format_alpaca, format_chatml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model state — loaded once at startup
_state = {"model": None, "tokeniser": None, "mode": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. Degrades gracefully if ML deps are unavailable."""
    merged_path = Path(settings.MERGED_PATH)
    adapter_path = Path(settings.ADAPTER_PATH)

    try:
        from app.model_setup import load_for_inference

        if merged_path.exists() and any(merged_path.iterdir()):
            logger.info(f"Loading MERGED model from {merged_path}")
            model, tokeniser = load_for_inference(merged_path=str(merged_path), quantise=False)
            _state["mode"] = "merged"
        elif adapter_path.exists() and any(adapter_path.iterdir()):
            logger.info(f"Loading ADAPTER from {adapter_path}")
            model, tokeniser = load_for_inference(adapter_path=str(adapter_path), quantise=True)
            _state["mode"] = "adapter"
        else:
            logger.warning(
                "No trained adapter or merged model found. "
                "Run 'python scripts/train.py' first. "
                "Serving will return 503 until a model is available."
            )
            model, tokeniser = None, None
            _state["mode"] = "none"
    except ImportError as e:
        # ML dependencies (torch/transformers/peft) not installed —
        # e.g. running in a CI environment for API contract tests only.
        logger.warning(f"ML dependencies unavailable ({e}). Serving will return 503.")
        model, tokeniser = None, None
        _state["mode"] = "deps_unavailable"

    _state["model"]     = model
    _state["tokeniser"] = tokeniser
    yield
    # Cleanup on shutdown
    _state["model"] = None
    _state["tokeniser"] = None


app = FastAPI(
    title="Domain-Adapted LLaMA Inference Server",
    description="Serves a LoRA fine-tuned LLaMA model for IT/tech-support domain tasks.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    instruction: str = Field(..., description="The task instruction")
    input: str = Field("", description="Optional input context (Alpaca format)")
    max_new_tokens: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    model_config = {
        "json_schema_extra": {
            "example": {
                "instruction": "Classify the urgency of this IT support ticket as low, medium, or high.",
                "input": "The office printer on the 3rd floor is out of toner.",
                "max_new_tokens": 100,
                "temperature": 0.1,
            }
        }
    }


class GenerateResponse(BaseModel):
    instruction: str
    input: str
    response: str
    model_mode: str


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return f"""<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🧠 Domain-Adapted LLaMA Inference Server</h1>
    <p>Base model: {settings.BASE_MODEL}</p>
    <p>Mode: <b>{_state.get('mode','not loaded')}</b></p>
    <p><a href="/docs">📖 API Docs</a></p>
    <h3>Domain: IT/Tech Support</h3>
    <p>Tasks: urgency classification, diagnosis, email drafting, summarisation,
    entity extraction, commit messages, error rewriting, sentiment analysis</p>
    </body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model"] is not None else "no_model_loaded",
        "mode": _state["mode"],
        "base_model": settings.BASE_MODEL,
    }


@app.get("/info")
async def info():
    return {
        "base_model": settings.BASE_MODEL,
        "mode": _state["mode"],
        "lora_r": settings.LORA_R,
        "lora_alpha": settings.LORA_ALPHA,
        "target_modules": settings.target_modules_list,
        "dataset_format": settings.DATASET_FORMAT,
        "quantised": settings.USE_4BIT,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate a response using the fine-tuned model.

    The model was fine-tuned on IT/tech-support tasks:
    urgency classification, technical diagnosis, support email drafting,
    changelog summarisation, entity extraction, commit message generation,
    error message rewriting, and sentiment analysis.
    """
    if _state["model"] is None:
        raise HTTPException(
            503,
            "No model loaded. Run 'python scripts/train.py' to train an adapter, "
            "then restart the server."
        )

    if not req.instruction.strip():
        raise HTTPException(400, "Instruction cannot be empty.")

    from app.evaluation import generate_response

    prompt = format_alpaca({
        "instruction": req.instruction,
        "input": req.input,
        "output": "",
    })

    try:
        result = generate_response(
            _state["model"], _state["tokeniser"], prompt,
            max_new_tokens=req.max_new_tokens,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(500, f"Generation failed: {str(e)}")

    return GenerateResponse(
        instruction=req.instruction,
        input=req.input,
        response=result,
        model_mode=_state["mode"],
    )
