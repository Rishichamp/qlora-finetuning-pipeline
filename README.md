# QLoRA Fine-Tuning Pipeline

[![tests](https://github.com/Rishichamp/qlora-finetuning-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/qlora-finetuning-pipeline/actions/workflows/tests.yml)

An end-to-end pipeline for adapting an open LLM (default **Llama 3.2 3B**) to a domain, here IT and tech support, with **QLoRA**: data preparation, 4-bit training with LoRA adapters, evaluation against the base model, adapter merging and a FastAPI inference server.

## Pipeline

```
JSONL data ──► data_prep ──► 4-bit NF4 base model + LoRA (PEFT) ──► SFTTrainer (TRL)
   (Alpaca / ChatML / plain)                                          │
                                                                      ▼
               FastAPI /generate ◄── merge_model.py ◄── evaluation (base vs fine-tuned)
```

## Features

- **Data preparation:** Alpaca, ChatML and plain-text formats; quality filters for length, duplicates and artefacts; prompt templating; train/eval split.
- **QLoRA:** 4-bit NF4 quantisation with double quantisation (bitsandbytes), and LoRA adapters (r=16) on the attention and MLP projections.
- **Training** with TRL's `SFTTrainer`: response masking, sequence packing, warmup, gradient accumulation.
- **Evaluation:** perplexity, ROUGE and BLEU, comparing the fine-tuned model against the base model.
- **Merging:** folds the adapter into the base weights for faster inference without PEFT.
- **Inference API:** works in adapter mode (base + LoRA) or merged mode.

## Tech stack

Python 3.11 · PyTorch · Hugging Face Transformers · PEFT · TRL · bitsandbytes · Accelerate · Datasets · evaluate · FastAPI · pytest

## Project structure

```
app/
  data_prep.py     loading, quality filtering, prompt formatting
  model_setup.py   4-bit loading, LoRA config, tokenizer setup
  evaluation.py    perplexity / ROUGE / BLEU, base vs fine-tuned comparison
  config.py        all hyperparameters via .env
  main.py          FastAPI inference server
scripts/
  train.py         run fine-tuning
  merge_model.py   merge the adapter into the base model
data/              sample train/eval sets (IT support)
tests/             21 tests
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # set HF_TOKEN; Llama models need access approval on Hugging Face
python scripts/train.py        # needs an NVIDIA GPU (4-bit training uses CUDA)
python -m app.evaluation --compare
python scripts/merge_model.py
python run.py                  # http://127.0.0.1:8000/docs
```

The tests and the API skeleton run on CPU. Training and 4-bit inference need a CUDA GPU; a free Colab T4 is enough for the 3B model.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/generate` | Generate a response for an instruction (and optional input) |
| GET | `/info` | Model and adapter configuration |
| GET | `/health` | Health check (reports whether a model is loaded) |

## Tests

```bash
pytest tests/ -q
```

21 tests covering configuration, every prompt format, the quality filters, JSONL loading (including malformed lines) and the API (503 before a model is loaded), all without downloading a model.

## Notes

The sample data is intentionally small (35 training and 6 evaluation examples), enough to exercise the pipeline. Built as part of Module 6 of a Generative AI engineering programme.

## Author

**Rishi Singh** — [GitHub](https://github.com/Rishichamp)
