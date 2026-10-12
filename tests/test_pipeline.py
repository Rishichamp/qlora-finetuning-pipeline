"""
Tests for the Domain-Adapted LLaMA Fine-Tuning Pipeline.

These tests avoid downloading or loading actual models — they test:
  - Data preparation (formatting, filtering, dataset building)
  - Configuration defaults
  - API contract (with mocked model state)
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.config import settings
from app.data_prep import (
    format_alpaca, format_chatml, format_example,
    load_jsonl, quality_filter, build_dataset, BAD_PATTERNS
)


# ── Config tests ──────────────────────────────────────────────────
def test_config_defaults():
    assert settings.LORA_R == 16
    assert settings.LORA_ALPHA == 32
    assert settings.DATASET_FORMAT == "alpaca"

def test_target_modules_list():
    modules = settings.target_modules_list
    assert "q_proj" in modules
    assert "v_proj" in modules
    assert len(modules) == 7

def test_effective_batch_size():
    assert settings.effective_batch_size == settings.PER_DEVICE_BATCH * settings.GRAD_ACCUM


# ── Formatting tests ───────────────────────────────────────────────
def test_format_alpaca_with_input():
    example = {"instruction": "Classify sentiment.", "input": "I love this!", "output": "Positive"}
    result = format_alpaca(example)
    assert "### Instruction:" in result
    assert "### Input:" in result
    assert "### Response:" in result
    assert "Classify sentiment." in result
    assert "I love this!" in result
    assert "Positive" in result

def test_format_alpaca_without_input():
    example = {"instruction": "Say hello.", "input": "", "output": "Hello!"}
    result = format_alpaca(example)
    assert "### Instruction:" in result
    assert "### Input:" not in result
    assert "### Response:" in result

def test_format_chatml():
    example = {"conversations": [
        {"from": "system", "value": "You are helpful."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "Hello!"},
    ]}
    result = format_chatml(example)
    assert "<|im_start|>system" in result
    assert "<|im_start|>user" in result   # human -> user
    assert "<|im_start|>assistant" in result  # gpt -> assistant
    assert "Hello!" in result

def test_format_example_routes_correctly():
    alpaca_ex = {"instruction": "Test", "input": "", "output": "Done"}
    assert "### Instruction:" in format_example(alpaca_ex, "alpaca")

def test_format_example_invalid_format():
    with pytest.raises(ValueError):
        format_example({}, "unknown_format")


# ── Quality filter tests ──────────────────────────────────────────
def test_quality_filter_removes_short_responses():
    examples = [
        {"instruction": "Test", "input": "", "output": "Yes"},  # too short
        {"instruction": "Test", "input": "", "output": " ".join(["word"] * 30)},  # ok
    ]
    filtered = quality_filter(examples, "alpaca", min_tokens=20, max_tokens=1500)
    assert len(filtered) == 1
    assert len(filtered[0]["output"].split()) >= 20

def test_quality_filter_removes_bad_patterns():
    examples = [
        {"instruction": "Test", "input": "", "output": "As an AI language model " + " ".join(["word"]*20)},
        {"instruction": "Test2", "input": "", "output": " ".join(["valid", "response", "text"] * 10)},
    ]
    filtered = quality_filter(examples, "alpaca", min_tokens=10, max_tokens=1500)
    assert len(filtered) == 1
    assert "as an ai" not in filtered[0]["output"].lower()

def test_quality_filter_deduplication():
    examples = [
        {"instruction": "Same question", "input": "context A", "output": " ".join(["text"]*20)},
        {"instruction": "Same question", "input": "context A", "output": " ".join(["other"]*20)},  # true duplicate
        {"instruction": "Same question", "input": "context B", "output": " ".join(["different"]*20)},  # different input — keep
    ]
    filtered = quality_filter(examples, "alpaca", min_tokens=10, max_tokens=1500)
    assert len(filtered) == 2  # second is a duplicate (same instruction+input); third has different input

def test_quality_filter_too_long():
    examples = [
        {"instruction": "Test", "input": "", "output": " ".join(["word"] * 2000)},  # too long
    ]
    filtered = quality_filter(examples, "alpaca", min_tokens=20, max_tokens=1500)
    assert len(filtered) == 0

def test_bad_patterns_list_not_empty():
    assert len(BAD_PATTERNS) > 0
    assert all(isinstance(p, str) for p in BAD_PATTERNS)


# ── JSONL loading tests ────────────────────────────────────────────
def test_load_jsonl(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(
        '{"instruction": "Test1", "input": "", "output": "Response1"}\n'
        '{"instruction": "Test2", "input": "", "output": "Response2"}\n'
    )
    examples = load_jsonl(str(test_file))
    assert len(examples) == 2
    assert examples[0]["instruction"] == "Test1"

def test_load_jsonl_skips_malformed(tmp_path):
    test_file = tmp_path / "test.jsonl"
    test_file.write_text(
        '{"instruction": "Test1", "input": "", "output": "Response1"}\n'
        'not valid json\n'
        '{"instruction": "Test2", "input": "", "output": "Response2"}\n'
    )
    examples = load_jsonl(str(test_file))
    assert len(examples) == 2  # malformed line skipped


# ── Dataset building (uses real train.jsonl) ───────────────────────
def test_build_dataset_from_real_data():
    """Test against the actual data/train.jsonl shipped with the project."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
    if not data_path.exists():
        pytest.skip("train.jsonl not found")

    datasets = build_dataset(
        data_path=str(data_path), fmt="alpaca", eval_split=0.2, seed=42
    )
    assert "train" in datasets
    assert "eval" in datasets
    assert len(datasets["train"]) > 0
    assert len(datasets["eval"]) > 0
    # Each example should have formatted text
    assert "text" in datasets["train"][0]
    assert "### Instruction:" in datasets["train"][0]["text"]


# ── API tests (mocked model — no actual model loading) ─────────────
@pytest.fixture
def client_with_mock_model():
    """TestClient with model state pre-populated to avoid loading real models."""
    import app.main as main_module
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        # Set AFTER lifespan startup so our mock is not overwritten
        main_module._state["model"] = MagicMock()
        main_module._state["tokeniser"] = MagicMock()
        main_module._state["mode"] = "adapter"
        yield client

    main_module._state["model"] = None
    main_module._state["tokeniser"] = None
    main_module._state["mode"] = "none"


def test_health_no_model():
    """When no model is loaded, health should report no_model_loaded."""
    import app.main as main_module
    from fastapi.testclient import TestClient

    main_module._state["model"] = None
    main_module._state["mode"] = "none"

    with TestClient(main_module.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "no_model_loaded"


def test_info_endpoint():
    import app.main as main_module
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert data["lora_r"] == settings.LORA_R
        assert "target_modules" in data


def test_generate_no_model_returns_503():
    import app.main as main_module
    from fastapi.testclient import TestClient

    main_module._state["model"] = None
    main_module._state["mode"] = "none"

    with TestClient(main_module.app) as client:
        r = client.post("/generate", json={
            "instruction": "Classify urgency.",
            "input": "Server down",
        })
        assert r.status_code == 503


def test_generate_empty_instruction(client_with_mock_model):
    r = client_with_mock_model.post("/generate", json={
        "instruction": "",
        "input": "test",
    })
    assert r.status_code == 400


@patch("app.evaluation.generate_response", return_value="High urgency - escalate immediately.")
def test_generate_success(mock_gen, client_with_mock_model):
    r = client_with_mock_model.post("/generate", json={
        "instruction": "Classify the urgency of this IT support ticket.",
        "input": "Production server is down.",
        "max_new_tokens": 50,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["response"] == "High urgency - escalate immediately."
    assert data["model_mode"] == "adapter"
