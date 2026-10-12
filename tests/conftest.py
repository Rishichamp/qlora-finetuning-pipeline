"""
Pytest configuration for the fine-tuning pipeline tests.

Injects a stub `torch` module into sys.modules if torch is not installed.
This allows app.evaluation and app.model_setup to be imported (and their
functions patched/mocked) in CI environments without GPU/torch — these
tests verify API contracts and data logic, not actual model behaviour.

When torch IS installed (e.g. on a GPU dev machine), this is a no-op.
"""
import sys
import importlib.machinery
from unittest.mock import MagicMock

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        fake_torch = MagicMock()
        # datasets/transformers call importlib.util.find_spec("torch") to check
        # availability — a bare MagicMock has no __spec__, which raises ValueError.
        fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        # datasets does isinstance(obj, torch.Tensor) — needs a real type, not a MagicMock.
        fake_torch.Tensor = type("Tensor", (), {})
        sys.modules["torch"] = fake_torch
