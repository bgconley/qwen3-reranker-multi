"""Pytest configuration and fixtures."""

import os
import sys

import pytest

# Add src to path for imports
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def clean_env():
    """Clean up QWEN_RERANK_ environment variables before/after tests."""
    # Store original values
    original = {}
    for key in list(os.environ.keys()):
        if key.startswith("QWEN_RERANK_"):
            original[key] = os.environ.pop(key)

    yield

    # Restore original values
    for key in list(os.environ.keys()):
        if key.startswith("QWEN_RERANK_"):
            os.environ.pop(key)
    os.environ.update(original)


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir
