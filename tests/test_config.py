"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from qwen3_reranker.core.config import (
    AppConfig,
    ProfileConfig,
    ProfilesFile,
    ServiceSettings,
    clear_config_cache,
    load_profiles_yaml,
)
from qwen3_reranker.core.errors import ConfigurationError


@pytest.fixture
def sample_profiles_yaml() -> str:
    """Sample profiles YAML content."""
    return """
profiles:
  test_profile:
    description: "Test profile"
    backend: "pytorch"
    model_id: "test/model"
    scoring:
      method: "yes_no_next_token_prob"
      yes_token: "yes"
      no_token: "no"
      prefix: "<|im_start|>system\\nTest prefix<|im_end|>\\n"
      query_template: "<Query>: {query}\\n<Document>: {doc}"
      suffix: "<|im_end|>\\n<|im_start|>assistant\\n"
    limits:
      max_length: 4096
      max_length_hard_cap: 8192
      max_docs_per_request: 100
      max_query_chars: 5000
      max_doc_chars: 10000
    batching:
      batch_size: 8
      max_concurrent_forwards: 1
    defaults:
      instruction: "Test instruction"

  another_profile:
    description: "Another test profile"
    backend: "mlx"
    model_id: "test/other-model"
    scoring:
      method: "yes_no_next_token_prob"
      yes_token: "yes"
      no_token: "no"
      prefix: "prefix"
      query_template: "{query} {doc}"
      suffix: "suffix"
    limits:
      max_length: 2048
      max_length_hard_cap: 4096
      max_docs_per_request: 50
      max_query_chars: 2000
      max_doc_chars: 5000
    batching:
      batch_size: 4
      max_concurrent_forwards: 2
    defaults:
      instruction: "Another instruction"
"""


@pytest.fixture
def temp_config_dir(sample_profiles_yaml: str):
    """Create a temporary config directory with profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        profiles_path = config_dir / "reranker_profiles.yaml"
        profiles_path.write_text(sample_profiles_yaml)
        yield config_dir


class TestProfilesFile:
    """Tests for profiles YAML loading."""

    def test_load_profiles_success(self, temp_config_dir: Path) -> None:
        """Test successful profiles loading."""
        profiles = load_profiles_yaml(temp_config_dir)

        assert isinstance(profiles, ProfilesFile)
        assert "test_profile" in profiles.profiles
        assert "another_profile" in profiles.profiles

    def test_load_profiles_content(self, temp_config_dir: Path) -> None:
        """Test that profile content is loaded correctly."""
        profiles = load_profiles_yaml(temp_config_dir)
        profile = profiles.profiles["test_profile"]

        assert profile.description == "Test profile"
        assert profile.model_id == "test/model"
        assert profile.scoring.method == "yes_no_next_token_prob"
        assert profile.scoring.yes_token == "yes"
        assert profile.limits.max_length == 4096
        assert profile.batching.batch_size == 8
        assert profile.defaults.instruction == "Test instruction"

    def test_load_profiles_missing_file(self) -> None:
        """Test error when profiles file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ConfigurationError) as exc_info:
                load_profiles_yaml(Path(tmpdir))

            assert "not found" in str(exc_info.value.message).lower()

    def test_load_profiles_invalid_yaml(self) -> None:
        """Test error when YAML is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            profiles_path = config_dir / "reranker_profiles.yaml"
            profiles_path.write_text("invalid: yaml: content: [")

            with pytest.raises(ConfigurationError) as exc_info:
                load_profiles_yaml(config_dir)

            assert "parse" in str(exc_info.value.message).lower()


class TestServiceSettings:
    """Tests for ServiceSettings model."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        # Clear any env vars that might interfere
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            settings = ServiceSettings()

            assert settings.profile == "qwen3_4b_cuda"
            assert settings.port == 9003
            assert settings.host == "0.0.0.0"
            assert settings.log_level == "INFO"
            assert settings.log_format == "json"
            assert settings.backend == "auto"
        finally:
            # Restore env vars
            os.environ.update(env_backup)

    def test_settings_from_env(self) -> None:
        """Test settings loaded from environment."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            os.environ["QWEN_RERANK_PROFILE"] = "custom_profile"
            os.environ["QWEN_RERANK_PORT"] = "8080"
            os.environ["QWEN_RERANK_LOG_LEVEL"] = "DEBUG"

            settings = ServiceSettings()

            assert settings.profile == "custom_profile"
            assert settings.port == 8080
            assert settings.log_level == "DEBUG"
        finally:
            # Restore env vars
            for key in ["QWEN_RERANK_PROFILE", "QWEN_RERANK_PORT", "QWEN_RERANK_LOG_LEVEL"]:
                os.environ.pop(key, None)
            os.environ.update(env_backup)

    def test_log_level_validation(self) -> None:
        """Test log level validation."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            os.environ["QWEN_RERANK_LOG_LEVEL"] = "INVALID"

            with pytest.raises(ValueError):
                ServiceSettings()
        finally:
            os.environ.pop("QWEN_RERANK_LOG_LEVEL", None)
            os.environ.update(env_backup)

    def test_model_aliases_parsing(self) -> None:
        """Test model alias allowlist parsing."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            os.environ["QWEN_RERANK_MODEL_ALIAS_ALLOWLIST"] = "model1, model2, model3"

            settings = ServiceSettings()
            aliases = settings.get_model_aliases()

            assert aliases is not None
            assert "model1" in aliases
            assert "model2" in aliases
            assert "model3" in aliases
        finally:
            os.environ.pop("QWEN_RERANK_MODEL_ALIAS_ALLOWLIST", None)
            os.environ.update(env_backup)

    def test_model_aliases_empty(self) -> None:
        """Test empty model alias allowlist."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            settings = ServiceSettings()
            aliases = settings.get_model_aliases()

            assert aliases is None
        finally:
            os.environ.update(env_backup)


class TestAppConfig:
    """Tests for AppConfig computed properties."""

    @pytest.fixture
    def sample_app_config(self, temp_config_dir: Path) -> AppConfig:
        """Create a sample AppConfig."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            profiles = load_profiles_yaml(temp_config_dir)
            settings = ServiceSettings(profile="test_profile")
            profile = profiles.profiles["test_profile"]

            return AppConfig(
                settings=settings,
                profile=profile,
                profile_name="test_profile",
            )
        finally:
            os.environ.update(env_backup)

    def test_model_id_from_profile(self, sample_app_config: AppConfig) -> None:
        """Test model_id from profile."""
        assert sample_app_config.model_id == "test/model"

    def test_model_id_override(self, temp_config_dir: Path) -> None:
        """Test model_id override from settings."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            profiles = load_profiles_yaml(temp_config_dir)
            settings = ServiceSettings(
                profile="test_profile",
                model_id="override/model",
            )
            profile = profiles.profiles["test_profile"]

            config = AppConfig(
                settings=settings,
                profile=profile,
                profile_name="test_profile",
            )

            assert config.model_id == "override/model"
        finally:
            os.environ.update(env_backup)

    def test_max_length_from_profile(self, sample_app_config: AppConfig) -> None:
        """Test max_length from profile."""
        assert sample_app_config.max_length == 4096

    def test_max_length_capped(self, temp_config_dir: Path) -> None:
        """Test max_length is capped at hard limit."""
        env_backup = {}
        for key in list(os.environ.keys()):
            if key.startswith("QWEN_RERANK_"):
                env_backup[key] = os.environ.pop(key)

        try:
            profiles = load_profiles_yaml(temp_config_dir)
            settings = ServiceSettings(
                profile="test_profile",
                max_length=100000,  # Way above hard cap
            )
            profile = profiles.profiles["test_profile"]

            config = AppConfig(
                settings=settings,
                profile=profile,
                profile_name="test_profile",
            )

            # Should be capped at hard cap (8192)
            assert config.max_length == 8192
        finally:
            os.environ.update(env_backup)

    def test_batch_size_from_profile(self, sample_app_config: AppConfig) -> None:
        """Test batch_size from profile."""
        assert sample_app_config.batch_size == 8

    def test_max_concurrent_from_profile(self, sample_app_config: AppConfig) -> None:
        """Test max_concurrent_forwards from profile."""
        assert sample_app_config.max_concurrent_forwards == 1
