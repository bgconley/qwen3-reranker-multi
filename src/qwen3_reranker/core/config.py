"""Configuration management for Qwen3 Reranker Service.

Loads configuration from:
1. config/reranker_profiles.yaml (profile definitions)
2. Environment variables (override any setting)

Supports multiple backends:
- PyTorch (PRIMARY): CUDA/MPS/CPU cross-platform
- vLLM (SECONDARY): High-throughput CUDA
- MLX (TERTIARY): Apple Silicon optimization
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qwen3_reranker.core.errors import ConfigurationError


def get_config_dir() -> Path:
    """Get the config directory path."""
    # Check for explicit config dir env var
    config_dir = os.environ.get("QWEN_RERANK_CONFIG_DIR")
    if config_dir:
        return Path(config_dir)

    # Default: look relative to package or cwd
    # First try relative to this file (package install)
    pkg_config = Path(__file__).parent.parent.parent.parent / "config"
    if pkg_config.exists():
        return pkg_config

    # Fall back to cwd/config
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    raise ConfigurationError(
        "Config directory not found",
        {"searched": [str(pkg_config), str(cwd_config)]},
    )


class ScoringConfig(BaseModel):
    """Scoring method configuration."""

    method: str = "yes_no_next_token_prob"
    yes_token: str = "yes"
    no_token: str = "no"
    prefix: str
    query_template: str
    suffix: str


class LimitsConfig(BaseModel):
    """Request and processing limits."""

    max_length: int = 4096
    max_length_hard_cap: int = 8192
    max_docs_per_request: int = 200
    max_query_chars: int = 8000
    max_doc_chars: int = 20000


class BatchingConfig(BaseModel):
    """Batching and concurrency settings."""

    batch_size: int = 8
    max_concurrent_forwards: int = 1


class DefaultsConfig(BaseModel):
    """Default values for optional request fields."""

    instruction: str = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )


class PyTorchOptions(BaseModel):
    """PyTorch-specific options."""

    device: str = "auto"  # auto | cuda | cuda:N | mps | cpu
    dtype: str = "float16"
    use_flash_attn: bool = True


class VLLMOptions(BaseModel):
    """vLLM-specific options."""

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    enable_prefix_caching: bool = True
    max_model_len: int | None = None


class MLXOptions(BaseModel):
    """MLX-specific options."""

    compile: bool = True  # Use mx.compile() for JIT optimization


class ProfileConfig(BaseModel):
    """A single reranker profile configuration."""

    description: str
    backend: str = "pytorch"  # pytorch | vllm | mlx
    model_id: str
    scoring: ScoringConfig
    limits: LimitsConfig
    batching: BatchingConfig
    defaults: DefaultsConfig
    pytorch_options: PyTorchOptions | None = None
    vllm_options: VLLMOptions | None = None
    mlx_options: MLXOptions | None = None
    hf_tokenizer_id: str | None = None  # Override tokenizer (for MLX)


class ProfilesFile(BaseModel):
    """Root structure of reranker_profiles.yaml."""

    profiles: dict[str, ProfileConfig]


class ServiceSettings(BaseSettings):
    """Service-level settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="QWEN_RERANK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend and profile selection
    backend: str = Field(
        default="auto", description="Backend: auto | pytorch | vllm | mlx"
    )
    profile: str = Field(default="qwen3_4b_cuda", description="Profile name to load")

    # Service binding
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: int = Field(default=9003, description="Port to bind to")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format: json or console")

    # Overrides (optional, override profile values)
    model_id: str | None = Field(default=None, description="Override profile model_id")
    max_length: int | None = Field(default=None, description="Override max_length")
    batch_size: int | None = Field(default=None, description="Override batch_size")
    max_concurrent_forwards: int | None = Field(
        default=None, description="Override max_concurrent_forwards"
    )

    # PyTorch device override
    device: str | None = Field(
        default=None, description="PyTorch device: auto | cuda | mps | cpu"
    )

    # vLLM overrides
    tensor_parallel_size: int | None = Field(
        default=None, description="vLLM tensor parallel size"
    )
    gpu_memory_utilization: float | None = Field(
        default=None, description="vLLM GPU memory utilization"
    )

    # Model alias allowlist (comma-separated)
    model_alias_allowlist: str | None = Field(
        default=None, description="Comma-separated list of allowed model aliases"
    )

    # Correlation header name
    correlation_header: str = Field(
        default="X-Correlation-Id", description="Correlation ID header"
    )

    # Timeouts
    request_timeout: float = Field(
        default=120.0, description="Request timeout in seconds"
    )
    warmup_timeout: float = Field(
        default=300.0, description="Warmup timeout in seconds"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        valid = {"json", "console"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(f"log_format must be one of {valid}")
        return lower

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        valid = {"auto", "pytorch", "vllm", "mlx"}
        lower = v.lower()
        if lower not in valid:
            raise ValueError(f"backend must be one of {valid}")
        return lower

    def get_model_aliases(self) -> set[str] | None:
        """Parse model_alias_allowlist into a set."""
        if not self.model_alias_allowlist:
            return None
        return {
            alias.strip()
            for alias in self.model_alias_allowlist.split(",")
            if alias.strip()
        }


class AppConfig(BaseModel):
    """Complete application configuration."""

    settings: ServiceSettings
    profile: ProfileConfig
    profile_name: str

    @property
    def model_id(self) -> str:
        """Get effective model_id (with override)."""
        return self.settings.model_id or self.profile.model_id

    @property
    def max_length(self) -> int:
        """Get effective max_length (with override and cap)."""
        requested = self.settings.max_length or self.profile.limits.max_length
        return min(requested, self.profile.limits.max_length_hard_cap)

    @property
    def batch_size(self) -> int:
        """Get effective batch_size (with override)."""
        return self.settings.batch_size or self.profile.batching.batch_size

    @property
    def max_concurrent_forwards(self) -> int:
        """Get effective max_concurrent_forwards (with override)."""
        return (
            self.settings.max_concurrent_forwards
            or self.profile.batching.max_concurrent_forwards
        )

    @property
    def backend(self) -> str:
        """Get effective backend (with override)."""
        return (
            self.settings.backend
            if self.settings.backend != "auto"
            else self.profile.backend
        )


def load_profiles_yaml(config_dir: Path) -> ProfilesFile:
    """Load and parse reranker_profiles.yaml."""
    profiles_path = config_dir / "reranker_profiles.yaml"
    if not profiles_path.exists():
        raise ConfigurationError(
            "Profiles file not found",
            {"path": str(profiles_path)},
        )

    try:
        with open(profiles_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            "Failed to parse profiles YAML",
            {"path": str(profiles_path), "error": str(e)},
        ) from e

    try:
        return ProfilesFile.model_validate(data)
    except Exception as e:
        raise ConfigurationError(
            "Invalid profiles configuration",
            {"error": str(e)},
        ) from e


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and return the complete application configuration.

    This function is cached to ensure the config is only loaded once.
    """
    config_dir = get_config_dir()

    # Load profiles
    profiles_file = load_profiles_yaml(config_dir)

    # Load service settings from environment
    settings = ServiceSettings()

    # Get the requested profile
    profile_name = settings.profile
    if profile_name not in profiles_file.profiles:
        available = list(profiles_file.profiles.keys())
        raise ConfigurationError(
            f"Profile '{profile_name}' not found",
            {"available_profiles": available},
        )

    profile = profiles_file.profiles[profile_name]

    return AppConfig(
        settings=settings,
        profile=profile,
        profile_name=profile_name,
    )


def clear_config_cache() -> None:
    """Clear the cached configuration (for testing)."""
    get_config.cache_clear()
