"""
Configuration management for the MCP client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from constants import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_VERTEXAI_MODEL,
    MAX_PROMPT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_RETRY_MAX_WAIT,
)
from exceptions import ConfigurationError
from validation import validate_temperature, validate_token_limit


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers."""

    model_name: str
    temperature: float
    max_prompt_tokens: int = MAX_PROMPT_TOKENS

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_temperature(self.temperature)
        validate_token_limit(self.max_prompt_tokens)


@dataclass(frozen=True)
class VertexAIConfig(LLMConfig):
    """Configuration for Vertex AI (Google Gemini)."""

    api_key: str | None = None
    convert_system_message_to_human: bool = True

    def __post_init__(self) -> None:
        """Validate Vertex AI configuration."""
        super().__post_init__()
        if not self.api_key:
            raise ConfigurationError(
                "API key is required for Vertex AI",
                config_key="api_key"
            )

    @classmethod
    def from_env(cls) -> VertexAIConfig:
        """
        Load Vertex AI configuration from environment variables.

        Environment variables:
            VERTEXAI_MODEL: Model name (default: gemini-2.5-pro)
            VERTEXAI_TEMPERATURE: Temperature (default: 0.1)
            GOOGLE_API_KEY: API key (required)

        Returns:
            VertexAIConfig instance

        Raises:
            ConfigurationError: If required environment variables are missing
        """
        model_name = os.getenv("VERTEXAI_MODEL", DEFAULT_VERTEXAI_MODEL)

        try:
            temperature = float(os.getenv("VERTEXAI_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid VERTEXAI_TEMPERATURE: {os.getenv('VERTEXAI_TEMPERATURE')}",
                config_key="VERTEXAI_TEMPERATURE",
            ) from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY environment variable is required for Vertex AI",
                config_key="GOOGLE_API_KEY",
            )

        return cls(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )


@dataclass(frozen=True)
class OllamaConfig(LLMConfig):
    """Configuration for Ollama."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL

    @classmethod
    def from_env(cls) -> OllamaConfig:
        """
        Load Ollama configuration from environment variables.

        Environment variables:
            OLLAMA_MODEL: Model name (default: qwen3:4b)
            OLLAMA_TEMPERATURE: Temperature (default: 0.1)
            OLLAMA_BASE_URL: Base URL (default: http://localhost:11434)

        Returns:
            OllamaConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

        try:
            temperature = float(os.getenv("OLLAMA_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid OLLAMA_TEMPERATURE: {os.getenv('OLLAMA_TEMPERATURE')}",
                config_key="OLLAMA_TEMPERATURE",
            ) from e

        return cls(
            model_name=model_name,
            temperature=temperature,
            base_url=base_url,
        )


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = DEFAULT_MAX_RETRIES
    min_wait: float = DEFAULT_RETRY_MIN_WAIT
    max_wait: float = DEFAULT_RETRY_MAX_WAIT
    exponential_base: int = 2

    def __post_init__(self) -> None:
        """Validate retry configuration."""
        if self.max_retries < 0:
            raise ConfigurationError(
                f"max_retries must be non-negative, got {self.max_retries}",
                config_key="max_retries",
            )
        if self.min_wait < 0:
            raise ConfigurationError(
                f"min_wait must be non-negative, got {self.min_wait}",
                config_key="min_wait",
            )
        if self.max_wait < self.min_wait:
            raise ConfigurationError(
                f"max_wait ({self.max_wait}) must be >= min_wait ({self.min_wait})",
                config_key="max_wait",
            )

    @classmethod
    def from_env(cls) -> RetryConfig:
        """
        Load retry configuration from environment variables.

        Environment variables:
            MCP_MAX_RETRIES: Maximum retry attempts (default: 3)
            MCP_RETRY_MIN_WAIT: Minimum wait time in seconds (default: 1.0)
            MCP_RETRY_MAX_WAIT: Maximum wait time in seconds (default: 10.0)

        Returns:
            RetryConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            max_retries = int(os.getenv("MCP_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
            min_wait = float(os.getenv("MCP_RETRY_MIN_WAIT", str(DEFAULT_RETRY_MIN_WAIT)))
            max_wait = float(os.getenv("MCP_RETRY_MAX_WAIT", str(DEFAULT_RETRY_MAX_WAIT)))
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid retry configuration: {e}",
                config_key="retry_config",
            ) from e

        return cls(
            max_retries=max_retries,
            min_wait=min_wait,
            max_wait=max_wait,
        )


@dataclass
class MCPClientConfig:
    """Complete configuration for the MCP client."""

    llm_config: LLMConfig
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    max_prompt_tokens: int = MAX_PROMPT_TOKENS

    def __post_init__(self) -> None:
        """Validate configuration."""
        validate_token_limit(self.max_prompt_tokens)

    @classmethod
    def from_env(cls, provider: str = "vertexai") -> MCPClientConfig:
        """
        Load configuration from environment variables.

        Args:
            provider: LLM provider to use ("vertexai" or "ollama")

        Returns:
            MCPClientConfig instance

        Raises:
            ConfigurationError: If provider is unknown or config is invalid
        """
        if provider.lower() == "vertexai":
            llm_config = VertexAIConfig.from_env()
        elif provider.lower() == "ollama":
            llm_config = OllamaConfig.from_env()
        else:
            raise ConfigurationError(
                f"Unknown LLM provider: {provider}. Use 'vertexai' or 'ollama'",
                config_key="provider",
            )

        retry_config = RetryConfig.from_env()

        return cls(
            llm_config=llm_config,
            retry_config=retry_config,
        )
