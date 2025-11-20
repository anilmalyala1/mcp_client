"""
Constants used throughout the MCP client.
"""

from __future__ import annotations

# Token limits
MAX_PROMPT_TOKENS = 6000
"""Maximum number of tokens allowed in the prompt history."""

FALLBACK_MAX_MESSAGES = 15
"""Fallback maximum number of messages when token counting fails."""

# Data truncation limits
MAX_ERROR_DATA_LENGTH = 500
"""Maximum length of error data samples in error messages."""

MAX_SAMPLE_DATA_LENGTH = 200
"""Maximum length of data samples for parsing errors."""

# LLM configuration defaults
DEFAULT_VERTEXAI_MODEL = "gemini-2.5-pro"
"""Default Vertex AI model name."""

DEFAULT_OLLAMA_MODEL = "qwen3:4b"
"""Default Ollama model name."""

DEFAULT_TEMPERATURE = 0.1
"""Default temperature for LLM generation."""

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
"""Default Ollama base URL."""

# Temperature bounds
MIN_TEMPERATURE = 0.0
"""Minimum allowed temperature value."""

MAX_TEMPERATURE = 2.0
"""Maximum allowed temperature value."""

# Token limit bounds
MIN_TOKEN_LIMIT = 1
"""Minimum allowed token limit."""

MAX_TOKEN_LIMIT = 1_000_000
"""Maximum reasonable token limit."""

# MCP configuration
DEFAULT_MCP_SERVER_NAME = "default"
"""Default MCP server name for legacy configuration."""

# Retry configuration
DEFAULT_MAX_RETRIES = 3
"""Default maximum number of retry attempts."""

DEFAULT_RETRY_MIN_WAIT = 1.0
"""Default minimum wait time between retries (seconds)."""

DEFAULT_RETRY_MAX_WAIT = 10.0
"""Default maximum wait time between retries (seconds)."""

# Timeout configuration
DEFAULT_COMMAND_TIMEOUT = 120_000
"""Default command timeout in milliseconds (2 minutes)."""

DEFAULT_LONG_COMMAND_TIMEOUT = 600_000
"""Default timeout for long-running commands in milliseconds (10 minutes)."""

# Logging configuration
DEFAULT_LOG_LEVEL = "INFO"
"""Default logging level."""

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""Default log format string."""

# Agent configuration
AGENT_STEP_TOTAL = 3
"""Total number of steps in the LogAnalysisAgent workflow."""

# Tool configuration
ELASTIC_SEARCH_TOOL = "elastic_search"
"""Name of the Elasticsearch tool."""

# Encoding
DEFAULT_ENCODING = "cl100k_base"
"""Default tiktoken encoding to use for token counting."""
