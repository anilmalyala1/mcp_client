"""
Factory for creating LLM clients.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI as ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from constants import (
    DEFAULT_VERTEXAI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_OLLAMA_BASE_URL,
)
from exceptions import ConfigurationError, LLMError
from validation import validate_environment_variable, validate_temperature

logger = logging.getLogger(__name__)

LLMProvider = Literal["openai", "vertex", "ollama"]

class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_llm(provider: LLMProvider | None = None) -> BaseChatModel:
        """
        Create an LLM client based on the provider.

        Args:
            provider: The LLM provider to use. If None, tries to determine from env.

        Returns:
            Configured LLM client.

        Raises:
            ConfigurationError: If configuration is invalid.
            LLMError: If LLM initialization fails.
        """
        if not provider:
            provider = os.getenv("LLM_PROVIDER", "openai") # Default to openai if not set

        logger.info("Creating LLM for provider: %s", provider)

        if provider == "openai":
            return LLMFactory._build_openai()
        elif provider == "vertex":
            return LLMFactory._build_vertex()
        elif provider == "ollama":
            return LLMFactory._build_ollama()
        else:
            raise ConfigurationError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _build_openai() -> ChatOpenAI:
        """Build an OpenAI LLM client with validated configuration."""
        model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

        try:
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
            validate_temperature(temperature)
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid OPENAI_TEMPERATURE value: {os.getenv('OPENAI_TEMPERATURE')}",
                config_key="OPENAI_TEMPERATURE",
            ) from e

        api_key = validate_environment_variable("OPENAI_API_KEY", required=True)

        logger.info("Building OpenAI LLM with model=%s, temperature=%s", model_name, temperature)

        try:
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize OpenAI LLM: {str(e)}",
                model=model_name,
                reason=str(e),
            ) from e

    @staticmethod
    def _build_vertex() -> ChatVertexAI:
        """Build a Vertex AI LLM client with validated configuration."""
        model_name = os.getenv("VERTEXAI_MODEL", DEFAULT_VERTEXAI_MODEL)

        try:
            temperature = float(os.getenv("VERTEXAI_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
            validate_temperature(temperature)
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid VERTEXAI_TEMPERATURE value: {os.getenv('VERTEXAI_TEMPERATURE')}",
                config_key="VERTEXAI_TEMPERATURE",
            ) from e

        api_key = validate_environment_variable("GOOGLE_API_KEY", required=True)

        logger.info("Building Vertex AI LLM with model=%s, temperature=%s", model_name, temperature)

        try:
            return ChatVertexAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                convert_system_message_to_human=True,
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize Vertex AI LLM: {str(e)}",
                model=model_name,
                reason=str(e),
            ) from e

    @staticmethod
    def _build_ollama() -> ChatOllama:
        """Build an Ollama LLM client with validated configuration."""
        model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

        try:
            temperature = float(os.getenv("OLLAMA_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
            validate_temperature(temperature)
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid OLLAMA_TEMPERATURE value: {os.getenv('OLLAMA_TEMPERATURE')}",
                config_key="OLLAMA_TEMPERATURE",
            ) from e

        logger.info("Building Ollama LLM with model=%s, base_url=%s, temperature=%s", model_name, base_url, temperature)

        try:
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize Ollama LLM: {str(e)}",
                model=model_name,
                reason=str(e),
            ) from e
