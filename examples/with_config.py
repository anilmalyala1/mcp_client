"""
Example using configuration management.

This example shows how to:
1. Use configuration classes
2. Configure retry logic
3. Use different LLM providers
"""

import asyncio

from logging_config import setup_logging
from config import MCPClientConfig, VertexAIConfig, OllamaConfig, RetryConfig
from client_streamable_vertex_ai import MCPChatClient


async def example_with_custom_config():
    """Example using custom configuration."""
    print("=== Using Custom Configuration ===\n")

    # Load configuration from environment
    config = MCPClientConfig.from_env(provider="vertexai")

    print(f"LLM Model: {config.llm_config.model_name}")
    print(f"Temperature: {config.llm_config.temperature}")
    print(f"Max Retries: {config.retry_config.max_retries}")

    # Create client (would need to accept config parameter)
    # This is an example of how it could work
    print("\n✅ Configuration loaded successfully")


async def example_different_providers():
    """Example showing different provider configs."""
    print("\n=== Different LLM Providers ===\n")

    # Vertex AI configuration
    try:
        vertex_config = VertexAIConfig.from_env()
        print(f"✅ Vertex AI: {vertex_config.model_name}")
    except Exception as e:
        print(f"❌ Vertex AI not configured: {e}")

    # Ollama configuration
    try:
        ollama_config = OllamaConfig.from_env()
        print(f"✅ Ollama: {ollama_config.model_name} at {ollama_config.base_url}")
    except Exception as e:
        print(f"❌ Ollama not configured: {e}")


async def example_retry_config():
    """Example showing retry configuration."""
    print("\n=== Retry Configuration ===\n")

    # Custom retry config
    retry_config = RetryConfig(
        max_retries=5,
        min_wait=2.0,
        max_wait=30.0,
    )

    print(f"Max Retries: {retry_config.max_retries}")
    print(f"Min Wait: {retry_config.min_wait}s")
    print(f"Max Wait: {retry_config.max_wait}s")

    # Load from environment
    env_retry_config = RetryConfig.from_env()
    print(f"\nFrom env - Max Retries: {env_retry_config.max_retries}")


async def main():
    """Run all configuration examples."""
    setup_logging(level="INFO")

    await example_with_custom_config()
    await example_different_providers()
    await example_retry_config()

    print("\n✅ All configuration examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
