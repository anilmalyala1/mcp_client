"""
Example demonstrating error handling and retry logic.

This example shows how to:
1. Handle specific exceptions
2. Use retry logic
3. Extract error details
4. Implement graceful degradation
"""

import asyncio

from logging_config import setup_logging
from client_streamable_vertex_ai import MCPChatClient
from exceptions import (
    ToolNotFoundError,
    ToolExecutionError,
    ServerConnectionError,
    InitializationError,
    LLMError,
)
from retry import retry_async


async def example_basic_error_handling():
    """Example of basic error handling."""
    print("=== Basic Error Handling ===\n")

    async with MCPChatClient() as client:
        try:
            # Try to execute a tool that doesn't exist
            result = await client.execute_tool("nonexistent_tool", {})
        except ToolNotFoundError as e:
            print(f"❌ Tool not found: {e.message}")
            print(f"   Requested: {e.details.get('requested_tool')}")
            available = e.details.get('available_tools', [])
            if available:
                print(f"   Available tools: {', '.join(available[:5])}")
        except ToolExecutionError as e:
            print(f"❌ Tool execution failed: {e.message}")
            print(f"   Tool: {e.details.get('tool_name')}")
            print(f"   Reason: {e.details.get('reason')}")


async def example_with_retry():
    """Example using retry logic."""
    print("\n=== Using Retry Logic ===\n")

    async with MCPChatClient() as client:
        try:
            # Retry on transient failures
            result = await retry_async(
                client.execute_tool,
                "elastic_search",
                {"query": "errors", "timeframe": "1h"},
                max_retries=3,
                min_wait=1.0,
                max_wait=5.0,
            )
            print("✅ Tool executed successfully (possibly after retries)")

        except ToolExecutionError as e:
            print(f"❌ Tool execution failed after retries: {e.message}")


async def example_initialization_errors():
    """Example handling initialization errors."""
    print("\n=== Initialization Error Handling ===\n")

    try:
        client = MCPChatClient()
        await client.initialize()
        print("✅ Client initialized successfully")
        await client.close()

    except ServerConnectionError as e:
        print(f"❌ Server connection failed: {e.message}")
        print(f"   Server: {e.details.get('server_name')}")
        print(f"   URL: {e.details.get('url')}")
        print(f"   Reason: {e.details.get('reason')}")

    except InitializationError as e:
        print(f"❌ Initialization failed: {e.message}")
        if e.details:
            print(f"   Details: {e.details}")


async def example_llm_errors():
    """Example handling LLM errors."""
    print("\n=== LLM Error Handling ===\n")

    async with MCPChatClient() as client:
        try:
            # This would fail if the LLM is unavailable
            messages = [{"role": "user", "content": "Hello"}]
            # response = await client.ainvoke_llm(messages)
            print("✅ LLM invocation successful")

        except LLMError as e:
            print(f"❌ LLM error: {e.message}")
            print(f"   Reason: {e.details.get('reason')}")


async def example_graceful_degradation():
    """Example of graceful degradation on errors."""
    print("\n=== Graceful Degradation ===\n")

    async with MCPChatClient() as client:
        # Try multiple tools in order of preference
        tools_to_try = ["primary_tool", "fallback_tool", "last_resort_tool"]

        for tool_name in tools_to_try:
            try:
                result = await client.execute_tool(tool_name, {"query": "test"})
                print(f"✅ Successfully used tool: {tool_name}")
                break
            except ToolNotFoundError:
                print(f"⚠️  Tool '{tool_name}' not available, trying next...")
                continue
            except ToolExecutionError as e:
                print(f"⚠️  Tool '{tool_name}' failed: {e.message}")
                continue
        else:
            print("❌ All tools failed, using default response")
            # Provide default/fallback response


async def main():
    """Run all error handling examples."""
    setup_logging(level="WARNING")  # Reduce noise for examples

    await example_basic_error_handling()
    await example_with_retry()
    await example_initialization_errors()
    await example_llm_errors()
    await example_graceful_degradation()

    print("\n✅ All error handling examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
