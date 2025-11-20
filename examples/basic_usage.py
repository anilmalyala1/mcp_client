"""
Basic usage example for the MCP client.

This example shows how to:
1. Set up logging
2. Initialize the client
3. Execute a simple query
4. Use the context manager for automatic cleanup
"""

import asyncio

from logging_config import setup_logging
from client_streamable_vertex_ai import MCPChatClient


async def main():
    """Basic usage example."""
    # Set up logging
    setup_logging(level="INFO", colored=True)

    # Method 1: Manual initialization and cleanup
    print("\n=== Method 1: Manual Management ===\n")
    client = MCPChatClient()
    await client.initialize()

    try:
        print("Available tools:", client.available_tools)
        print("\nExecuting query...")

        async for chunk in client.run("Show me recent errors in the logs"):
            print(chunk, end="", flush=True)

    finally:
        await client.close()

    # Method 2: Using context manager (recommended)
    print("\n\n=== Method 2: Context Manager (Recommended) ===\n")

    async with MCPChatClient() as client:
        print("Available tools:", client.available_tools)
        print("\nExecuting query...")

        async for chunk in client.run("Search for exceptions in the last 24 hours"):
            print(chunk, end="", flush=True)

    print("\n\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
