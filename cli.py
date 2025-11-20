from __future__ import annotations

import asyncio
import logging
import os

from client_streamable_vertex_ai import run

logger = logging.getLogger(__name__)


async def _prompt_user(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def _interactive_loop(default_query: str) -> None:
    current_default = default_query
    while True:
        try:
            raw_query = (await _prompt_user(f"Enter your query (or 'exit' to quit) [{current_default}]: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        user_query = raw_query or current_default
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting.")
            return

        try:
            final_answer = await run(user_query)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to process query %r", user_query)
            continue

        print("Final Result:", final_answer)
        current_default = user_query


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    default_query = os.getenv("MCP_SAMPLE_QUERY", "Get all books from my_documents index")
    try:
        asyncio.run(_interactive_loop(default_query))
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
