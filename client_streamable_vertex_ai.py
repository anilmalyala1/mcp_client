"""
MCP client that uses the Streamable HTTP transport and Vertex AI via LangChain.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import cast
from collections.abc import Sequence

from dotenv import find_dotenv, load_dotenv
#from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI as ChatVertexAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


load_dotenv(find_dotenv())


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"{key} environment variable is required")
    return value


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.getenv("MCP_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _render_content_blocks(blocks: Sequence[types.ContentBlock]) -> str:
    rendered: list[str] = []
    for block in blocks:
        if isinstance(block, types.TextContent):
            rendered.append(block.text)
        else:
            # Fall back to JSON for non-text payloads (resources, blobs, etc.).
            try:
                rendered.append(json.dumps(block.model_dump(), default=str))
            except Exception:  # noqa: BLE001
                rendered.append(str(block))
    return "\n".join(part for part in rendered if part)


def _ai_content_to_str(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "text":
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                parts.append(json.dumps(item))
        else:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _build_llm() -> ChatVertexAI:
   # project = _require_env("VERTEXAI_PROJECT")
   # location = _require_env("VERTEXAI_LOCATION")
    model_name = os.getenv("VERTEXAI_MODEL", "gemini-2.5-pro")
    temperature = float(os.getenv("VERTEXAI_TEMPERATURE", "0.1"))
    return ChatVertexAI(
        model=model_name,
        api_key=os.getenv("GOOGLE_API_KEY"),
        
        temperature=temperature,
        convert_system_message_to_human=True,
    )

llm = _build_llm()
def _convert_tools(tools: Sequence[types.Tool]) -> list[dict[str, object]]:
    def _sanitize_schema(schema: dict[str, object] | list[object] | object) -> dict[str, object] | list[object] | object:
        """Ensure schemas use constructs accepted by Vertex function calling."""
        if isinstance(schema, list):
            return [_sanitize_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema

        sanitized: dict[str, object] = {
            key: _sanitize_schema(value)  # recurse into nested structures
            for key, value in schema.items()
        }

        schema_type = sanitized.get("type")

        def _type_in(value: object, expected: str) -> bool:
            if value == expected:
                return True
            if isinstance(value, list) and expected in value:
                return True
            return False

        if _type_in(schema_type, "object"):
            properties = sanitized.get("properties")
            if isinstance(properties, dict):
                sanitized["properties"] = {
                    key: _sanitize_schema(value)
                    for key, value in properties.items()
                }
        if _type_in(schema_type, "array"):
            items = sanitized.get("items")
            if isinstance(items, dict):
                sanitized["items"] = _sanitize_schema(items)
            elif isinstance(items, list):
                sanitized["items"] = _sanitize_schema(items[0]) if items else {"type": "string"}
            elif isinstance(items, (str, int, float, bool)) or items is None:
                sanitized["items"] = {"type": "string"}
            else:
                sanitized["items"] = {"type": "string"}

        for composite_key in ("anyOf", "oneOf", "allOf"):
            variants = sanitized.get(composite_key)
            if isinstance(variants, list):
                sanitized[composite_key] = [
                    _sanitize_schema(variant)
                    for variant in variants
                ]
                if "items" not in sanitized:
                    array_variants = [
                        variant for variant in sanitized[composite_key]
                        if isinstance(variant, dict) and _type_in(variant.get("type"), "array")
                    ]
                    if array_variants:
                        first_array = array_variants[0]
                        items_schema = first_array.get("items")
                        if isinstance(items_schema, dict):
                            sanitized["items"] = _sanitize_schema(items_schema)
                        elif isinstance(items_schema, list):
                            sanitized["items"] = _sanitize_schema(items_schema[0]) if items_schema else {"type": "string"}
                        elif items_schema is None:
                            sanitized["items"] = {"type": "string"}
                        if "type" not in sanitized:
                            sanitized["type"] = "array"

        for single_key in ("not", "if", "then", "else"):
            branch = sanitized.get(single_key)
            if isinstance(branch, dict):
                sanitized[single_key] = _sanitize_schema(branch)

        return sanitized

    tool_schemas: list[dict[str, object]] = []
    for tool in tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        schema = cast(dict[str, object], _sanitize_schema(schema))
        tool_schemas.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
                # Provide debug visibility into the final schema we hand to Gemini.
                # Useful when diagnosing schema compatibility issues.
            }
        )
        logger.debug("Tool %s schema passed to Vertex: %s", tool.name, json.dumps(schema))
    return tool_schemas


async def run(query: str) -> str:
    server_url = _require_env("MCP_SERVER_URL")
   

    headers = _build_headers()

    async with streamablehttp_client(server_url, headers=headers) as (read, write, get_session_id):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            logger.info("Connected to MCP server (protocol %s)", init_result.protocolVersion)
            session_id = get_session_id()
            if session_id:
                logger.info("Using MCP session id %s", session_id)

            tools_result = await session.list_tools()
            tool_schemas = _convert_tools(tools_result.tools)
            if tool_schemas:
                logger.debug("Binding %d tools for Vertex AI function calling", len(tool_schemas))
                llm_with_tools = llm.bind_tools(tool_schemas)
            else:
                logger.warning("Server did not expose any tools; proceeding without tool binding.")
                llm_with_tools = llm

            messages: list[HumanMessage | AIMessage | ToolMessage] = [HumanMessage(content=query)]

            response = await llm_with_tools.ainvoke(messages)
            while True:
                messages.append(response)

                if not response.tool_calls:
                    return _ai_content_to_str(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    call_id = tool_call.get("id") or tool_name
                    arguments = tool_call.get("args") or {}

                    logger.debug("Executing tool %s with args %s", tool_name, arguments)
                    result = await session.call_tool(tool_name, arguments=arguments)
                    payload = _render_content_blocks(result.content)
                    messages.append(
                        ToolMessage(
                            content=payload or "(no content returned)",
                            tool_call_id=call_id,
                            name=tool_name,
                        )
                    )

                response = await llm_with_tools.ainvoke(messages)


def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
    default_query = os.getenv("MCP_SAMPLE_QUERY", "Get all books from my_documents index")
   
    while True:
        try:
            raw_query = input(f"Enter your query (or 'exit' to quit) [{default_query}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        user_query = raw_query or default_query
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting.")
            return

        try:
            final_answer =  asyncio.run(run(user_query))
            
        except Exception:  # noqa: BLE001
            logger.exception("Failed to process query %r", user_query)
            continue

        print("Final Result:", final_answer)
        default_query = user_query


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
