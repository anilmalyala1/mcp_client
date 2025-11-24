"""
Core MCP client that uses the Streamable HTTP transport and Vertex AI via LangChain.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence
from textwrap import shorten

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage, BaseMessageChunk
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI as ChatVertexAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import tiktoken
from tiktoken import Encoding

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from parsers import get_parser
from utils import ai_content_to_str, get_token_count
from constants import (
    MAX_PROMPT_TOKENS,
    DEFAULT_VERTEXAI_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_MCP_SERVER_NAME,
    FALLBACK_MAX_MESSAGES,
    DEFAULT_ENCODING,
)
from exceptions import (
    InitializationError,
    ServerConnectionError,
    ToolNotFoundError,
    ToolExecutionError,
    ConfigurationError,
    ValidationError,
    LLMError,
    TokenLimitError,
    ParsingError,
)
from validation import (
    validate_server_config,
    validate_environment_variable,
    validate_temperature,
    validate_token_limit,
    validate_tool_arguments,
)
from llm_factory import LLMFactory, LLMProvider

logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())


@dataclass(frozen=True)
class MCPServerConfig:
    """Minimal configuration needed to connect to an MCP server."""

    name: str
    url: str


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
            try:
                rendered.append(json.dumps(block.model_dump(), default=str))
            except Exception:  # noqa: BLE001
                rendered.append(str(block))
    return "\n".join(part for part in rendered if part)






def _parse_server_mapping_fragment(fragment: str) -> MCPServerConfig:
    """Parse a server config fragment in 'name=url' format."""
    if "=" not in fragment:
        raise ConfigurationError(
            f"Invalid MCP_SERVERS entry: {fragment!r}. Expected 'name=url' format.",
            config_key="MCP_SERVERS",
        )

    name, url = fragment.split("=", 1)
    name = name.strip()
    url = url.strip()

    try:
        validate_server_config(name, url)
    except ValidationError as e:
        raise ConfigurationError(
            f"Invalid server configuration in fragment '{fragment}': {e.message}",
            config_key="MCP_SERVERS",
        ) from e

    return MCPServerConfig(name=name, url=url)


def _load_server_configs() -> list[MCPServerConfig]:
    """
    Load MCP server configurations from the environment.

    Supported formats:
    - MCP_SERVERS='[{"name": "...", "url": "..."}]'
    - MCP_SERVERS='name1=url1,name2=url2'
    - MCP_SERVERS='name1=url1;name2=url2'
    - Legacy single server fallback via MCP_SERVER_URL (+ optional MCP_SERVER_NAME)
    """
    env_value = os.getenv("MCP_SERVERS")
    logger.info("Loading MCP server configurations")

    if env_value:
        env_value = env_value.strip()
        logger.debug("MCP_SERVERS value: %s", env_value[:100])

        try:
            parsed = json.loads(env_value)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            configs: list[MCPServerConfig] = []
            for i, entry in enumerate(parsed):
                if not isinstance(entry, dict):
                    raise ConfigurationError(
                        f"Entry {i} in MCP_SERVERS JSON must be an object, got {type(entry).__name__}",
                        config_key="MCP_SERVERS",
                    )
                name = entry.get("name")
                url = entry.get("url")
                if not name or not url:
                    raise ConfigurationError(
                        f"Entry {i} in MCP_SERVERS is missing 'name' or 'url'",
                        config_key="MCP_SERVERS",
                    )

                try:
                    validate_server_config(str(name), str(url))
                except ValidationError as e:
                    raise ConfigurationError(
                        f"Invalid server config at entry {i}: {e.message}",
                        config_key="MCP_SERVERS",
                    ) from e

                configs.append(MCPServerConfig(name=str(name), url=str(url)))

            if configs:
                logger.info("Loaded %d server config(s) from MCP_SERVERS JSON", len(configs))
                return configs
        else:
            # Try parsing as comma or semicolon separated
            separators = "," if "," in env_value else ";"
            fragments = [frag.strip() for frag in env_value.split(separators) if frag.strip()]
            configs = [_parse_server_mapping_fragment(frag) for frag in fragments]
            if configs:
                logger.info("Loaded %d server config(s) from MCP_SERVERS string", len(configs))
                return configs

            raise ConfigurationError(
                "No valid MCP servers found in MCP_SERVERS environment variable",
                config_key="MCP_SERVERS",
            )

    # Fallback to legacy single server config
    legacy_url = os.getenv("MCP_SERVER_URL")
    if legacy_url:
        name = os.getenv("MCP_SERVER_NAME", DEFAULT_MCP_SERVER_NAME)
        logger.info("Using legacy MCP_SERVER_URL configuration for server '%s'", name)

        try:
            validate_server_config(name, legacy_url)
        except ValidationError as e:
            raise ConfigurationError(
                f"Invalid legacy server configuration: {e.message}",
                config_key="MCP_SERVER_URL",
            ) from e

        return [MCPServerConfig(name=name, url=legacy_url)]

    raise ConfigurationError(
        "No MCP servers configured. Set MCP_SERVERS or MCP_SERVER_URL environment variable.",
        config_key="MCP_SERVERS",
    )


def _convert_tools(
    tools: Sequence[types.Tool],
    *,
    server_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    def _sanitize_schema(schema: dict[str, Any] | list[Any] | Any) -> dict[str, Any] | list[Any] | Any:
        if isinstance(schema, list):
            return [_sanitize_schema(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        assert isinstance(schema, dict) # Mypy assert

        sanitized: dict[str, Any] = {
            key: _sanitize_schema(value)
            for key, value in schema.items()
        }

        schema_type = sanitized.get("type")

        def _type_in(value: Any, expected: str) -> bool:
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
                sanitized_items = _sanitize_schema(items)
                # Gemini requires items to have at least a 'type' field
                if isinstance(sanitized_items, dict) and "type" not in sanitized_items:
                    sanitized_items["type"] = "string"
                sanitized["items"] = sanitized_items
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
                            sanitized_items = _sanitize_schema(items_schema)
                            # Ensure items has a type field for Gemini
                            if isinstance(sanitized_items, dict) and "type" not in sanitized_items:
                                sanitized_items["type"] = "string"
                            sanitized["items"] = sanitized_items
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

    tool_schemas: list[dict[str, Any]] = []
    tool_lookup: dict[str, tuple[str, str]] = {}

    prefix = f"{server_name}__" if server_name else ""

    for tool in tools:
        schema = tool.inputSchema or {"type": "object", "properties": {}}
        schema = _sanitize_schema(schema)
        unique_name = f"{prefix}{tool.name}"
        description = tool.description or ""
        if server_name:
            description = f"[{server_name}] {description}".strip()

        tool_schemas.append(
            {
                "name": unique_name,
                "description": description,
                "parameters": schema,
            }
        )
        tool_lookup[unique_name] = (server_name or "default", tool.name)
        logger.debug("Tool %s schema passed to Vertex: %s", unique_name, json.dumps(schema))

    return tool_schemas, tool_lookup


class MCPChatClient:
    """High level facade that orchestrates MCP tool calls with Vertex AI."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm or LLMFactory.create_llm()
        self._initialized = False
        # Simplified for testing
        self._bound_llm: Runnable | None = None

    async def initialize(self) -> None:
        """
        Initializes the client, connects to servers, and prepares tools.

        Raises:
            InitializationError: If initialization fails
            ConfigurationError: If configuration is invalid
            ServerConnectionError: If connection to any server fails
        """
        if self._initialized:
            logger.debug("Client already initialized, skipping")
            return

        logger.info("Initializing MCP client...")

        try:
            server_configs = _load_server_configs()
            headers = _build_headers()

            self._exit_stack = AsyncExitStack()
            sessions: dict[str, ClientSession] = {}
            tool_schemas: list[dict[str, Any]] = []
            tool_lookup: dict[str, tuple[str, str]] = {}

            for config in server_configs:
                logger.info("Connecting to MCP server '%s' at %s", config.name, config.url)

                try:
                    read, write, get_session_id = await self._exit_stack.enter_async_context(
                        streamablehttp_client(config.url, headers=headers)
                    )
                    session = await self._exit_stack.enter_async_context(ClientSession(read, write))
                    init_result = await session.initialize()

                    logger.info(
                        "Connected to MCP server '%s' (protocol %s)",
                        config.name,
                        init_result.protocolVersion,
                    )

                    session_id = get_session_id()
                    if session_id:
                        logger.debug("Using MCP session id %s for server '%s'", session_id, config.name)

                    sessions[config.name] = session

                    # List available tools
                    tools_result = await session.list_tools()
                    logger.info("Server '%s' exposes %d tool(s)", config.name, len(tools_result.tools))

                    schemas, lookup = _convert_tools(tools_result.tools, server_name=config.name)
                    tool_schemas.extend(schemas)
                    tool_lookup.update(lookup)

                except Exception as e:
                    logger.error("Failed to connect to server '%s': %s", config.name, e)
                    raise ServerConnectionError(
                        server_name=config.name,
                        url=config.url,
                        reason=str(e)
                    ) from e

            self._sessions = sessions
            self._tool_schemas = tool_schemas
            self._tool_lookup = tool_lookup

            logger.info("Total of %d tool(s) available across all servers", len(tool_schemas))

            # Bind tools to LLM
            llm = self._llm
            if self._tool_schemas:
                logger.info("Binding %d tools to LLM for function calling", len(self._tool_schemas))
                self._bound_llm = llm.bind_tools(self._tool_schemas)
            else:
                logger.warning("No tools exposed by any MCP server; continuing without tool binding")
                self._bound_llm = llm

            self._initialized = True
            logger.info("MCP client initialized successfully")

        except (ConfigurationError, ServerConnectionError):
            # Re-raise these as-is
            raise
        except Exception as e:
            logger.error("Initialization failed: %s", e)
            raise InitializationError(
                f"Failed to initialize MCP client: {str(e)}",
                details={"error": str(e)}
            ) from e

    async def close(self) -> None:
        """Closes all connections and cleans up resources."""
        if self._exit_stack:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._sessions = {}
        self._tool_lookup = {}
        self._tool_schemas = []
        self._bound_llm = None
        self._initialized = False

    async def __aenter__(self) -> MCPChatClient:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._initialized

    @property
    def available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self._tool_lookup.keys())

    @property
    def bound_llm(self) -> Runnable | None:
        """Get the bound LLM instance."""
        return self._bound_llm

    def _log_prompt_tokens(self, messages: list[HumanMessage | BaseMessage | ToolMessage], *, context: str) -> None:
        """Log token usage for a prompt without blocking execution on failures."""
        try:
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
            token_count = get_token_count(messages, encoding)
            logger.info("%s: %d token(s) across %d message(s)", context, token_count, len(messages))
        except Exception as e:
            logger.debug("Could not log token usage for %s: %s", context, e)

    def _truncate_messages(
        self, messages: list[HumanMessage | BaseMessage | ToolMessage]
    ) -> list[HumanMessage | BaseMessage | ToolMessage]:
        """
        Truncates messages based on token count to fit within the prompt limit.

        Args:
            messages: List of messages to truncate

        Returns:
            Truncated list of messages

        Raises:
            TokenLimitError: If even a single message exceeds the token limit
        """
        try:
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
            token_count = get_token_count(messages, encoding)

            logger.debug(
                "Prompt tokens before truncation: %d token(s) across %d message(s)",
                token_count,
                len(messages),
            )


            if token_count > MAX_PROMPT_TOKENS:
                logger.warning(
                    "Token count %d exceeds limit %d. Truncating message history...",
                    token_count,
                    MAX_PROMPT_TOKENS
                )

                while get_token_count(messages, encoding) > MAX_PROMPT_TOKENS:
                    if len(messages) <= 1:
                        # Cannot truncate further
                        raise TokenLimitError(
                            current_tokens=get_token_count(messages, encoding),
                            max_tokens=MAX_PROMPT_TOKENS,
                        )
                    # Remove the oldest message after the initial user query
                    removed = messages.pop(1)
                    logger.debug("Removed message from history to reduce tokens")

                logger.info(
                    "Truncated to %d messages (%d tokens)",
                    len(messages),
                    get_token_count(messages, encoding)
                )

        except TokenLimitError:
            raise
        except Exception as e:
            logger.error("Could not use tiktoken for history truncation: %s. Using fallback.", e)
            # Fallback to simple message count-based truncation
            if len(messages) > FALLBACK_MAX_MESSAGES:
                logger.warning("Falling back to message count truncation: keeping last %d messages", FALLBACK_MAX_MESSAGES)
                messages = [messages[0]] + messages[-(FALLBACK_MAX_MESSAGES - 1):]

        return messages

    async def _compress_tool_payload(
        self,
        content: str,
        *,
        max_tokens: int | None = None,
        chunk_tokens: int = 800,
    ) -> str:
        """
        Summarize large tool payloads so a single tool result cannot exceed the prompt budget.
        """
        if not content or not self._llm:
            return content

        # Use env var or default if max_tokens not provided
        if max_tokens is None:
            max_tokens = int(os.getenv("MCP_TOOL_RESPONSE_MAX_TOKENS", "2000"))

        try:
            encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
        except Exception as e:
            logger.error("Could not create encoding for compression: %s", e)
            return shorten(content, width=4000, placeholder=" ...") if len(content) > 4000 else content

        try:
            if len(encoding.encode(content)) <= max_tokens:
                return content

            logger.info("Compressing tool output of length %d (exceeds %d tokens)", len(content), max_tokens)

            def _split_by_tokens(text: str) -> list[str]:
                tokens = encoding.encode(text)
                return [encoding.decode(tokens[i:i + chunk_tokens]) for i in range(0, len(tokens), chunk_tokens)]

            summaries: list[str] = []
            chunks = _split_by_tokens(content)
            
            # Limit number of chunks to avoid excessive LLM calls for massive files
            MAX_CHUNKS = 5
            if len(chunks) > MAX_CHUNKS:
                logger.warning("Tool output too large (%d chunks), summarizing first %d only", len(chunks), MAX_CHUNKS)
                chunks = chunks[:MAX_CHUNKS]
                chunks.append("... (remaining content truncated) ...")

            for idx, chunk in enumerate(chunks):
                if chunk == "... (remaining content truncated) ...":
                    summaries.append(chunk)
                    continue

                prompt = (
                    f"Compress this tool output chunk ({idx + 1}/{len(chunks)}). "
                    "EXTREMELY CONCISE. Preserve IDs, metrics, errors, and key data. "
                    "Remove formatting, whitespace, and verbose descriptions. "
                    f"Max {chunk_tokens // 2} tokens."
                )
                try:
                    summary_msg = await self._llm.ainvoke([HumanMessage(content=f"{prompt}\n\n{chunk}")])
                    summaries.append(ai_content_to_str(summary_msg))
                except Exception as e:
                    logger.warning("Failed to summarize chunk %d: %s", idx, e)
                    summaries.append(shorten(chunk, width=500, placeholder="..."))

            merged = "\n".join(summaries)
            if len(encoding.encode(merged)) > max_tokens:
                logger.warning("Compressed content still exceeds limit, truncating...")
                merged = shorten(merged, width=max_tokens * 4, placeholder=" ...") # Approx chars
            
            return merged

        except Exception as e:
            logger.error("Failed to compress tool payload: %s", e)
            return shorten(content, width=4000, placeholder=" ...") if len(content) > 4000 else content

    async def ainvoke_llm(self, messages: list[HumanMessage | BaseMessage | ToolMessage]) -> BaseMessage:
        """
        Invokes the LLM with a list of messages without any tool loop logic.

        Args:
            messages: List of messages to send to the LLM

        Returns:
            The LLM's response message

        Raises:
            InitializationError: If client is not initialized
            LLMError: If LLM invocation fails
            TokenLimitError: If messages exceed token limit
        """
        if not self._bound_llm:
            raise InitializationError("Client is not initialized or has no tools bound")

        messages = self._truncate_messages(messages)
        self._log_prompt_tokens(messages, context="LLM invoke prompt")
        logger.debug("Invoking LLM with %d messages", len(messages))

        try:
            return await self._bound_llm.ainvoke(messages)
        except Exception as e:
            logger.error("LLM invocation failed: %s", e)
            raise LLMError(f"LLM invocation failed: {str(e)}", reason=str(e)) from e

    async def astream_llm(self, messages: list[HumanMessage | BaseMessage | ToolMessage]) -> AsyncIterator[BaseMessageChunk]:
        """
        Streams the LLM response for a list of messages without any tool loop logic.

        Args:
            messages: List of messages to send to the LLM

        Yields:
            Message chunks from the LLM

        Raises:
            InitializationError: If client is not initialized
            LLMError: If LLM streaming fails
            TokenLimitError: If messages exceed token limit
        """
        if not self._bound_llm:
            raise InitializationError("Client is not initialized or has no tools bound")

        messages = self._truncate_messages(messages)
        self._log_prompt_tokens(messages, context="LLM streaming prompt")
        logger.debug("Streaming LLM with %d messages", len(messages))

        try:
            async for chunk in self._bound_llm.astream(messages):
                yield chunk
        except Exception as e:
            logger.error("LLM streaming failed: %s", e)
            raise LLMError(f"LLM streaming failed: {str(e)}", reason=str(e)) from e

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> ToolMessage:
        """
        Executes a tool by its simple name (e.g., 'elastic_search').
        Finds the correct server and executes the tool.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            ToolMessage containing the tool's response

        Raises:
            ToolNotFoundError: If the tool is not found
            ValidationError: If arguments are invalid
            ToolExecutionError: If tool execution fails
        """
        # Validate arguments
        validate_tool_arguments(tool_name, args)

        # Find the fully qualified tool name and server
        full_tool_name: str | None = None
        for lookup_name, (_, original_tool_name) in self._tool_lookup.items():
            if original_tool_name.startswith(tool_name):
                full_tool_name = lookup_name
                break

        if not full_tool_name:
            available_tools = list(self._tool_lookup.keys())
            raise ToolNotFoundError(tool_name, available_tools)

        server_name, original_tool_name = self._tool_lookup[full_tool_name]

        logger.info(
            "Executing tool '%s' on server '%s' with %d argument(s)",
            original_tool_name,
            server_name,
            len(args)
        )
        logger.debug("Tool arguments: %s", args)

        try:
            session = self._sessions[server_name]
            result = await session.call_tool(original_tool_name, arguments=args)
            payload = _render_content_blocks(result.content)

            logger.debug("Tool '%s' returned %d characters", original_tool_name, len(payload))

            # Apply custom parser if available
            parser = get_parser(original_tool_name)
            if parser:
                logger.debug("Using custom parser for tool '%s'", original_tool_name)
                try:
                    final_payload = parser(payload)
                except Exception as e:
                    logger.error("Parser failed for tool '%s': %s", original_tool_name, e)
                    raise ParsingError(
                        f"Failed to parse output from tool '{original_tool_name}': {str(e)}",
                        data_sample=payload[:200]
                    ) from e
            else:
                final_payload = payload

            if final_payload:
                final_payload = await self._compress_tool_payload(final_payload, max_tokens=3000)

            return ToolMessage(
                content=final_payload or "(no content returned)",
                tool_call_id=f"agent_call_{tool_name}",
                name=full_tool_name,
            )

        except (ToolNotFoundError, ValidationError, ParsingError):
            # Re-raise these as-is
            raise
        except Exception as e:
            logger.error("Tool execution failed for '%s': %s", original_tool_name, e)
            raise ToolExecutionError(
                tool_name=original_tool_name,
                reason=str(e),
                arguments=args
            ) from e


    async def run(self, query: str) -> AsyncIterator[str]:
        if not self._initialized or not self._bound_llm:
            logger.info("Client not initialized. Initializing now.")
            await self.initialize()

        if not self._bound_llm:
            raise RuntimeError("Client initialization failed.")

        llm = self._bound_llm
        messages: list[HumanMessage | BaseMessage | ToolMessage] = [HumanMessage(content=query)] # Changed type hint

        while True:
            self._log_prompt_tokens(messages, context="LLM tool-loop prompt")
            stream = llm.astream(messages)

            collected_chunks = []
            full_response: BaseMessage | BaseMessageChunk | None = None # Changed type hint

            async for chunk in stream:
                collected_chunks.append(chunk)
                if full_response is None:
                    full_response = chunk
                else:
                    full_response += chunk

            if not full_response:
                return

            messages.append(full_response)

            if not full_response.tool_calls:
                # Final answer. Stream it out.
                for chunk in collected_chunks:
                    content = ai_content_to_str(chunk)
                    if content:
                        yield content
                return

            # Tool calls present.
            tool_messages = []
            for tool_call in full_response.tool_calls:
                tool_call_name = tool_call.get("name")
                if not tool_call_name:
                    logger.error("Tool call is missing a name: %s", tool_call)
                    raise RuntimeError("Received tool call without a name.")

                server_name, original_tool_name = self._tool_lookup.get(tool_call_name, (None, None))
                if not server_name or server_name not in self._sessions or not original_tool_name:
                    raise RuntimeError(f"Unknown tool requested by LLM: {tool_call_name}")

                call_id = tool_call.get("id") or tool_call_name
                arguments = tool_call.get("args") or {}

                logger.debug(
                    "Executing tool %s on server %s with args %s",
                    original_tool_name,
                    server_name,
                    arguments,
                )
                session = self._sessions[server_name]
                result = await session.call_tool(original_tool_name, arguments=arguments)
                payload = _render_content_blocks(result.content)

                parser = get_parser(original_tool_name)
                if parser:
                    logger.debug("Using custom parser for tool '%s'", original_tool_name)
                    final_payload = parser(payload)
                else:
                    final_payload = payload

                if final_payload:
                    final_payload = await self._compress_tool_payload(final_payload, max_tokens=3000)

                tool_messages.append(
                    ToolMessage(
                        content=final_payload or "(no content returned)",
                        tool_call_id=call_id,
                        name=tool_call_name,
                    )
                )
            messages.extend(tool_messages)

            messages = self._truncate_messages(messages)



_client: MCPChatClient | None = None


async def get_client() -> MCPChatClient:
    global _client
    if _client is None:
        _client = MCPChatClient()
    if not _client._initialized:
        await _client.initialize()
    return _client


async def run(query: str) -> AsyncIterator[str]:
    """Run a query using the log analysis agent."""
    from agents import LogAnalysisAgent  # Local import to avoid circular dependency

    client = await get_client()
    agent = LogAnalysisAgent(client)
    async for chunk in agent.run(query):
        yield chunk


__all__ = ["MCPChatClient", "run", "get_client"]









