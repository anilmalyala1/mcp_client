"""
Type aliases and custom types for the MCP client.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

# Message type aliases
MessageList = list[HumanMessage | BaseMessage | ToolMessage]
"""Type alias for a list of messages."""

# Parser type aliases
ParserFunc = Callable[[Any], str | dict[str, Any]]
"""Type alias for parser functions that transform tool output."""

# Tool types
ToolName = str
"""Type alias for tool names."""

ToolArguments = dict[str, Any]
"""Type alias for tool arguments."""

ServerName = str
"""Type alias for server names."""

# Configuration types
Headers = dict[str, str]
"""Type alias for HTTP headers."""

ToolSchema = dict[str, Any]
"""Type alias for tool schema definitions."""

ToolLookup = dict[str, tuple[str, str]]
"""Type alias for tool lookup mapping (tool_name -> (server_name, original_name))."""
