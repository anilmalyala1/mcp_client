"""
Shared utility functions for the MCP client.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, BaseMessageChunk
from tiktoken import Encoding


def ai_content_to_str(message: BaseMessage | BaseMessageChunk) -> str:
    """
    Convert AI message content to a string.

    Handles both string content and structured content with multiple parts.

    Args:
        message: The message to convert

    Returns:
        String representation of the message content
    """
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


def get_token_count(messages: list[Any], encoding: Encoding) -> int:
    """
    Get the approximate token count for a list of messages.

    Args:
        messages: List of messages to count tokens for
        encoding: The tiktoken encoding to use

    Returns:
        Approximate number of tokens
    """
    num_tokens = 0
    for message in messages:
        # Simple approximation of token count
        num_tokens += len(encoding.encode(str(message.content)))
        if hasattr(message, "name") and message.name:
            num_tokens += len(encoding.encode(message.name))
    return num_tokens
