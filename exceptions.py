"""
Custom exceptions for the MCP client.
"""

from __future__ import annotations


class MCPError(Exception):
    """Base exception for all MCP client errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class InitializationError(MCPError):
    """Raised when client initialization fails."""

    def __init__(self, message: str = "Failed to initialize MCP client", details: dict | None = None):
        super().__init__(message, details)


class ServerConnectionError(MCPError):
    """Raised when connection to an MCP server fails."""

    def __init__(self, server_name: str, url: str, reason: str | None = None):
        message = f"Failed to connect to MCP server '{server_name}' at {url}"
        if reason:
            message += f": {reason}"
        details = {"server_name": server_name, "url": url}
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class ToolNotFoundError(MCPError):
    """Raised when a requested tool is not found."""

    def __init__(self, tool_name: str, available_tools: list[str] | None = None):
        message = f"Tool '{tool_name}' not found on any connected server"
        details = {"requested_tool": tool_name}
        if available_tools:
            message += f". Available tools: {', '.join(available_tools[:5])}"
            if len(available_tools) > 5:
                message += f" and {len(available_tools) - 5} more"
            details["available_tools"] = available_tools
        super().__init__(message, details)


class ToolExecutionError(MCPError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, reason: str, arguments: dict | None = None):
        message = f"Failed to execute tool '{tool_name}': {reason}"
        details = {"tool_name": tool_name, "reason": reason}
        if arguments:
            details["arguments"] = arguments
        super().__init__(message, details)


class ConfigurationError(MCPError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str | None = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details)


class ValidationError(MCPError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None, value: str | None = None):
        details = {}
        if field:
            details["field"] = field
        if value:
            details["value"] = value
        super().__init__(message, details)


class LLMError(MCPError):
    """Raised when LLM invocation fails."""

    def __init__(self, message: str, model: str | None = None, reason: str | None = None):
        details = {}
        if model:
            details["model"] = model
        if reason:
            details["reason"] = reason
        super().__init__(message, details)


class TokenLimitError(MCPError):
    """Raised when token limit is exceeded."""

    def __init__(self, current_tokens: int, max_tokens: int):
        message = f"Token limit exceeded: {current_tokens} tokens (max: {max_tokens})"
        details = {"current_tokens": current_tokens, "max_tokens": max_tokens}
        super().__init__(message, details)


class ParsingError(MCPError):
    """Raised when parsing data fails."""

    def __init__(self, message: str, data_sample: str | None = None):
        details = {}
        if data_sample:
            # Truncate sample to avoid huge error messages
            details["data_sample"] = data_sample[:200] + "..." if len(data_sample) > 200 else data_sample
        super().__init__(message, details)


class AgentError(MCPError):
    """Raised when agent execution fails."""

    def __init__(self, agent_name: str, step: str | None = None, reason: str | None = None):
        message = f"Agent '{agent_name}' failed"
        if step:
            message += f" at step '{step}'"
        if reason:
            message += f": {reason}"
        details = {"agent_name": agent_name}
        if step:
            details["step"] = step
        if reason:
            details["reason"] = reason
        super().__init__(message, details)
