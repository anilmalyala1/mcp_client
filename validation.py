"""
Input validation functions for the MCP client.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from exceptions import ValidationError, ConfigurationError


def validate_url(url: str, field_name: str = "url") -> None:
    """
    Validate that a URL is well-formed and uses http/https.

    Args:
        url: The URL to validate
        field_name: Name of the field for error messages

    Raises:
        ValidationError: If the URL is invalid
    """
    if not url:
        raise ValidationError(f"{field_name} cannot be empty", field=field_name)

    if not url.startswith(("http://", "https://")):
        raise ValidationError(
            f"{field_name} must start with http:// or https://",
            field=field_name,
            value=url[:50],
        )

    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValidationError(
                f"{field_name} is malformed",
                field=field_name,
                value=url[:50],
            )
    except Exception as e:
        raise ValidationError(
            f"Invalid {field_name}: {str(e)}",
            field=field_name,
            value=url[:50],
        ) from e


def validate_non_empty_string(value: str, field_name: str) -> None:
    """
    Validate that a string is not empty.

    Args:
        value: The string to validate
        field_name: Name of the field for error messages

    Raises:
        ValidationError: If the string is empty
    """
    if not value or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty", field=field_name)


def validate_server_config(name: str, url: str) -> None:
    """
    Validate server configuration.

    Args:
        name: Server name
        url: Server URL

    Raises:
        ValidationError: If configuration is invalid
    """
    validate_non_empty_string(name, "server name")
    validate_url(url, "server URL")


def validate_json_string(json_str: str, field_name: str = "JSON") -> dict[str, Any]:
    """
    Validate and parse a JSON string.

    Args:
        json_str: The JSON string to validate
        field_name: Name of the field for error messages

    Returns:
        Parsed JSON as a dictionary

    Raises:
        ValidationError: If JSON is invalid
    """
    if not json_str:
        raise ValidationError(f"{field_name} cannot be empty", field=field_name)

    try:
        result = json.loads(json_str)
        if not isinstance(result, (dict, list)):
            raise ValidationError(
                f"{field_name} must be a JSON object or array",
                field=field_name,
                value=json_str[:100],
            )
        return result
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid {field_name}: {str(e)}",
            field=field_name,
            value=json_str[:100],
        ) from e


def validate_temperature(temperature: float) -> None:
    """
    Validate temperature parameter for LLM.

    Args:
        temperature: Temperature value to validate

    Raises:
        ValidationError: If temperature is out of valid range
    """
    if not 0.0 <= temperature <= 2.0:
        raise ValidationError(
            f"Temperature must be between 0.0 and 2.0, got {temperature}",
            field="temperature",
            value=str(temperature),
        )


def validate_token_limit(token_limit: int) -> None:
    """
    Validate token limit parameter.

    Args:
        token_limit: Token limit to validate

    Raises:
        ValidationError: If token limit is invalid
    """
    if token_limit <= 0:
        raise ValidationError(
            f"Token limit must be positive, got {token_limit}",
            field="token_limit",
            value=str(token_limit),
        )

    if token_limit > 1_000_000:
        raise ValidationError(
            f"Token limit seems unreasonably high: {token_limit}",
            field="token_limit",
            value=str(token_limit),
        )


def validate_environment_variable(var_name: str, required: bool = True) -> str | None:
    """
    Validate that an environment variable is set.

    Args:
        var_name: Name of the environment variable
        required: Whether the variable is required

    Returns:
        The value of the environment variable, or None if not required and not set

    Raises:
        ConfigurationError: If required variable is not set
    """
    import os

    value = os.getenv(var_name)
    if required and not value:
        raise ConfigurationError(
            f"Required environment variable '{var_name}' is not set",
            config_key=var_name,
        )
    return value


def validate_tool_arguments(tool_name: str, args: dict[str, Any]) -> None:
    """
    Validate tool arguments.

    Args:
        tool_name: Name of the tool
        args: Arguments to validate

    Raises:
        ValidationError: If arguments are invalid
    """
    if not isinstance(args, dict):
        raise ValidationError(
            f"Tool '{tool_name}' arguments must be a dictionary",
            field="arguments",
            value=str(type(args)),
        )
