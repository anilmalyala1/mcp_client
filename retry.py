"""
Retry logic with exponential backoff for resilience.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, TypeVar, cast

from config import RetryConfig
from exceptions import LLMError, ServerConnectionError, ToolExecutionError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryHandler:
    """Handles retry logic with exponential backoff."""

    def __init__(self, config: RetryConfig):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self.config = config

    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.config.max_retries:
            return False

        # Retry on transient errors
        retryable_exceptions = (
            ServerConnectionError,
            LLMError,
            ToolExecutionError,
            ConnectionError,
            TimeoutError,
        )

        return isinstance(exception, retryable_exceptions)

    def _calculate_wait_time(self, attempt: int) -> float:
        """
        Calculate wait time using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Wait time in seconds
        """
        wait = min(
            self.config.min_wait * (self.config.exponential_base ** attempt),
            self.config.max_wait,
        )
        return wait

    async def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Exception: The last exception if all retries failed
        """
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if attempt > 0:
                    logger.info("Operation succeeded after %d retry attempt(s)", attempt)

                return cast(T, result)

            except Exception as e:
                last_exception = e

                if not self._should_retry(e, attempt):
                    logger.error(
                        "Operation failed and will not retry: %s (attempt %d/%d)",
                        e,
                        attempt + 1,
                        self.config.max_retries + 1,
                    )
                    raise

                wait_time = self._calculate_wait_time(attempt)
                logger.warning(
                    "Operation failed: %s. Retrying in %.2fs (attempt %d/%d)",
                    e,
                    wait_time,
                    attempt + 1,
                    self.config.max_retries + 1,
                )
                await asyncio.sleep(wait_time)

        # This should never be reached due to the raise in the loop
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic error: no exception to raise")


# Convenience function for simple retry
async def retry_async(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    **kwargs: Any,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retries
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Exception: The last exception if all retries failed

    Example:
        result = await retry_async(
            client.execute_tool,
            "elastic_search",
            {"query": "errors"},
            max_retries=3
        )
    """
    config = RetryConfig(
        max_retries=max_retries,
        min_wait=min_wait,
        max_wait=max_wait,
    )
    handler = RetryHandler(config)
    return await handler.execute_with_retry(func, *args, **kwargs)
