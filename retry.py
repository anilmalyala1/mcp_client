"""
Retry logic with exponential backoff for resilience.
"""

from __future__ import annotations

import asyncio
import logging
import random
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

        if isinstance(exception, retryable_exceptions):
            return True

        # Check for 429 Rate Limit or Resource Exhausted
        ex_str = str(exception).lower()
        if "429" in ex_str or "rate limit" in ex_str or "resource exhausted" in ex_str:
            logger.warning("Detected rate limit/resource exhaustion: %s", exception)
            return True

        # Check for Google ResourceExhausted if available
        try:
            from google.api_core.exceptions import ResourceExhausted
            if isinstance(exception, ResourceExhausted):
                return True
        except ImportError:
            pass

        return False

    def _calculate_wait_time(self, attempt: int) -> float:
        """
        Calculate wait time using exponential backoff with optional jitter.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Wait time in seconds
        """
        wait = min(
            self.config.min_wait * (self.config.exponential_base ** attempt),
            self.config.max_wait,
        )
        
        if self.config.jitter:
            # Add random jitter between 0 and 1 second (or smaller fraction of wait)
            # Standard "Full Jitter" approach: random_between(0, wait)
            # Or simple additive jitter. Here we use additive for simplicity but kept small.
            wait += random.uniform(0, 1)
            
        return wait

    def _get_retry_after(self, exception: Exception) -> float | None:
        """Extract retry-after value from exception if available."""
        # Check for 'retry_after' attribute (common in some libs)
        retry_after = getattr(exception, "retry_after", None)
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
        
        # Check for headers (e.g. OpenAI, HTTP exceptions)
        headers = getattr(exception, "headers", None)
        if headers and isinstance(headers, dict):
            val = headers.get("Retry-After") or headers.get("retry-after")
            if val:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

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
                
                # Check for explicit Retry-After header
                retry_after = self._get_retry_after(e)
                if retry_after:
                    logger.warning("Respecting Retry-After header: %.2fs", retry_after)
                    wait_time = max(wait_time, retry_after)

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
