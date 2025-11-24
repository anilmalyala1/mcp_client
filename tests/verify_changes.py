
import asyncio
import os
import logging
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock environment variables
os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
os.environ["GOOGLE_API_KEY"] = "dummy-key"
os.environ["MCP_TOOL_RESPONSE_MAX_TOKENS"] = "100"

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_factory import LLMFactory
from retry import RetryHandler, RetryConfig
from client_streamable_vertex_ai import MCPChatClient
from langchain_core.messages import HumanMessage

async def test_retry_jitter():
    logger.info("Testing RetryHandler Jitter...")
    
    config = RetryConfig(max_retries=3, min_wait=0.1, max_wait=0.2, jitter=True)
    handler = RetryHandler(config)
    
    # Test wait time calculation directly
    wait1 = handler._calculate_wait_time(0)
    wait2 = handler._calculate_wait_time(0)
    
    logger.info("Wait times with jitter: %.4f, %.4f", wait1, wait2)
    
    # It's statistically unlikely they are exactly equal with random jitter
    if wait1 != wait2:
        logger.info("✅ Jitter produced different wait times")
    else:
        logger.warning("⚠️ Jitter produced same wait times (unlikely but possible)")

async def test_retry_after_header():
    logger.info("Testing Retry-After Header...")
    
    config = RetryConfig(max_retries=1, min_wait=0.1, max_wait=0.2)
    handler = RetryHandler(config)
    
    mock_func = AsyncMock()
    
    # Create an exception with headers
    error_with_header = Exception("Rate Limit")
    error_with_header.headers = {"Retry-After": "0.5"}
    
    mock_func.side_effect = [error_with_header, "success"]
    
    start_time = asyncio.get_running_loop().time()
    await handler.execute_with_retry(mock_func)
    end_time = asyncio.get_running_loop().time()
    
    duration = end_time - start_time
    logger.info("Retry took %.4f seconds (expected > 0.5s)", duration)
    
    if duration >= 0.5:
        logger.info("✅ Respected Retry-After header")
    else:
        logger.error("❌ Did not respect Retry-After header")

async def main():
    await test_retry_jitter()
    await test_retry_after_header()

if __name__ == "__main__":
    asyncio.run(main())
