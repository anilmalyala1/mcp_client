"""
FastAPI application for the MCP client.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from client_streamable_vertex_ai import get_client
from exceptions import MCPError
from logging_config import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting up FastAPI application")
    try:
        client = await get_client()
        logger.info("MCP client initialized successfully")
        logger.info("Available tools: %s", client.available_tools)
    except Exception as e:
        logger.error("Failed to initialize MCP client: %s", e)
        raise

    yield

    # Shutdown
    logger.info("Shutting down FastAPI application")
    # The global client will be cleaned up automatically


app = FastAPI(title="MCP Chat API", version="1.0.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    metadata: dict[str, Any] | None = None


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request.

    Args:
        request: ChatRequest with query

    Returns:
        ChatResponse with the answer

    Raises:
        HTTPException: If processing fails
    """
    logger.info("Received chat request: %s", request.query[:100])

    try:
        # Get the initialized client (now with await!)
        client = await get_client()

        # Collect the streamed response
        response_parts = []
        async for chunk in client.run(request.query):
            response_parts.append(chunk)

        full_response = "".join(response_parts)

        logger.info("Chat request completed successfully, response length: %d", len(full_response))

        return ChatResponse(
            response=full_response,
            metadata={
                "query_length": len(request.query),
                "response_length": len(full_response),
            },
        )

    except MCPError as exc:
        logger.error("MCP error processing request: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"MCP error: {exc.message}",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error processing chat request")
        raise HTTPException(
            status_code=500,
            detail="Failed to process chat request",
        ) from exc


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Process a chat request with streaming response.

    Args:
        request: ChatRequest with query

    Returns:
        StreamingResponse with chunks

    Raises:
        HTTPException: If processing fails
    """
    logger.info("Received streaming chat request: %s", request.query[:100])

    async def generate():
        """Generate response chunks."""
        try:
            client = await get_client()
            async for chunk in client.run(request.query):
                yield chunk
        except MCPError as exc:
            logger.error("MCP error in streaming: %s", exc)
            yield f"\n\nError: {exc.message}"
        except Exception as exc:
            logger.exception("Unexpected error in streaming")
            yield f"\n\nError: {str(exc)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    try:
        client = await get_client()
        return {
            "status": "healthy",
            "initialized": client.is_initialized,
            "available_tools": len(client.available_tools),
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.get("/tools")
async def list_tools():
    """
    List available tools.

    Returns:
        List of available tools
    """
    try:
        client = await get_client()
        return {
            "tools": client.available_tools,
            "count": len(client.available_tools),
        }
    except Exception as e:
        logger.error("Failed to list tools: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Failed to list tools",
        ) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
