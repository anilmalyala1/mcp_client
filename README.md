# MCP Client Examples

This repository contains a collection of Python examples demonstrating how to build clients for the MCP (Minecraft Control Protocol) framework. These examples showcase various approaches to interacting with MCP servers, from simple direct tool calls to more advanced orchestration using Large Language Models (LLMs) like OpenAI's GPT and Google's Vertex AI.

## Features

- **Multiple Client Implementations:** See different ways to build MCP clients.
- **LLM Orchestration:** Examples of using LangChain with OpenAI and Vertex AI to control MCP servers with natural language.
- **Dynamic Tool Discovery:** Clients that can dynamically fetch available tools from an MCP server.
- **Stdio and HTTP Transports:** Examples of connecting to MCP servers using both stdio and HTTP.
- **Simple MCP Server:** A basic `weather.py` server to demonstrate tool exposure.

## Getting Started

### Prerequisites

- Python 3.14.0 or higher
- `uv` package manager (or `pip`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd mcp_client
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt`, you can install from `pyproject.toml` using a modern package manager like `uv` or `pip` with `.[all]`)*


## Usage

This project includes several standalone client examples.

### `weather.py`

A simple MCP server that exposes a `get_weather` tool. You can run this to have a server to test clients against.

```bash
uv run weather.py
```

### `client_simple.py`

A basic client that connects to a stdio-based MCP server (like the Airbnb example server) and calls a tool directly.

```bash
uv run client_simple.py
```

### `client_simple_local.py`

A client that connects to a local Elasticsearch MCP server and executes a few predefined tool calls.

```bash
uv run client_simple_local.py
```

### `client_orchestrator.py`

A FastAPI application that acts as a decoupled orchestrator. It fetches tools from a running MCP server and uses an LLM to fulfill user prompts.

**First, ensure another MCP server (e.g., `weather.py`) is running.**

Then, run the orchestrator:
```bash
uv run client_orchestrator.py
```
You can then send requests to `http://127.0.0.1:8002/chat`.

### `client_simple_input.py`

An interactive command-line client that uses an LLM to answer questions by calling tools from one of two MCP servers.

```bash
uv run client_simple_input.py
```

### `client_simple_local_ai.py`

A command-line client that takes a query and uses an LLM to execute tools on a local Elasticsearch MCP server.

```bash
uv run client_simple_local_ai.py
```

### `client_streamable_vertex_ai.py`

A more advanced, interactive client that uses a streamable HTTP transport to connect to an MCP server and Vertex AI for tool orchestration.

```bash
# Make sure to set the required environment variables first
uv run client_streamable_vertex_ai.py
```

## Configuration

Several of the clients require environment variables to be set. You can create a `.env` file in the project root to manage these.

-   **`OPENAI_API_KEY`**: Your API key for OpenAI, used by clients that leverage GPT models.
-   **`MCP_SERVER_BASE_URL`**: The base URL for an HTTP-based MCP server (e.g., `http://127.0.0.1:8000`), used by `client_orchestrator.py`.
-   **`VERTEXAI_PROJECT`**: Your Google Cloud project ID for Vertex AI.
-   **`VERTEXAI_LOCATION`**: The Google Cloud location for your Vertex AI project (e.g., `us-central1`).
-   **`GOOGLE_API_KEY`**: Your Google API key for authentication with Vertex AI.
-   **`MCP_API_KEY`**: An API key for the MCP server, if it requires authentication.
-   **`MCP_SERVER_URL`**: The URL for the streamable HTTP MCP server, used by `client_streamable_vertex_ai.py`.
-   **`MCP_SAMPLE_QUERY`**: A default query to use in the interactive clients.

### LLM Configuration (New)

You can switch between LLM providers using `LLM_PROVIDER`.

-   **`LLM_PROVIDER`**: One of `openai`, `vertex`, `ollama` (default: `openai`).
-   **`OPENAI_MODEL`**: Model name for OpenAI (default: `gpt-4-turbo`).
-   **`VERTEXAI_MODEL`**: Model name for Vertex AI (default: `gemini-2.5-pro`).
-   **`OLLAMA_MODEL`**: Model name for Ollama (default: `qwen3:4b`).
-   **`OLLAMA_BASE_URL`**: Base URL for Ollama (default: `http://localhost:11434`).

### Rate Limiting & Resilience

The client includes built-in handling for rate limits (429 errors) with exponential backoff and jitter.

-   **`MCP_MAX_RETRIES`**: Maximum number of retries (default: 3).
-   **`MCP_RETRY_MIN_WAIT`**: Minimum wait time in seconds (default: 1.0).
-   **`MCP_RETRY_MAX_WAIT`**: Maximum wait time in seconds (default: 10.0).

### Context Compression

Large tool outputs are automatically compressed to fit within the context window.

-   **`MCP_TOOL_RESPONSE_MAX_TOKENS`**: Maximum tokens for a single tool response before compression kicks in (default: 2000).
