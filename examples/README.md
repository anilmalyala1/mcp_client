# MCP Client Examples

This directory contains example scripts demonstrating various features of the MCP client.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the fundamental usage patterns:
- Setting up logging
- Initializing the client
- Executing queries
- Using the context manager pattern

**Run:**
```bash
python examples/basic_usage.py
```

### 2. Configuration Management (`with_config.py`)

Shows how to use the configuration system:
- Loading configuration from environment
- Using different LLM providers (Vertex AI, Ollama)
- Configuring retry behavior
- Custom configuration objects

**Run:**
```bash
python examples/with_config.py
```

### 3. Error Handling (`error_handling.py`)

Demonstrates robust error handling:
- Catching specific exception types
- Extracting error details
- Using retry logic for resilience
- Implementing graceful degradation
- Handling initialization errors

**Run:**
```bash
python examples/error_handling.py
```

## Prerequisites

Make sure you have:
1. Configured environment variables (see `.env.example`)
2. MCP server running at the configured URL
3. Valid API keys for your chosen LLM provider

## Environment Variables

```bash
# LLM Configuration
VERTEXAI_MODEL=gemini-2.5-pro
VERTEXAI_TEMPERATURE=0.1
GOOGLE_API_KEY=your_api_key_here

# Or for Ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_TEMPERATURE=0.1
OLLAMA_BASE_URL=http://localhost:11434

# MCP Server Configuration
MCP_SERVERS='[{"name": "elastic-server", "url": "http://localhost:8000/mcp"}]'

# Logging Configuration
MCP_LOG_LEVEL=INFO

# Retry Configuration
MCP_MAX_RETRIES=3
MCP_RETRY_MIN_WAIT=1.0
MCP_RETRY_MAX_WAIT=10.0
```

## Common Patterns

### Context Manager (Recommended)

```python
async with MCPChatClient() as client:
    result = await client.run("Your query here")
    # Client automatically cleaned up
```

### Error Handling

```python
try:
    result = await client.execute_tool("tool_name", args)
except ToolNotFoundError as e:
    print(f"Available tools: {e.details['available_tools']}")
except ToolExecutionError as e:
    print(f"Execution failed: {e.details['reason']}")
```

### Retry Logic

```python
from retry import retry_async

result = await retry_async(
    client.execute_tool,
    "tool_name",
    {"arg": "value"},
    max_retries=3
)
```

## Tips

1. **Always use the context manager** for automatic resource cleanup
2. **Catch specific exceptions** instead of generic Exception
3. **Use retry logic** for operations that might fail transiently
4. **Configure logging** appropriately for your use case
5. **Check error details** for debugging information

## Troubleshooting

### "GOOGLE_API_KEY environment variable is required"
Set your Google API key: `export GOOGLE_API_KEY=your_key_here`

### "No MCP servers configured"
Set MCP_SERVERS: `export MCP_SERVERS='[{"name": "server", "url": "http://localhost:8000/mcp"}]'`

### "Tool 'X' not found"
Check available tools: `client.available_tools`

### Connection errors
1. Ensure MCP server is running
2. Check server URL is correct
3. Verify network connectivity
4. Review server logs for errors
