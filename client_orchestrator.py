
# mcp_orchestrator.py
"""
A FastAPI module that serves as a DECOUPLED orchestrator client.

This application no longer contains any RCON logic. Instead, it is
designed to connect to an existing MCP HTTP Server that exposes its
own tools.

NEW ARCHITECTURE:
1.  On startup, this app calls `GET /tools` on the MCP_SERVER_BASE_URL
    to dynamically fetch the list of available tool schemas.
2.  When a user calls this app's `/chat` endpoint, this app sends the
    prompt and the *dynamic tool schemas* to the LLM.
3.  If the LLM requests a tool call, this app makes a `POST /execute_tool`
    request to the MCP_SERVER_BASE_URL.
4.  It returns the LLM's final, summarized response.

To run this application:
1.  Ensure you have the required libraries installed:
    pip install fastapi uvicorn pydantic langchain-google-vertexai httpx
2.  Ensure your *other* MCP HTTP server is running and accessible at
    the URL defined in `MCP_SERVER_BASE_URL`.
3.  Ensure your Google Cloud environment is authenticated.
4.  Run this server:
    uvicorn mcp_orchestrator:app --reload
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI as ChatVertexAI
from langchain.schema import HumanMessage, SystemMessage, FunctionMessage
import httpx
import os
from contextlib import asynccontextmanager

# --- 1. Configuration ---

# CRITICAL: This is the URL of your *other* server.
# This server must expose:
# 1. GET /tools: Returns a JSON list of tool schemas.
# 2. POST /execute_tool: Accepts {"name": "...", "args": {...}} and returns {"result": "..."}
MCP_SERVER_BASE_URL = "http://127.0.0.1:8000" 

# LLM System Instructions
SYSTEM_INSTRUCTIONS = (
    "You are an expert Minecraft Server Operator AI. Your goal is to fulfill user requests "
    "by calling the provided tools. You will be given a list of tools; review them "
    "and call them as needed. Your final response MUST be a clear, conversational summary."
)

# --- 2. FastAPI Lifespan (Startup/Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan.
    On startup, it creates an HTTP client and dynamically fetches tools
    from the MCP server. On shutdown, it closes the client.
    """
    print("--- Orchestrator App Starting Up ---")
    
    # 1. Initialize a reusable HTTP client
    app.state.http_client = httpx.AsyncClient(base_url=MCP_SERVER_BASE_URL, timeout=10.0)
    
    # 2. Dynamically fetch tools from the MCP server
    tool_url = "/mcp"
    try:
        print(f"Fetching tools from {MCP_SERVER_BASE_URL}{tool_url}...")
        app.state.http_client.headers.update({"Content-Type": "application/json"})
        app.state.http_client.headers.update({"Accept": "application/json"})
        app.state.http_client.headers.update({"Accept": "text/event-stream"})
        

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
       
        response = await  app.state.http_client.post(tool_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad responses (4xx, 5xx)
        
        # Store the dynamically fetched tool schemas
        app.state.dynamic_tools = response.json()
        print(f"Successfully fetched {len(app.state.dynamic_tools)} tools.")
        #
        # Example of what /tools should return:
        # [
        #   {
        #     "name": "send_command",
        #     "description": "Sends a raw command to the server.",
        #     "parameters": {
        #       "type": "object",
        #       "properties": {"command": {"type": "string"}},
        #       "required": ["command"]
        #     }
        #   },
        #   ...
        # ]
        
    except httpx.RequestError as e:
        print(f"CRITICAL ERROR: Failed to connect to MCP server at {e.request.url!r}.")
        app.state.dynamic_tools = []
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to fetch or parse tools: {e}")
        app.state.dynamic_tools = []

    yield  # --- Application is now running ---

    # --- Shutdown Logic ---
    print("--- Orchestrator App Shutting Down ---")
    await app.state.http_client.aclose()


# --- 3. FastAPI and LLM Setup ---

app = FastAPI(
    title="MCP Dynamic Orchestrator Client",
    description="Fetches tools dynamically and forwards execution to an MCP server.",
    version="2.0.0",
    lifespan=lifespan  # Use the new lifespan manager
    
)

# Initialize LLM
try:
    llm = ChatVertexAI(model="gemini-2.5-flash", temperature=0.0)
except Exception as e:
    print(f"Error initializing ChatVertexAI: {e}")
    llm = None

# Pydantic Model for API Request
class OrchestratorRequest(BaseModel):
    prompt: str = Field(
        ..., 
        example="Who is online right now?",
        description="The natural language command for the Minecraft server."
    )


# --- 4. FastAPI Endpoint with Dynamic Orchestration ---

@app.post("/chat", tags=["Orchestration"])
async def chat_with_mcp_server(
    fastapi_request: Request,          # Access the app's state
    request_body: OrchestratorRequest  # Get the user's JSON input
):
    """
    Sends a natural language prompt to the LLM and executes tool calls
    by forwarding them to the configured MCP server.
    """
    if not llm:
        return {"error": "LLM agent is not initialized. Check server logs for VertexAI configuration errors."}

    # Get the resources created during startup
    http_client = fastapi_request.app.state.http_client
    dynamic_tools = fastapi_request.app.state.dynamic_tools

    if not dynamic_tools:
        return {
            "error": "Tool list is empty. Check if the MCP server is running and if /tools is configured correctly."
        }

    # 1. Initialize message history
    messages = [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=request_body.prompt),
    ]

    try:
        # 2. First LLM Call with Dynamic Tools
        print(f"Sending prompt to LLM with {len(dynamic_tools)} tools...")
        response = llm.invoke(messages, tools=dynamic_tools)
        
        if response.tool_calls:
            messages.append(response) 

            # 3. Execute Tool Calls by forwarding to MCP Server
            for tool_call in response.tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.args
                
                print(f"LLM requested tool: {tool_name}, Args: {tool_args}")
                
                # Make the HTTP call to the MCP server's execution endpoint
                try:
                    tool_response = await http_client.post(
                        "/execute_tool",
                        json={"name": tool_name, "args": tool_args}
                    )
                    tool_response.raise_for_status()
                    
                    # Get the result from the MCP server
                    tool_output = tool_response.json().get("result", "No result returned")
                    
                except httpx.HTTPStatusError as e:
                    tool_output = f"Error executing tool: Server responded with {e.response.status_code}"
                except Exception as e:
                    tool_output = f"Error executing tool: {str(e)}"

                print(f"Tool {tool_name} returned: {tool_output}")
                
                # 4. Append the function output to messages
                messages.append(FunctionMessage(
                    content=tool_output, 
                    name=tool_name,
                    tool_call_id=tool_call.id
                ))
            
            # 5. Second LLM Call: Final response generation
            print("Sending tool results back to LLM for final summary...")
            final_response = llm.invoke(messages, tools=dynamic_tools)
            output_content = final_response.content
        else:
            # If no tool call, the first response is the final answer
            output_content = response.content

        # 6. Return the result
        return {
            "prompt": request_body.prompt,
            "orchestrator_response": output_content,
        }

    except Exception as e:
        return {
            "error": "An error occurred during orchestration.", 
            "details": str(e)
        }

# Basic root endpoint for health check
@app.get("/", tags=["Health"])
def read_root():
    return {"status": "MCP Orchestrator API is running. Awaiting /chat requests."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 


