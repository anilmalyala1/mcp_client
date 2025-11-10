from mcp import ClientSession, StdioServerParameters, types 
from mcp.client.stdio import stdio_client
import asyncio
import traceback
import json

import os
from dotenv import load_dotenv,find_dotenv

from openai import OpenAI

load_dotenv(find_dotenv())
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not found"

server_params = StdioServerParameters(
    command="uv",
    args=["--directory","/Users/anil/work/python_work/mcp_elastic_search/", "run","mcp_elastic.py"],
   # args=["--directory","/Users/anil/work/python_work/mcp_client/", "run","weather.py"],
)

SYSTEM_PROMPT = (
    "You are an autonomous retrieval assistant that answers user questions using the MCP "
    "tools provided. Always inspect available tools and their input schema before deciding "
    "whether to call one; use them whenever they can improve factual accuracy, especially "
    "for Elasticsearch lookups. When you call a tool, explain to yourself why you need it, "
    "wait for the tool result, and incorporate that content into your next response. Cite "
    "tool outputs explicitly in your final answer (mention the tool name and what it returned). "
    "If no tool is sufficient, say so and explain what additional data you would need. Keep "
    "final responses concise, structured, and focused on the user's request."
)


async def run(query):
    try:
        print("Starting stdio_client...")
        async with stdio_client(server_params) as (read, write):
            print("Client connected, creating session...")
            async with ClientSession(read, write) as session:

                # Initialize server
                print("Initializing session...")
                await session.initialize()

                # Get tools
                print("Listing tools...")
                tools_result = await session.list_tools()
                print("Available tools:", tools_result)

                openai_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                    for tool in tools_result.tools
                ]

                # Make OpenAI LLM call
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query}
                ]

                client = OpenAI()
                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                messages.append(response.choices[0].message)

                # Handle any tool calls
                if response.choices[0].message.tool_calls:
                    for tool_execution in response.choices[0].message.tool_calls:
                        # Execute tool call
                        result = await session.call_tool(
                            tool_execution.function.name,
                            arguments=json.loads(tool_execution.function.arguments),
                        )

                        # Add tool response to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_execution.id,
                                "content": result.content[0].text,
                            }
                        )
                else:
                    return response.choices[0].message.content


                # Get final response from LLM
                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                return response.choices[0].message.content

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == "__main__":
    query = 'Get all books from my_documents index'
    result=asyncio.run(run(query))
    print("Final Result:", result)
