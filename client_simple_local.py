from mcp import ClientSession, StdioServerParameters,types
from mcp.client.stdio import stdio_client
import asyncio
import traceback

server_params = StdioServerParameters(
    command="uv",
    args=["--directory","/Users/anil/work/python_work/mcp_elastic_search/", "run","mcp_elastic.py"],
   # args=["--directory","/Users/anil/work/python_work/mcp_client/", "run","weather.py"],
)


async def run():
    try:
        print("Starting MCP client...")
        async with stdio_client(server_params) as (read,write):
            print("Client Connected...")
            async with ClientSession(read, write) as session:
                print("Session started...")
                await session.initialize()
                print("Session initialized...")

                print("Listing Tools...")
                tools = await session.list_tools()
                #print(f"Found {len(tools)} tools:")
                print(f"Available Tools: {tools}")

                print("Invoking 'elastic' tool...")
                result=await session.call_tool(
                    "get_mapping",arguments={
                        "index": "my_documents"})
                print("Tool Result:", result)

                result=await session.call_tool(
                    "plan_query",arguments={
                        "nl": "Count the number of books",
                        "indices": ["my_documents"]})
                print("Tool Result:", result)

                result=await session.call_tool(
                    "plan_query",arguments={
                        "nl": "Find all documents about machine learning",
                        "indices": ["my_documents"]})
                print("Tool Result:", result)


    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
    
if __name__ == "__main__":
    asyncio.run(run())