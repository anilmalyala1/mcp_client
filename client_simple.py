from mcp import ClientSession, StdioServerParameters,types
from mcp.client.stdio import stdio_client
import asyncio
import traceback

server_params = StdioServerParameters(
    command="npx",
    args=["-y","@openbnb/mcp-server-airbnb", "--ignore-robots-txt"],
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

                print("Invoking 'airbnb_search' tool...")
                result=await session.call_tool(
                    "airbnb_search",arguments={
                        "location": "Singapore"})
                print("Tool Result:", result)


    except Exception as e:
        print("An error occurred:", str(e))
        traceback.print_exc()
    
if __name__ == "__main__":
    asyncio.run(run())