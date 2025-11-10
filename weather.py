

from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Weather Server")

@mcp.tool("get_weather", description="Get the current weather for a given location.")
def get_weather(location: str) -> str:
    """Returns a mock weather report for the given location."""
    return f"The current weather in {location} is Sunny, 25°C."

@mcp.resource("weather://statements", description="Provides available weather commands.")
def weather_statements()-> str:
 """Provides a list of available weather commands."""
 return "Available weather commands: get_weather(location: str) -> str"

if __name__ == "__main__":
    mcp.run()
