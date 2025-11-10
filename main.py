def main():
    print("Hello from mcp-client!")


if __name__ == "__main__":
    main()


curl -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "MCP-Session: asdsdsdsadasdasdasdasd" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
