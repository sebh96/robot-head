"""A tiny MCP server for the tests: one tool that only looks something up, one that does something."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("demo")


@server.tool(annotations={"readOnlyHint": True})
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@server.tool()
def send_note(text: str) -> str:
    """Send a note to someone."""
    return f"sent: {text}"


if __name__ == "__main__":
    server.run()
