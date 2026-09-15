"""Custom RAG MCP server for codebase retrieval in Continue"""

import asyncio
import os
import lancedb
from mcp.server import InitializationOptions, NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

# 1. Connect to local LanceDB instance
DB_PATH = os.getenv("LANCEDB_PATH", "/tmp/lancedb_codebase")
db = lancedb.connect(DB_PATH)
table = db.open_table("code_index")

# 2. Initialize MCP App
app = Server("custom-rag-server")


@app.tool()
async def search_codebase(query: str, limit: int = 5) -> list[TextContent]:
    """Search the codebase using vector similarity via OpenRouter embeddings.

    Args:
        query: Natural language or semantic search query (e.g., 'data scaling
          function').
        limit: Maximum number of relevant code chunks to return.
    """
    results = table.search(query).limit(limit).to_list()

    formatted_results = []
    for r in results:
        header = f"--- File: {r['filepath']} (Lines {r['start_line']}-{r['end_line']}) ---"
        body = f"{header}\n{r['text']}"
        formatted_results.append(TextContent(type="text", text=body))

    return formatted_results


@app.tool()
async def get_file_context(filepath: str) -> list[TextContent]:
    """Retrieve all indexed chunks belonging to a specific file.

    Args:
        filepath: Relative file path (e.g., 'data_processor.py').
    """
    # Escape single quotes in filenames to prevent SQL syntax errors
    safe_path = filepath.replace("'", "''")
    results = (
        table.where(f"filepath = '{safe_path}'").to_list()
    )

    if not results:
        return [
            TextContent(
                type="text", text=f"No indexed chunks found for file: {filepath}"
            )
        ]

    # Sort chunks by starting line number to reconstruct file order
    results.sort(key=lambda x: x["start_line"])
    full_text = f"=== File Context: {filepath} ===\n\n" + "\n\n".join(
        [r["text"] for r in results]
    )

    return [TextContent(type="text", text=full_text)]


# 3. Server Runner
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="custom-rag-server",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())