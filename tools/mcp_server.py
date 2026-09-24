"""Custom RAG MCP server for codebase retrieval in Continue (MCP v2)"""

import os
import sys
import lancedb
import asyncio
from mcp.server.mcpserver import MCPServer
from embeddings import get_embedding_func


def log_debug(msg: str):
    print(f"[DEBUG mcp_server] {msg}", file=sys.stderr, flush=True)

log_debug("Script execution started.")

# 1. Database Setup
try:
    DB_PATH = os.getenv("LANCEDB_PATH", "./lancedb_index")
    db = lancedb.connect(DB_PATH)
    
    # LanceDB v2 fix: list_tables() replaces table_names()
    existing_tables = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    
    # LanceDB >=0.38 returns ListTablesResponse (has .tables); older versions return list[str]
    if hasattr(existing_tables, "tables"):
        existing_tables = existing_tables.tables

    if "code_index" in existing_tables:
        table = db.open_table("code_index")
    else:
        table = None
        log_debug("WARNING: Table 'code_index' does not exist yet.")
except Exception as e:
    log_debug(f"FATAL ERROR during DB setup: {e}")
    sys.exit(1)

# 2. Instantiate MCP Server
mcp = MCPServer("custom-rag-server")

@mcp.tool()
async def search_codebase(query: str, limit: int = 5) -> str:
    """Search the codebase using vector similarity via OpenRouter embeddings.

    Args:
        query: Natural language search query.
        limit: Maximum number of chunks to return.
    """
    # instantiate embedding function
    func = get_embedding_func()

    if table is None:
        return "Error: Code index table is not initialized."

    try:
        results = table.search(query).limit(limit).to_list()
        formatted = [
            f"--- File: {r['filepath']} (Lines {r['start_line']}-{r['end_line']}) ---\n{r['text']}"
            for r in results
        ]
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error executing search: {str(e)}"

@mcp.tool()
async def get_file_context(filepath: str) -> str:
    """Retrieve all indexed chunks belonging to a specific file.

    Args:
        filepath: Relative file path.
    """

    # instantiate embedding function
    func = get_embedding_func()
    
    if table is None:
        return "Error: Code index table is not initialized."

    try:
        safe_path = filepath.replace("'", "''")
        results = table.where(f"filepath = '{safe_path}'").to_list()
        if not results:
            return f"No indexed chunks found for file: {filepath}"

        results.sort(key=lambda x: x["start_line"])
        return f"=== File Context: {filepath} ===\n\n" + "\n\n".join(r["text"] for r in results)
    except Exception as e:
        return f"Error retrieving file context: {str(e)}"


if __name__ == "__main__":
    log_debug("Starting MCPServer stdio loop...")
    
    # FastMCP/MCPServer run handles stdio by default when passed transport or run as async
    try:
        # If using MCPServer / FastMCP v2 interface
        if hasattr(mcp, "run_stdio_async"):
            asyncio.run(mcp.run_stdio_async())
        else:
            # Standard async runner for mcp.run()
            asyncio.run(mcp.run())
    except KeyboardInterrupt:
        log_debug("Server stopped by user.")
    except Exception as e:
        log_debug(f"Server crashed: {e}")