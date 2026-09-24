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
    """Search the indexed codebase using vector similarity.

    Embeds the provided natural-language ``query`` with the configured
    OpenRouter embedding function and returns the top ``limit`` most
    similar code chunks from the LanceDB ``code_index`` table.

    Args:
        query: Natural-language or code query string to search for.
        limit: Maximum number of matching chunks to return. Defaults to 5.

    Returns:
        A newline-separated string where each result is formatted as::

            --- File: <filepath> (Lines <start_line>-<end_line>) ---
            <chunk text>

        If the code index table has not been initialized, or if the
        underlying search raises an exception, an error message string
        is returned instead of raising.
    """
    if table is None:
        return "Error: Code index table is not initialized."

    try:
        func = get_embedding_func()  # must run before table.search so registry var is set
        results = table.search(query).limit(limit).to_list()
        formatted = [
            f"--- File: {r['filepath']} (Lines {r['start_line']}-{r['end_line']}) ---\n{r['text']}"
            for r in results
        ]
        return "\n\n".join(formatted)
    except Exception as e:
        log_debug(f"search_codebase failed: {type(e).__name__}: {e}")
        return f"Error executing search: {type(e).__name__}: {e}"


@mcp.tool()
async def get_file_context(filepath: str) -> str:
    """Retrieve every indexed chunk that belongs to a specific file.

    Looks up all rows in the ``code_index`` table whose ``filepath``
    column exactly matches the provided path, sorts them by their
    starting line number, and concatenates their raw text. This is
    useful for reconstructing the full indexed contents of a file so
    that additional context can be provided to a downstream model.

    Args:
        filepath: Path of the file (as stored at indexing time) whose
            chunks should be retrieved.

    Returns:
        A string beginning with a ``=== File Context: <filepath> ===``
        header followed by each chunk separated by blank lines. If no
        chunks are found for the file, a "No indexed chunks found"
        message is returned. If the code index table is unavailable or
        the lookup raises, an error message string is returned.
    """
    if table is None:
        return "Error: Code index table is not initialized."

    try:
        func = get_embedding_func()
        safe_path = filepath.replace("'", "''")
        results = table.where(f"filepath = '{safe_path}'").to_list()
        if not results:
            return f"No indexed chunks found for file: {filepath}"

        results.sort(key=lambda x: x["start_line"])
        return f"=== File Context: {filepath} ===\n\n" + "\n\n".join(r["text"] for r in results)
    except Exception as e:
        log_debug(f"get_file_context failed: {type(e).__name__}: {e}")
        return f"Error retrieving file context: {type(e).__name__}: {e}"


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