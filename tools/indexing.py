import os
import pathspec
import lancedb
from lancedb.embeddings import get_registry
from lancedb.embeddings.openai import OpenAIEmbeddings
from lancedb.pydantic import LanceModel, Vector

from embeddings import get_embedding_func
func = get_embedding_func()

"""# 1. Custom OpenRouter Embedding Class
@get_registry().register("openrouter")
class OpenRouterEmbeddings(OpenAIEmbeddings):
    name: str = "voyageai/voyage-code-4"
    dim: int = 1024

    @property
    def _ndims(self) -> int:
        return self.dim


# Registry Setup
registry = get_registry()
if "OPENROUTER_API_KEY" not in os.environ:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set!")
registry.set_var("openrouter_key", os.environ["OPENROUTER_API_KEY"])

func = registry.get("openrouter").create(
    name="voyageai/voyage-code-4",
    base_url="https://openrouter.ai/api/v1",
    api_key="$var:openrouter_key",
    dim=1024,
)"""


# 2. Database Schema
class CodeChunkSchema(LanceModel):
    filepath: str
    start_line: int
    end_line: int
    text: str = func.SourceField()
    vector: Vector(1024) = func.VectorField()


# 3. Language-Agnostic Fixed-Length Sliding Window Chunker
def chunk_text_by_lines(
    text: str, chunk_size: int = 60, overlap: int = 15
) -> list[dict]:
    """Splits any text or code file into overlapping line chunks."""
    lines = text.splitlines()
    if not lines:
        return []

    # Strategy Option 1: Truncate single-chunk small files (1 file = 1 chunk)
    if len(lines) <= chunk_size:
        return [
            {
                "start_line": 1,
                "end_line": len(lines),
                "text": text,
            }
        ]

    # Strategy Option 2: Sliding Window over larger files
    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i : i + chunk_size]
        chunk_text = "\n".join(chunk_lines)

        if chunk_text.strip():
            chunks.append(
                {
                    "start_line": i + 1,
                    "end_line": i + len(chunk_lines),
                    "text": chunk_text,
                }
            )

        # Move the sliding window forward
        i += chunk_size - overlap

    return chunks


# 4. Universal Repository Traversal
def load_codebase_chunks(
    root_dir: str, ignore_patterns: list[str] = None
) -> list[dict]:
    if ignore_patterns is None:
        ignore_patterns = [
            ".git/",
            "__pycache__/",
            ".devenv/",
            "venv/",
            "*.pyc",
            "*.lance",
            "*.db",
            "node_modules/",
            "dist/",
            "build/",
            ".idea/",
            ".vscode/",
        ]

    spec = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)
    all_chunks = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root_dir)

            if spec.match_file(rel_path):
                continue

            # Safely skip non-text/binary files (images, compiled libs, etc.)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError):
                continue

            # Chunk any valid text file (C++, Java, Python, Markdown, JSON, etc.)
            file_chunks = chunk_text_by_lines(content)
            for chunk in file_chunks:
                all_chunks.append(
                    {
                        "filepath": rel_path,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "text": chunk["text"],
                    }
                )

    return all_chunks


# 5. Pipeline Execution
def index_codebase(target_directory: str, db_path: str = "./lancedb_index"):
    print(f"Indexing repository at {target_directory}...")
    chunks = load_codebase_chunks(target_directory)
    print(f"Generated {len(chunks)} chunks across repository files.")

    if not chunks:
        print("No indexable code found.")
        return

    db = lancedb.connect(db_path)
    table = db.create_table("code_index", schema=CodeChunkSchema, mode="overwrite")

    print("Embedding and writing chunks to LanceDB via OpenRouter...")
    table.add(chunks)
    print("Indexing complete!")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    index_codebase(target)