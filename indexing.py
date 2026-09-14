import os
import lancedb
from lancedb.embeddings import get_registry
from lancedb.embeddings.openai import OpenAIEmbeddings
from lancedb.pydantic import LanceModel, Vector

# 1. Extend OpenAIEmbeddings to bypass the hardcoded model name checks
@get_registry().register("openrouter")
class OpenRouterEmbeddings(OpenAIEmbeddings):
    name: str = "voyageai/voyage-code-4"
    dim: int = 1024

    @property
    def _ndims(self) -> int:
        # Always return the explicitly defined dimension
        return self.dim


# 2. Setup database connection and secrets
db = lancedb.connect("/tmp/db")
registry = get_registry()

# Ensure the key exists in your environment
if "OPENROUTER_API_KEY" not in os.environ:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set!")

registry.set_var("openrouter_key", os.environ["OPENROUTER_API_KEY"])

# 3. Reference the variable string using the `$var:` syntax
func = registry.get("openrouter").create(
    name="voyageai/voyage-code-4",
    base_url="https://openrouter.ai/api/v1",
    api_key="$var:openrouter_key",
    dim=1024,
)


# 3. Define the database schema using Pydantic
class CodeChunks(LanceModel):
    filename: str
    text: str = func.SourceField()
    # Voyage Code 4 uses 1024 dimensions by default
    vector: Vector(1024) = func.VectorField()


# 4. Create the table
table = db.create_table("code_chunks", schema=CodeChunks, mode="overwrite")

# 5. Ingest sample data (LanceDB calculates embeddings via OpenRouter automatically)
table.add(
    [
        {"text": "print('hello world!')", "filename": "hello.py"},
        {"text": "print('goodbye world!')", "filename": "goodbye.py"},
        {"text": "print('I'm sorry')", "filename": "sorry.py"},
    ]
)

# 6. Query the database using semantic search
query = "apologies"
actual = table.search(query).limit(1).to_pydantic(CodeChunks)[0]

print(actual.text)







