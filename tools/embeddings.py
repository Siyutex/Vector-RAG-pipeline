import os

from lancedb.embeddings import get_registry
from lancedb.embeddings.openai import OpenAIEmbeddings
from lancedb.pydantic import Vector


@get_registry().register("openrouter")
class OpenRouterEmbeddings(OpenAIEmbeddings):
    name: str = "voyageai/voyage-code-4"
    dim: int = 1024

    @property
    def _ndims(self) -> int:
        return self.dim


def get_embedding_func():
    registry = get_registry()
    if "OPENROUTER_API_KEY" not in os.environ:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set!")
    registry.set_var("openrouter_key", os.environ["OPENROUTER_API_KEY"])
    return registry.get("openrouter").create(
        name="voyageai/voyage-code-4",
        base_url="https://openrouter.ai/api/v1",
        api_key="$var:openrouter_key",
        dim=1024,
    )