"""Memory module — episodic memory with SQLite + sqlite-vec.

Re-exports:
    Episode — frozen dataclass for a memory episode
    cosine_similarity — pure functional core
    serialize_embedding / deserialize_embedding — pure core
    content_hash — pure core (SHA-256 for dedup)
    MemoryStoreError — custom exception wrapping sqlite3 errors
    MemoryStore — imperative shell (SQLite + sqlite-vec)
    SupportsStore / SupportsRecall — Protocol for DI
    Embedder / FakeEmbedder / ApiEmbedder / build_embedder / EmbedderError
    is_significant_event / build_event_content — significance core
    MEMORY_PRIOR_DIM / encode_memory_prior — memory prior core
    MemoryRouter — shell orchestrating recall→prior and episode storage
"""

from .embedder import (
    DEFAULT_API_DIM,
    DEFAULT_API_MODEL,
    DEFAULT_BASE_URL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_DIM,
    ENV_MODEL,
    ApiEmbedder,
    Embedder,
    EmbedderError,
    FakeEmbedder,
    build_embedder,
    embedder_settings_from_env,
)
from .errors import MemoryStoreError
from .events import build_event_content, is_significant_event
from .hash import content_hash
from .models import Episode
from .prior import MEMORY_PRIOR_DIM, encode_memory_prior
from .protocols import SupportsRecall, SupportsStore
from .router import MemoryRouter
from .serialize import deserialize_embedding, serialize_embedding
from .similarity import cosine_similarity
from .store import MemoryStore

__all__ = [
    "DEFAULT_API_DIM",
    "DEFAULT_API_MODEL",
    "DEFAULT_BASE_URL",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_DIM",
    "ENV_MODEL",
    "MEMORY_PRIOR_DIM",
    "ApiEmbedder",
    "Embedder",
    "EmbedderError",
    "Episode",
    "FakeEmbedder",
    "MemoryRouter",
    "MemoryStore",
    "MemoryStoreError",
    "SupportsRecall",
    "SupportsStore",
    "build_embedder",
    "build_event_content",
    "content_hash",
    "cosine_similarity",
    "deserialize_embedding",
    "embedder_settings_from_env",
    "encode_memory_prior",
    "is_significant_event",
    "serialize_embedding",
]
