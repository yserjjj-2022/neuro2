"""Memory module — episodic memory with SQLite + sqlite-vec.

Re-exports:
    Episode — frozen dataclass for a memory episode
    cosine_similarity — pure functional core
    serialize_embedding / deserialize_embedding — pure core
    content_hash — pure core (SHA-256 for dedup)
    MemoryStoreError — custom exception wrapping sqlite3 errors
    MemoryStore — imperative shell (SQLite + sqlite-vec)
    SupportsStore / SupportsRecall — Protocol for DI
"""

from .errors import MemoryStoreError
from .hash import content_hash
from .models import Episode
from .protocols import SupportsRecall, SupportsStore
from .serialize import deserialize_embedding, serialize_embedding
from .similarity import cosine_similarity
from .store import MemoryStore

__all__ = [
    "Episode",
    "MemoryStore",
    "MemoryStoreError",
    "SupportsRecall",
    "SupportsStore",
    "content_hash",
    "cosine_similarity",
    "deserialize_embedding",
    "serialize_embedding",
]
