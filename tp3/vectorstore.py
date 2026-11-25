from functools import lru_cache

import chromadb

import config


@lru_cache(maxsize=1)
def get_collection():
    """Open the persistent DB once and return the same collection every call."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )