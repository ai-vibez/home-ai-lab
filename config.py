import os

# Paths
DOCS_DIR = os.path.expanduser("~/lab/rag-docs")
MANIFEST_FILE = os.path.expanduser("~/lab/.ingest_manifest.json")

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION = "lab_docs"
DENSE_DIM = 1024

# Models
EMBED_MODEL = "mxbai-embed-large"
SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_LLM_MODEL = "deepseek-r1:7b"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Ingest
UPSERT_BATCH_SIZE = 100

# Query
DEFAULT_LIMIT = 10
MAX_CONTEXT_CHARS = 18000
LLAMA_SERVER_URL = "http://localhost:8081"

import requests as _requests
def get_active_model():
    try:
        r = _requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        models = r.json().get("models", [])
        if models:
            return models[0]["name"]
    except Exception:
        pass
    return "unknown"

DEFAULT_LLM_MODEL = get_active_model()
