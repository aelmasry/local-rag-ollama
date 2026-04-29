import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "faiss_index"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
RETRIEVER_K = 5

# First N characters of each uploaded file are indexed as a dedicated high-signal chunk.
DOCUMENT_HEADER_CHAR_LIMIT = 500
