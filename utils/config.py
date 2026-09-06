import os
from dotenv import load_dotenv

load_dotenv()

# Embedding and LLM configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
USE_HUGGINGFACE = os.getenv("USE_HUGGINGFACE", "false").lower() == "true"

# Storage paths
STORAGE_DIR = os.getenv("STORAGE_DIR", "data")
os.makedirs(STORAGE_DIR, exist_ok=True)

import hashlib

def get_repo_storage_dir(repo_url: str) -> dict:
    """Returns a dictionary of isolated file paths for a given repository."""
    repo_hash = hashlib.md5(repo_url.encode('utf-8')).hexdigest()
    repo_dir = os.path.join(STORAGE_DIR, repo_hash)
    os.makedirs(repo_dir, exist_ok=True)
    
    return {
        "repo_dir": repo_dir,
        "vector_index": os.path.join(repo_dir, "faiss_index"),
        "graph": os.path.join(repo_dir, "graph.gpickle"),
        "chunks": os.path.join(repo_dir, "chunks.jsonl")
    }
