import os
from pathlib import Path

# Root of the is-vector-search project
ROOT_DIR = Path(__file__).resolve().parent.parent

# Base data path (can be overridden if you want to separate from mcp-server)
DATA_DIR = Path(os.getenv("IS_VECTOR_SEARCH_DATA_PATH", ROOT_DIR))

# Paths for the database and file storage (used by Supabase-related utils)
DB_DIR = DATA_DIR / "hermes_db"
FILE_STORAGE_DIR = DATA_DIR / "file_storage"

DB_DIR.mkdir(parents=True, exist_ok=True)
FILE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "127.0.0.1:3001",
    "*",
]
