import json
import os
import sqlite3

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_openai import ChatOpenAI

from app.config import DB_DIR, FILE_STORAGE_DIR
from app.models import DocumentMetadata

DEEP_INFRA_API_KEY = None
OPENAI_API_KEY = None
CHROMA_HOST = None

# These will be initialized in initialize_env
embeddings = None
client = None
vector_store = None
llm = None

def initialize_env():
    """Initializes the environment by loading variables from .env."""
    global DEEP_INFRA_API_KEY, OPENAI_API_KEY, CHROMA_HOST
    global embeddings, client, vector_store, llm
    load_dotenv()
    DEEP_INFRA_API_KEY = os.getenv("DEEP_INFRA_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_HOST = os.getenv("CHROMA_HOST")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5-mini")
    MODEL_REASONING_EFFORT = os.getenv("MODEL_REASONING_EFFORT", "low")
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

    # Initialize vector store
    embeddings = DeepInfraEmbeddings(
        model_id=MODEL_NAME,
        query_instruction="",
        embed_instruction="",
        deepinfra_api_token=DEEP_INFRA_API_KEY,
    )
    client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    vector_store = Chroma(
        client=client,
        collection_name="example_collection",
        embedding_function=embeddings,
    )
    if MODEL_NAME == 'gpt-4.1-mini':
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        # LLM for chat
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY,
            reasoning_effort=MODEL_REASONING_EFFORT
        )

class Database:
    def __init__(self):
        self.db_path = DB_DIR / "metadata.db"
        self.file_storage_path = FILE_STORAGE_DIR
        self._conn = None

    def get_conn(self):
        """Lazily creates a database connection if one doesn't exist."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Closes the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_db(self):
        """Initializes the database schema."""
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_md5_checksum TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                type TEXT,
                category TEXT,
                tags TEXT, -- Stored as a JSON string
                parsed_at TEXT NOT NULL,
                published_date TEXT,
                userId TEXT,
                productType TEXT,
                serviceType TEXT,
                json_file_path TEXT NOT NULL
            )
        """
        )
        conn.commit()

    def add_metadata(self, metadata: DocumentMetadata):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """--sql
            INSERT INTO file_metadata (
                file_md5_checksum, file_name, type, category, tags,
                parsed_at, published_date, userId, productType, serviceType, json_file_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.id,
                metadata.name,
                metadata.type,
                metadata.category,
                json.dumps(metadata.tags),
                metadata.uploadDate.isoformat(),
                metadata.publishedDate.isoformat() if metadata.publishedDate else None,
                metadata.userId,
                metadata.productType,
                metadata.serviceType,
                metadata.json_file_path,
            ),
        )
        conn.commit()

    def get_metadata(self, md5_checksum: str) -> DocumentMetadata | None:
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM file_metadata WHERE file_md5_checksum = ?", (md5_checksum,)
        )
        row = cursor.fetchone()
        if row:
            row_dict = dict(row)
            if row_dict.get("tags"):
                row_dict["tags"] = json.loads(row_dict["tags"])
            # Manually handle the alias for md5Checksum
            row_dict["file_md5_checksum_alias"] = row_dict["file_md5_checksum"]
            return DocumentMetadata.model_validate(row_dict)
        return None

    def delete_metadata_and_files(self, file_reference: str):
        """Deletes metadata, vector embeddings, and associated files."""
        metadata = self.get_metadata(file_reference)
        if not metadata:
            return  # Or raise an error

        # 1. Delete the parsed JSON file
        try:
            if metadata.json_file_path and os.path.exists(metadata.json_file_path):
                os.remove(metadata.json_file_path)
        except OSError as e:
            print(f"Error deleting JSON file {metadata.json_file_path}: {e}")

        # 2. Delete the original file from file_storage
        try:
            file_extension = os.path.splitext(metadata.name)[1]
            stored_file_name = f"{file_reference}{file_extension}"
            stored_file_path = os.path.join(FILE_STORAGE_DIR, stored_file_name)
            if os.path.exists(stored_file_path):
                os.remove(stored_file_path)
        except OSError as e:
            print(f"Error deleting original file {stored_file_path}: {e}")

        # 3. Delete from ChromaDB by finding all associated chunk_ids
        try:
            # Query for documents with the matching file_md5_checksum
            retrieved_docs = vector_store.get(
                where={"file_md5_checksum": file_reference}
            )
            if retrieved_docs and retrieved_docs.get("ids"):
                chunk_ids_to_delete = retrieved_docs["ids"]
                if chunk_ids_to_delete:
                    vector_store.delete(ids=chunk_ids_to_delete)
                    print(
                        f"Deleted {len(chunk_ids_to_delete)} chunks from vector store."
                    )
        except Exception as e:
            print(f"Error deleting from vector store: {e}")

        # 4. Delete from SQLite
        self.delete_record(file_reference)

    def get_all_metadata(self) -> list[DocumentMetadata]:
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM file_metadata ORDER BY parsed_at DESC")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get("tags"):
                row_dict["tags"] = json.loads(row_dict["tags"])
            # Manually handle the alias for md5Checksum
            row_dict["file_md5_checksum_alias"] = row_dict["file_md5_checksum"]
            results.append(DocumentMetadata.model_validate(row_dict))
        return results

    def delete_record(self, md5_checksum: str):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM file_metadata WHERE file_md5_checksum = ?", (md5_checksum,)
        )
        conn.commit()



