import json
import os
from datetime import datetime
from io import BytesIO
from typing import List, Optional

import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DeepInfraEmbeddings
from langchain_openai import ChatOpenAI
from supabase import Client, create_client

from app.config import FILE_STORAGE_DIR
from app.models import DocumentMetadata

# Global variables for initialization
DEEP_INFRA_API_KEY = None
OPENAI_API_KEY = None
CHROMA_HOST = None
SUPABASE_URL = None
SUPABASE_KEY = None

# These will be initialized in initialize_env
embeddings = None
client = None
vector_store = None
llm = None
supabase: Optional[Client] = None
df_hermes = None
df_instrument = None
df_phatrax = None
df_out = None
df_tran = None
df_cust = None


def initialize_env():
    """Initializes the environment by loading variables from .env."""
    global DEEP_INFRA_API_KEY, OPENAI_API_KEY, CHROMA_HOST, SUPABASE_URL, SUPABASE_KEY
    global embeddings, client, vector_store, llm, supabase
    global df_hermes, df_instrument, df_phatrax, df_out, df_tran, df_cust
    load_dotenv()
    DEEP_INFRA_API_KEY = os.getenv("DEEP_INFRA_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_HOST = os.getenv("CHROMA_HOST")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
    USE_SUPABASE_AS_SOURCE = (
        os.getenv("USE_SUPABASE_AS_SOURCE", "True").lower() == "true"
    )
    SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "mcp")
    SUPABASE_STORAGE_PREFIX = os.getenv(
        "SUPABASE_STORAGE_PREFIX", ""
    )  # e.g. "data" or ""

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Initialize vector store (keeping Chroma as-is)
    embeddings = DeepInfraEmbeddings(
        model_id="BAAI/bge-m3",
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

    # LLM for chat
    llm = ChatOpenAI(
        model="gpt-4.1-mini", temperature=0.1, openai_api_key=OPENAI_API_KEY
    )

    def _download_parquet_from_storage(filename: str) -> pd.DataFrame:
        path = (
            f"{SUPABASE_STORAGE_PREFIX}/{filename}"
            if SUPABASE_STORAGE_PREFIX
            else filename
        )
        content = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).download(path)
        return pd.read_parquet(BytesIO(content))

    if USE_SUPABASE_AS_SOURCE:
        # From Supabase Storage bucket
        df_hermes = _download_parquet_from_storage("hermes_files_conso.parquet")
        df_instrument = _download_parquet_from_storage("df_instrument_all.parquet")
        df_phatrax = _download_parquet_from_storage("df_phatrax.parquet")
        df_out = _download_parquet_from_storage("df_out_new.parquet")
        df_tran = _download_parquet_from_storage("df_tran.parquet")
        df_cust = _download_parquet_from_storage("df_cust_mock.parquet")
    else:
        # From local filesystem under data/
        df_hermes = pd.read_parquet("data/hermes_files_conso.parquet")
        df_instrument = pd.read_parquet("data/df_instrument_all.parquet")
        df_phatrax = pd.read_parquet("data/df_phatrax.parquet")
        df_out = pd.read_parquet("data/df_out.parquet")
        df_tran = pd.read_parquet("data/df_tran.parquet")
        df_cust = pd.read_parquet("data/df_cust_mock.parquet")

    # Post-load normalizations matching existing tool expectations
    if "pageSections_description_for_chatbot" in df_hermes.columns:
        df_hermes["pageSections_description_for_chatbot"] = df_hermes[
            "pageSections_description_for_chatbot"
        ].fillna("")

    # Instrument dataset normalization for MarketData tools
    if df_instrument is not None:
        if "symbol" in df_instrument.columns:
            df_instrument["symbol"] = df_instrument["symbol"].astype(str)
        # Ensure symbol_norm exists for fuzzy search
        if (
            "symbol_norm" not in df_instrument.columns
            and "symbol" in df_instrument.columns
        ):
            df_instrument["symbol_norm"] = (
                df_instrument["symbol"]
                .str.replace(r"[^A-Za-z0-9]", "", regex=True)
                .str.lower()
            )
        # Parse top pick dates if present
        for _col in ("top_pick_since", "top_pick_until"):
            if _col in df_instrument.columns:
                df_instrument[_col] = pd.to_datetime(
                    df_instrument[_col], errors="coerce"
                )

    # Portfolio datasets shape adjustments
    df_tran = df_tran.sort_values(
        by=["trade_date", "asset_class", "src_symbol"], ascending=[True, True, True]
    )
    # Convert to JSON-friendly dict
    df_out = df_out.where(pd.notnull(df_out), None)
    df_out["expected_return"] = df_out["expected_return"].fillna(0)
    if "trade_date" in df_tran.columns:
        df_tran["trade_date"] = pd.to_datetime(
            df_tran["trade_date"]
        )  # used later with dt.strftime
    if "client_open_date" in df_cust.columns:
        df_cust["client_open_date"] = pd.to_datetime(df_cust["client_open_date"])
    # Keep column rename and typing consistent with tools
    if "client_first_name_en" in df_cust.columns:
        df_cust = df_cust.rename(columns={"client_first_name_en": "customer_name"})
    if "customer_id" in df_cust.columns:
        df_cust["customer_id"] = df_cust["customer_id"].astype(str)


def normalize_instrument_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the given instrument DataFrame with:
    - date columns ('top_pick_since', 'top_pick_until') converted to ISO strings (YYYY-MM-DD), NaT -> None
    Note: 'mfet_data' parsing is no longer needed because instrument metadata is flattened into columns.
    """
    df_copy = df.copy()
    # Convert date columns to ISO strings (python str) with None for missing
    for date_col in ("top_pick_since", "top_pick_until"):
        if date_col in df_copy.columns:
            ser = pd.to_datetime(df_copy[date_col], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
            # ser will contain strings and NaT; convert NaT/NaN to None explicitly
            df_copy[date_col] = ser.where(ser.notna(), None)

    return df_copy


class SupabaseDatabase:
    """Database class that uses Supabase instead of SQLite."""

    def __init__(self):
        self.file_storage_path = FILE_STORAGE_DIR
        self.supabase = supabase

    def init_db(self):
        """Initialize database schema - handled by Supabase migrations."""
        # Schema initialization is handled by Supabase migrations
        # No need to create tables programmatically
        pass

    def add_metadata(self, metadata: DocumentMetadata) -> bool:
        """Add metadata to Supabase."""
        try:
            # Convert DocumentMetadata to dict for Supabase
            data = {
                "file_md5_checksum": metadata.file_md5_checksum,
                "file_name": metadata.file_name,
                "type": metadata.type,
                "category": metadata.category,
                "tags": json.dumps(metadata.tags) if metadata.tags else None,
                "parsed_at": metadata.parsed_at.isoformat(),
                "published_date": (
                    metadata.published_date.isoformat()
                    if metadata.published_date
                    else None
                ),
                "user_id": metadata.user_id,
                "product_type": metadata.product_type,
                "service_type": metadata.service_type,
                "json_file_path": metadata.json_file_path,
                "file_status": getattr(metadata, "file_status", None),
                "json_status": getattr(metadata, "json_status", None),
            }

            response = self.supabase.table("file_metadata").upsert(data).execute()
            return len(response.data) > 0

        except Exception as e:
            print(f"Error adding metadata to Supabase: {e}")
            return False

    def get_metadata(self, md5_checksum: str) -> Optional[DocumentMetadata]:
        """Get metadata from Supabase by MD5 checksum."""
        try:
            response = (
                self.supabase.table("file_metadata")
                .select("*")
                .eq("file_md5_checksum", md5_checksum)
                .execute()
            )

            if response.data and len(response.data) > 0:
                row = response.data[0]

                # Handle tags parsing
                tags = []
                if row.get("tags"):
                    if isinstance(row["tags"], str):
                        tags = json.loads(row["tags"])
                    else:
                        tags = row["tags"]

                # Create DocumentMetadata object
                metadata_data = {
                    "id": row.get("id"),
                    "file_md5_checksum": row["file_md5_checksum"],
                    "file_name": row["file_name"],
                    "type": row["type"],
                    "category": row["category"],
                    "tags": tags,
                    "parsed_at": datetime.fromisoformat(row["parsed_at"]),
                    "published_date": (
                        datetime.fromisoformat(row["published_date"])
                        if row["published_date"]
                        else None
                    ),
                    "user_id": row["user_id"],
                    "product_type": row["product_type"],
                    "service_type": row["service_type"],
                    "json_file_path": row["json_file_path"],
                    "file_status": row.get("file_status"),
                    "json_status": row.get("json_status"),
                    "created_at": (
                        datetime.fromisoformat(row["created_at"])
                        if row.get("created_at")
                        else None
                    ),
                    "updated_at": (
                        datetime.fromisoformat(row["updated_at"])
                        if row.get("updated_at")
                        else None
                    ),
                }

                return DocumentMetadata.model_validate(metadata_data)

        except Exception as e:
            print(f"Error getting metadata from Supabase: {e}")

        return None

    def add_file_storage(self, data: dict) -> bool:
        """Insert a record into file_storage table.

        Expected keys in data:
        - file_md5_checksum (str)
        - storage_path (str)
        - file_name (str)
        - file_size (int) optional
        - content_type (str) optional
        - bucket_name (str) default 'file-storage' handled by DB default if omitted
        - metadata (dict) optional
        """
        try:
            payload = {
                "file_md5_checksum": data["file_md5_checksum"],
                "storage_path": data["storage_path"],
                "file_name": data["file_name"],
                "file_size": data.get("file_size"),
                "content_type": data.get("content_type"),
                "bucket_name": data.get("bucket_name", "file-storage"),
                "metadata": (
                    json.dumps(data.get("metadata"))
                    if isinstance(data.get("metadata"), (dict, list))
                    else data.get("metadata")
                ),
            }
            response = self.supabase.table("file_storage").insert(payload).execute()
            return bool(response.data)
        except Exception as e:
            print(f"Error inserting into file_storage: {e}")
            return False

    def delete_metadata_and_files(self, file_reference: str) -> bool:
        """Delete metadata, vector embeddings, and associated files."""
        try:
            metadata = self.get_metadata(file_reference)
            if not metadata:
                return False

            # 1. Delete the parsed JSON file
            try:
                if metadata.json_file_path:
                    # New scheme: JSON stored in Supabase Storage under the 'parsed-json' bucket
                    # with a virtual URI like 'supabase://parsed-json/<md5>.json'.
                    if metadata.json_file_path.startswith("supabase://parsed-json/"):
                        filename = metadata.json_file_path.replace(
                            "supabase://parsed-json/", ""
                        )
                        try:
                            self.supabase.storage.from_("parsed-json").remove(
                                [filename]
                            )
                        except Exception as e:
                            print(
                                f"Error deleting JSON from Supabase 'parsed-json' bucket ({filename}): {e}"
                            )
                    # Backward compatibility: local filesystem path
                    elif os.path.exists(metadata.json_file_path):
                        os.remove(metadata.json_file_path)
            except OSError as e:
                print(f"Error deleting JSON file {metadata.json_file_path}: {e}")

            # 2. Delete from Supabase Storage
            try:
                # Get file storage record
                storage_response = (
                    self.supabase.table("file_storage")
                    .select("*")
                    .eq("file_md5_checksum", file_reference)
                    .execute()
                )

                if storage_response.data:
                    for storage_record in storage_response.data:
                        # Delete from Supabase Storage
                        file_path = storage_record["storage_path"]
                        bucket_name = storage_record["bucket_name"]

                        self.supabase.storage.from_(bucket_name).remove([file_path])

                        # Delete storage record
                        self.supabase.table("file_storage").delete().eq(
                            "id", storage_record["id"]
                        ).execute()

            except Exception as e:
                print(f"Error deleting from Supabase Storage: {e}")

            # 3. Delete from ChromaDB
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

            # 4. Delete from Supabase file_metadata
            self.supabase.table("file_metadata").delete().eq(
                "file_md5_checksum", file_reference
            ).execute()

            return True

        except Exception as e:
            print(f"Error deleting metadata and files: {e}")
            return False

    def get_all_metadata(self) -> List[DocumentMetadata]:
        """Get all metadata from Supabase."""
        try:
            response = (
                self.supabase.table("file_metadata")
                .select("*")
                .order("parsed_at", desc=True)
                .execute()
            )

            results = []
            for row in response.data:
                # Handle tags parsing
                tags = []
                if row.get("tags"):
                    if isinstance(row["tags"], str):
                        tags = json.loads(row["tags"])
                    else:
                        tags = row["tags"]

                # Create DocumentMetadata object
                metadata_data = {
                    "id": row.get("id"),
                    "file_md5_checksum": row["file_md5_checksum"],
                    "file_name": row["file_name"],
                    "type": row["type"],
                    "category": row["category"],
                    "tags": tags,
                    "parsed_at": datetime.fromisoformat(row["parsed_at"]),
                    "published_date": (
                        datetime.fromisoformat(row["published_date"])
                        if row["published_date"]
                        else None
                    ),
                    "user_id": row["user_id"],
                    "product_type": row["product_type"],
                    "service_type": row["service_type"],
                    "json_file_path": row["json_file_path"],
                    "created_at": (
                        datetime.fromisoformat(row["created_at"])
                        if row.get("created_at")
                        else None
                    ),
                    "updated_at": (
                        datetime.fromisoformat(row["updated_at"])
                        if row.get("updated_at")
                        else None
                    ),
                }

                results.append(DocumentMetadata.model_validate(metadata_data))

            return results

        except Exception as e:
            print(f"Error getting all metadata from Supabase: {e}")
            return []

    def delete_record(self, md5_checksum: str) -> bool:
        """Delete a record from file_metadata."""
        try:
            response = (
                self.supabase.table("file_metadata")
                .delete()
                .eq("file_md5_checksum", md5_checksum)
                .execute()
            )
            return len(response.data) > 0
        except Exception as e:
            print(f"Error deleting record from Supabase: {e}")
            return False

    def upload_file_to_storage(
        self,
        file_content: bytes,
        file_name: str,
        content_type: str = "application/octet-stream",
        user_id: str = None,
    ) -> Optional[str]:
        """Upload file to Supabase Storage."""
        try:
            # Generate unique filename
            import uuid

            file_extension = os.path.splitext(file_name)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"

            # Upload to Supabase Storage
            response = self.supabase.storage.from_("documents").upload(
                unique_filename, file_content, {"content-type": content_type}
            )

            if response.path:
                return response.path

        except Exception as e:
            print(f"Error uploading file to Supabase Storage: {e}")

        return None

    def get_file_from_storage(self, file_path: str) -> Optional[bytes]:
        """Get file from Supabase Storage."""
        try:
            response = self.supabase.storage.from_("documents").download(file_path)
            return response
        except Exception as e:
            print(f"Error downloading file from Supabase Storage: {e}")
            return None

    def update_metadata(
        self, file_reference: str, updates: dict
    ) -> Optional[DocumentMetadata]:
        """Update metadata for a document."""
        try:
            # Prepare update data
            update_data = {}
            print(f"update: {updates}")
            # Map field names from API format to database format
            field_mapping = {
                "name": "file_name",
                "type": "type",
                "category": "category",
                "tags": "tags",
                "publishedDate": "published_date",
                "userId": "user_id",
                "productType": "product_type",
                "serviceType": "service_type",
                "fileMd5Checksum": "file_md5_checksum",
                "fileStatus": "file_status",
                "jsonStatus": "json_status",
                "jsonFilePath": "json_file_path",
            }

            # Process updates
            for api_field, db_field in field_mapping.items():
                if api_field in updates:
                    value = updates[api_field]
                    if api_field == "tags":
                        # Handle tags - ensure it's a list and serialize to JSON
                        if isinstance(value, list):
                            tag_list = [tag.strip() for tag in value if tag.strip()]
                        else:
                            tag_list = [
                                tag.strip()
                                for tag in str(value).split(",")
                                if tag.strip()
                            ]
                        update_data[db_field] = (
                            json.dumps(tag_list) if tag_list else None
                        )
                    elif api_field == "publishedDate":
                        # Handle date formatting
                        if value:
                            try:
                                if isinstance(value, str):
                                    update_data[db_field] = datetime.fromisoformat(
                                        value.replace("Z", "+00:00")
                                    ).isoformat()
                                else:
                                    update_data[db_field] = value.isoformat()
                            except (ValueError, AttributeError):
                                update_data[db_field] = value
                    else:
                        update_data[db_field] = value

            if not update_data:
                return None

            # Update the record in Supabase
            response = (
                self.supabase.table("file_metadata")
                .update(update_data)
                .eq("file_md5_checksum", file_reference)
                .execute()
            )

            if response.data and len(response.data) > 0:
                # Return the updated metadata as DocumentMetadata object
                row = response.data[0]

                # Handle tags parsing
                tags = []
                if row.get("tags"):
                    if isinstance(row["tags"], str):
                        tags = json.loads(row["tags"])
                    else:
                        tags = row["tags"]

                metadata_data = {
                    "id": row.get("id"),
                    "file_md5_checksum": row["file_md5_checksum"],
                    "file_name": row["file_name"],
                    "type": row["type"],
                    "category": row["category"],
                    "tags": tags,
                    "parsed_at": datetime.fromisoformat(row["parsed_at"]),
                    "published_date": (
                        datetime.fromisoformat(row["published_date"])
                        if row["published_date"]
                        else None
                    ),
                    "user_id": row["user_id"],
                    "product_type": row["product_type"],
                    "service_type": row["service_type"],
                    "json_file_path": row["json_file_path"],
                    "created_at": (
                        datetime.fromisoformat(row["created_at"])
                        if row.get("created_at")
                        else None
                    ),
                    "updated_at": (
                        datetime.fromisoformat(row["updated_at"])
                        if row.get("updated_at")
                        else None
                    ),
                }

                return DocumentMetadata.model_validate(metadata_data)

        except Exception as e:
            print(f"Error updating metadata in Supabase: {e}")

        return None
