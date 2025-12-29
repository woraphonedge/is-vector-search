"""
Migration script to migrate data from Chroma SQLite to PostgreSQL with pgvector
"""

import asyncio
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List

import asyncpg
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_DB_PATH = "./hermes_db/chroma.sqlite3"
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "vector_search"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}


async def create_postgres_schema():
    """Create PostgreSQL schema that matches Chroma metadata structure"""
    conn = await asyncpg.connect(**POSTGRES_CONFIG)

    try:
        # Drop existing tables if they exist (for clean migration)
        await conn.execute("DROP TABLE IF EXISTS document_embeddings CASCADE")
        await conn.execute("DROP TABLE IF EXISTS files CASCADE")

        # Create files table (matching Chroma metadata)
        await conn.execute(
            """
            CREATE TABLE files (
                id SERIAL PRIMARY KEY,
                file_name VARCHAR(255) NOT NULL,
                file_md5_checksum VARCHAR(64) UNIQUE NOT NULL,
                published_date TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX idx_files_checksum ON files(file_md5_checksum);
            CREATE INDEX idx_files_name ON files(file_name);
        """
        )

        # Create document_embeddings table with Chroma-like schema
        await conn.execute(
            """
            CREATE TABLE document_embeddings (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(255) UNIQUE NOT NULL,
                file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                document_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536),
                file_name VARCHAR(255) NOT NULL,
                file_md5_checksum VARCHAR(64) NOT NULL,
                published_date TIMESTAMP WITH TIME ZONE,
                page_number INTEGER,
                chunk_type VARCHAR(50) NOT NULL,
                token_count INTEGER,
                chroma_document TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX idx_document_embeddings_chunk_id ON document_embeddings(chunk_id);
            CREATE INDEX idx_document_embeddings_file_id ON document_embeddings(file_id);
            CREATE INDEX idx_document_embeddings_document_id ON document_embeddings(document_id);
            CREATE INDEX idx_document_embeddings_chunk_type ON document_embeddings(chunk_type);
            CREATE INDEX idx_document_embeddings_page_number ON document_embeddings(page_number);
            CREATE INDEX idx_document_embeddings_token_count ON document_embeddings(token_count);

            -- Create vector index for similarity search using HNSW
            CREATE INDEX idx_document_embeddings_embedding ON document_embeddings
            USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        """
        )

        # Create trigger for updated_at
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';

            CREATE TRIGGER update_document_embeddings_updated_at
                BEFORE UPDATE ON document_embeddings
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        )

        print("✅ PostgreSQL schema created successfully")

    except Exception as e:
        print(f"\nError creating PostgreSQL schema: {e}")
    finally:
        await conn.close()


def get_chroma_data() -> List[Dict[str, Any]]:
    """Extract data from Chroma SQLite database"""
    conn = sqlite3.connect(CHROMA_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all embeddings with their metadata
    query = """
        SELECT
            e.embedding_id,
            e.segment_id,
            GROUP_CONCAT(
                CASE
                    WHEN em.key = 'chunk_type' THEN em.string_value
                    WHEN em.key = 'chunk_id' THEN em.string_value
                    WHEN em.key = 'file_md5_checksum' THEN em.string_value
                    WHEN em.key = 'file_name' THEN em.string_value
                    WHEN em.key = 'published_date' THEN em.string_value
                    WHEN em.key = 'page_number' THEN CAST(em.int_value AS TEXT)
                    WHEN em.key = 'token_count' THEN CAST(em.int_value AS TEXT)
                    WHEN em.key = 'chroma:document' THEN em.string_value
                    ELSE NULL
                END, '|'
            ) as metadata_str,
            s.content
        FROM embeddings e
        JOIN embedding_metadata em ON e.id = em.id
        JOIN segments s ON e.segment_id = s.id
        GROUP BY e.embedding_id, e.segment_id, s.content
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    documents = []
    for row in rows:
        # Parse metadata
        metadata_parts = row["metadata_str"].split("|")
        metadata = {}

        # Extract individual metadata fields
        for part in metadata_parts:
            if part and ":" in part:
                key, value = part.split(":", 1)
                metadata[key] = value

        # Build document dict
        doc = {
            "chunk_id": row["embedding_id"],
            "segment_id": row["segment_id"],
            "content": row["content"],
            "chunk_type": metadata.get("chunk_type", "unknown"),
            "file_name": metadata.get("file_name", ""),
            "file_md5_checksum": metadata.get("file_md5_checksum", ""),
            "published_date": metadata.get("published_date"),
            "page_number": (
                int(metadata.get("page_number", 0))
                if metadata.get("page_number")
                else None
            ),
            "token_count": (
                int(metadata.get("token_count", 0))
                if metadata.get("token_count")
                else None
            ),
            "chroma_document": metadata.get("chroma:document", ""),
        }

        documents.append(doc)

    conn.close()
    print(f"✅ Extracted {len(documents)} documents from Chroma")
    return documents


def get_embeddings_from_chroma() -> Dict[str, List[float]]:
    """Extract embeddings from Chroma SQLite database"""
    conn = sqlite3.connect(CHROMA_DB_PATH)
    cursor = conn.cursor()

    # Get embeddings
    cursor.execute("SELECT embedding_id, embedding FROM embeddings")
    rows = cursor.fetchall()

    embeddings = {}
    for embedding_id, embedding_blob in rows:
        # Convert binary blob to numpy array
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        embeddings[embedding_id] = embedding.tolist()

    conn.close()
    print(f"✅ Extracted {len(embeddings)} embeddings from Chroma")
    return embeddings


async def migrate_to_postgres():
    """Migrate Chroma data to PostgreSQL"""
    # Get data from Chroma
    documents = get_chroma_data()
    embeddings = get_embeddings_from_chroma()

    # Connect to PostgreSQL
    conn = await asyncpg.connect(**POSTGRES_CONFIG)

    try:
        # Start transaction
        async with conn.transaction():
            # First, insert unique files
            unique_files = {}
            for doc in documents:
                checksum = doc["file_md5_checksum"]
                if checksum and checksum not in unique_files:
                    unique_files[checksum] = {
                        "file_name": doc["file_name"],
                        "published_date": doc["published_date"],
                    }

            # Insert files and get their IDs
            file_ids = {}
            for checksum, file_info in unique_files.items():
                if checksum:  # Only insert if checksum exists
                    file_id = await conn.fetchval(
                        """
                        INSERT INTO files (file_name, file_md5_checksum, published_date)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (file_md5_checksum) DO NOTHING
                        RETURNING id
                    """,
                        file_info["file_name"],
                        checksum,
                        file_info["published_date"],
                    )

                    if file_id:
                        file_ids[checksum] = file_id
                    else:
                        # File already exists, get its ID
                        file_ids[checksum] = await conn.fetchval(
                            "SELECT id FROM files WHERE file_md5_checksum = $1",
                            checksum,
                        )

            print(f"✅ Inserted {len(file_ids)} unique files")

            # Now insert document embeddings
            inserted_count = 0
            for doc in documents:
                # Get embedding
                embedding = embeddings.get(doc["chunk_id"], [])

                if embedding and doc["file_md5_checksum"] in file_ids:
                    # Convert embedding to PostgreSQL vector format
                    embedding_str = f"[{','.join(map(str, embedding))}]"

                    # Parse published_date
                    published_date = None
                    if doc["published_date"]:
                        try:
                            published_date = datetime.fromisoformat(
                                doc["published_date"].replace("Z", "+00:00")
                            )
                        except Exception:
                            pass

                    await conn.execute(
                        """
                        INSERT INTO document_embeddings (
                            chunk_id, file_id, document_id, content, embedding,
                            file_name, file_md5_checksum, published_date,
                            page_number, chunk_type, token_count, chroma_document
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (chunk_id) DO NOTHING
                    """,
                        doc["chunk_id"],
                        file_ids[doc["file_md5_checksum"]],
                        doc["segment_id"],
                        doc["content"],
                        embedding_str,
                        doc["file_name"],
                        doc["file_md5_checksum"],
                        published_date,
                        doc["page_number"],
                        doc["chunk_type"],
                        doc["token_count"],
                        doc["chroma_document"],
                    )
                    inserted_count += 1

            print(f"✅ Inserted {inserted_count} document embeddings")

    except Exception as e:
        print(f"\nError migrating data to PostgreSQL: {e}")
    finally:
        await conn.close()

    print("\n🎉 Migration completed successfully!")


async def verify_migration():
    """Verify the migration by comparing counts"""
    try:
        chroma_conn = sqlite3.connect(CHROMA_DB_PATH)
        chroma_count = chroma_conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        chroma_conn.close()

        pg_conn = await asyncpg.connect(**POSTGRES_CONFIG)
        pg_count = await pg_conn.fetchval("SELECT COUNT(*) FROM document_embeddings")
        await pg_conn.close()

        print(f"\n📊 Migration Summary:")
        print(f"Chroma embeddings: {chroma_count}")
        print(f"PostgreSQL embeddings: {pg_count}")

        if chroma_count == pg_count:
            print("✅ Migration verification successful!")
        else:
            print("❌ Migration verification failed - counts don't match!")
    except Exception as e:
        print(f"\nMigration failed with error: {e}")


async def main():
    """Main migration function"""
    print("🚀 Starting Chroma to PostgreSQL migration...\n")

    # Step 1: Create PostgreSQL schema
    await create_postgres_schema()

    # Step 2: Migrate data
    await migrate_to_postgres()

    # Step 3: Verify migration
    await verify_migration()


if __name__ == "__main__":
    asyncio.run(main())
