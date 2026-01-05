"""
Test script for pgvector functionality
"""

import asyncio
import os

import asyncpg
import numpy as np
from dotenv import load_dotenv

load_dotenv()


async def test_pgvector():
    """Test pgvector connection and basic operations"""

    # Connect to PostgreSQL
    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "vector_search"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )

    try:
        print("Testing pgvector functionality...")

        # Test 1: Check if pgvector extension is enabled
        result = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        print(f"pgvector extension version: {result}")

        # Test 2: Create a test embedding
        test_embedding = np.random.rand(1536).astype(np.float32)
        embedding_str = "[" + ",".join(map(str, test_embedding)) + "]"

        # Test 3: Insert a test document (match current schema)
        test_checksum = "test_checksum_1"
        test_file_name = "test.pdf"
        test_chunk_id = "test_chunk_1"
        test_document_id = "test_doc_1"
        test_content = "This is a test document for pgvector"

        file_id = await conn.fetchval(
            """
            INSERT INTO files (file_name, file_md5_checksum, published_date)
            VALUES ($1, $2, NOW())
            ON CONFLICT (file_md5_checksum)
            DO UPDATE SET
                file_name = EXCLUDED.file_name,
                processed_at = NOW()
            RETURNING id
            """,
            test_file_name,
            test_checksum,
        )

        await conn.execute(
            """
            INSERT INTO document_embeddings (
                chunk_id, file_id, document_id, content, embedding,
                file_name, file_md5_checksum, published_date,
                page_number, chunk_type, token_count, chroma_document
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, NOW(),
                $8, $9, $10, $11
            )
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            test_chunk_id,
            int(file_id),
            test_document_id,
            test_content,
            embedding_str,
            test_file_name,
            test_checksum,
            1,
            "page",
            None,
            None,
        )

        print("Inserted test document")

        # Test 4: Perform similarity search
        results = await conn.fetch(
            """
            SELECT
                document_id,
                content,
                1 - (embedding <=> $1) as similarity
            FROM document_embeddings
            ORDER BY embedding <=> $1
            LIMIT 5
            """,
            embedding_str,
        )

        print("\nSimilarity search results:")
        for row in results:
            print(f"- {row['document_id']}: similarity={row['similarity']:.4f}")

        # Test 5: Clean up
        await conn.execute(
            "DELETE FROM document_embeddings WHERE chunk_id = $1", test_chunk_id
        )
        await conn.execute(
            "DELETE FROM files WHERE file_md5_checksum = $1", test_checksum
        )
        print("\nCleaned up test document")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(test_pgvector())
