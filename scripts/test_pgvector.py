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

        # Test 3: Insert a test document
        await conn.execute(
            """
            INSERT INTO document_embeddings (document_id, content, embedding, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (document_id) DO NOTHING
        """,
            "test_doc_1",
            "This is a test document for pgvector",
            embedding_str,
            {"source": "test"},
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
            "DELETE FROM document_embeddings WHERE document_id = $1", "test_doc_1"
        )
        print("\nCleaned up test document")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(test_pgvector())
