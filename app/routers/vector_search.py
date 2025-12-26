import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


class VectorSearchRequest(BaseModel):
    query: str
    query_embedding: Optional[List[float]] = None
    limit: int = 10
    threshold: float = 0.7
    filter_metadata: Optional[Dict[str, Any]] = None


class VectorSearchResult(BaseModel):
    document_id: str
    content: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None


class DocumentEmbeddingRequest(BaseModel):
    document_id: str
    content: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None


async def get_db_pool(request: Request):
    """Dependency to get database connection pool from app state"""
    return request.app.state.db_pool


@router.post("/vector-search", response_model=List[VectorSearchResult])
async def search_vectors(
    request_data: VectorSearchRequest, db_pool=Depends(get_db_pool)
):
    """
    Search for similar documents using vector similarity
    """
    if not request_data.query_embedding:
        # If no embedding provided, you would typically generate it here
        # For now, return an error
        raise HTTPException(
            status_code=400,
            detail="Query embedding is required. Use /embed endpoint to generate embeddings.",
        )

    try:
        async with db_pool.acquire() as conn:
            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, request_data.query_embedding))}]"

            # Build the query
            query = """
                SELECT
                    document_id,
                    content,
                    1 - (embedding <=> $1) as similarity,
                    metadata
                FROM document_embeddings
                WHERE 1 - (embedding <=> $1) > $2
            """
            params = [embedding_str, request_data.threshold]

            # Add metadata filtering if provided
            if request_data.filter_metadata:
                for key, value in request_data.filter_metadata.items():
                    query += f" AND metadata->>'{key}' = ${len(params) + 1}"
                    params.append(str(value))

            query += " ORDER BY similarity DESC LIMIT $" + str(len(params) + 1)
            params.append(request_data.limit)

            rows = await conn.fetch(query, *params)

            results = [
                VectorSearchResult(
                    document_id=row["document_id"],
                    content=row["content"],
                    similarity=float(row["similarity"]),
                    metadata=row["metadata"],
                )
                for row in rows
            ]

            return results

    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/embeddings", response_model=Dict[str, str])
async def store_embedding(
    request_data: DocumentEmbeddingRequest, db_pool=Depends(get_db_pool)
):
    """
    Store document embedding in PostgreSQL
    """
    try:
        async with db_pool.acquire() as conn:
            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, request_data.embedding))}]"

            # Insert or update embedding
            await conn.execute(
                """
                INSERT INTO document_embeddings (document_id, content, embedding, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                request_data.document_id,
                request_data.content,
                embedding_str,
                request_data.metadata,
            )

            return {
                "status": "success",
                "message": f"Embedding stored for {request_data.document_id}",
            }

    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/embeddings/{document_id}", response_model=Optional[DocumentEmbeddingRequest]
)
async def get_embedding(document_id: str, db_pool=Depends(get_db_pool)):
    """
    Retrieve embedding for a specific document
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_id, content, embedding, metadata
                FROM document_embeddings
                WHERE document_id = $1
            """,
                document_id,
            )

            if not row:
                return None

            # Convert PostgreSQL vector to list
            embedding_str = row["embedding"].strip("[]")
            embedding = [float(x) for x in embedding_str.split(",")]

            return DocumentEmbeddingRequest(
                document_id=row["document_id"],
                content=row["content"],
                embedding=embedding,
                metadata=row["metadata"],
            )

    except Exception as e:
        logger.error(f"Error retrieving embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/embeddings/{document_id}")
async def delete_embedding(document_id: str, db_pool=Depends(get_db_pool)):
    """
    Delete embedding for a specific document
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM document_embeddings
                WHERE document_id = $1
            """,
                document_id,
            )

            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Document not found")

            return {
                "status": "success",
                "message": f"Embedding deleted for {document_id}",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/embeddings")
async def list_embeddings(
    limit: int = 100, offset: int = 0, db_pool=Depends(get_db_pool)
):
    """
    List all stored embeddings with pagination
    """
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id, content, metadata, created_at, updated_at
                FROM document_embeddings
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """,
                limit,
                offset,
            )

            total = await conn.fetchval(
                """
                SELECT COUNT(*) FROM document_embeddings
            """
            )

            return {
                "embeddings": [
                    {
                        "document_id": row["document_id"],
                        "content": (
                            row["content"][:200] + "..."
                            if len(row["content"]) > 200
                            else row["content"]
                        ),
                        "metadata": row["metadata"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ],
                "total": total,
                "limit": limit,
                "offset": offset,
            }

    except Exception as e:
        logger.error(f"Error listing embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
