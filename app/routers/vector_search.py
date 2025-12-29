import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_community.embeddings import DeepInfraEmbeddings
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()


class VectorSearchRequest(BaseModel):
    query: str
    query_embedding: Optional[List[float]] = None
    limit: int = 10
    threshold: float = 0.7
    filter_metadata: Optional[Dict[str, Any]] = None
    use_time_decay: bool = False
    time_decay_grace_days: int = 7
    time_decay_cutoff_days: int = 180


class VectorSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    similarity: float
    file_name: str
    file_md5_checksum: str
    page_number: Optional[int] = None
    chunk_type: str
    token_count: Optional[int] = None


class DocumentEmbeddingRequest(BaseModel):
    chunk_id: str
    file_id: int
    document_id: str
    content: str
    embedding: List[float]
    file_name: str
    file_md5_checksum: str
    published_date: Optional[str] = None
    page_number: Optional[int] = None
    chunk_type: str = "unknown"
    token_count: Optional[int] = None
    chroma_document: Optional[str] = None


async def get_embeddings_model():
    """Get the embeddings model instance"""
    return DeepInfraEmbeddings(
        model_id="BAAI/bge-m3",
        query_instruction="",
        embed_instruction="",
        deepinfra_api_token=os.getenv("DEEP_INFRA_API_KEY"),
    )


async def get_db_pool(request: Request):
    """Dependency to get database connection pool from app state"""
    if not hasattr(request.app.state, "db_pool"):
        raise HTTPException(
            status_code=503,
            detail="Database connection pool not initialized. Please ensure the application has started properly.",
        )
    return request.app.state.db_pool


@router.post("/vector-search", response_model=List[VectorSearchResult])
async def search_vectors(
    request_data: VectorSearchRequest,
    db_pool=Depends(get_db_pool),
    embeddings_model=Depends(get_embeddings_model),
):
    """
    Search for similar documents using vector similarity
    """
    # Generate embedding if not provided
    if not request_data.query_embedding:
        try:
            embedding_list = await embeddings_model.aembed_query(request_data.query)
            request_data.query_embedding = embedding_list
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding for query",
            ) from e

    try:
        async with db_pool.acquire() as conn:
            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, request_data.query_embedding))}]"

            if request_data.use_time_decay:
                grace_days = max(0, int(request_data.time_decay_grace_days))
                cutoff_days = max(1, int(request_data.time_decay_cutoff_days))
                if grace_days >= cutoff_days:
                    raise HTTPException(
                        status_code=422,
                        detail="time_decay_grace_days must be less than time_decay_cutoff_days",
                    )

                # Build the query with a freshness factor derived from published_date.
                # - No penalty within grace window.
                # - Linear decay to 0 at cutoff.
                # - Exclude older than cutoff.
                query = """
                    WITH scored AS (
                        SELECT
                            chunk_id,
                            document_id,
                            content,
                            file_name,
                            file_md5_checksum,
                            page_number,
                            chunk_type,
                            token_count,
                            published_date,
                            (1 - (embedding <=> $1)) AS similarity,
                            CASE
                                WHEN published_date IS NULL THEN 1.0
                                WHEN NOW() - published_date <= ($3 * INTERVAL '1 day') THEN 1.0
                                WHEN NOW() - published_date >= ($4 * INTERVAL '1 day') THEN 0.0
                                ELSE 1.0 - (
                                    (EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400.0 - $3)
                                    / ($4 - $3)
                                )
                            END AS freshness_factor
                        FROM document_embeddings
                        WHERE (1 - (embedding <=> $1)) > $2
                          AND (published_date IS NULL OR published_date >= NOW() - ($4 * INTERVAL '1 day'))
                    )
                    SELECT
                        chunk_id,
                        document_id,
                        content,
                        (similarity * freshness_factor) AS similarity,
                        file_name,
                        file_md5_checksum,
                        page_number,
                        chunk_type,
                        token_count
                    FROM scored
                    WHERE (similarity * freshness_factor) > $2
                """
                params = [
                    embedding_str,
                    request_data.threshold,
                    grace_days,
                    cutoff_days,
                ]
            else:
                # Build the query (no time decay)
                query = """
                    SELECT
                        chunk_id,
                        document_id,
                        content,
                        1 - (embedding <=> $1) as similarity,
                        file_name,
                        file_md5_checksum,
                        page_number,
                        chunk_type,
                        token_count
                    FROM document_embeddings
                    WHERE 1 - (embedding <=> $1) > $2
                """
                params = [embedding_str, request_data.threshold]

            # Whitelist of allowed filter keys to prevent SQL injection
            allowed_filters = {
                "file_md5_checksum": "file_md5_checksum = ${}",
                "chunk_type": "chunk_type = ${}",
                "file_name": "file_name = ${}",
                "document_id": "document_id = ${}",
            }

            # Add file filtering if provided
            if request_data.filter_metadata:
                for key, value in request_data.filter_metadata.items():
                    if key in allowed_filters:
                        # Use safe parameterized query
                        query += f" AND {allowed_filters[key].format(len(params) + 1)}"
                        params.append(str(value))

            query += " ORDER BY similarity DESC LIMIT $" + str(len(params) + 1)
            params.append(request_data.limit)

            rows = await conn.fetch(query, *params)

            results = [
                VectorSearchResult(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    similarity=float(row["similarity"]),
                    file_name=row["file_name"],
                    file_md5_checksum=row["file_md5_checksum"],
                    page_number=row["page_number"],
                    chunk_type=row["chunk_type"],
                    token_count=row["token_count"],
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

            # Parse published_date if provided
            published_date = None
            if request_data.published_date:
                try:
                    published_date = datetime.fromisoformat(
                        request_data.published_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # Insert or update embedding
            await conn.execute(
                """
                INSERT INTO document_embeddings (
                    chunk_id, file_id, document_id, content, embedding,
                    file_name, file_md5_checksum, published_date,
                    page_number, chunk_type, token_count, chroma_document
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    file_name = EXCLUDED.file_name,
                    file_md5_checksum = EXCLUDED.file_md5_checksum,
                    published_date = EXCLUDED.published_date,
                    page_number = EXCLUDED.page_number,
                    chunk_type = EXCLUDED.chunk_type,
                    token_count = EXCLUDED.token_count,
                    chroma_document = EXCLUDED.chroma_document,
                    updated_at = NOW()
                """,
                request_data.chunk_id,
                request_data.file_id,
                request_data.document_id,
                request_data.content,
                embedding_str,
                request_data.file_name,
                request_data.file_md5_checksum,
                published_date,
                request_data.page_number,
                request_data.chunk_type,
                request_data.token_count,
                request_data.chroma_document,
            )

            return {
                "status": "success",
                "message": f"Embedding stored for {request_data.chunk_id}",
            }

    except Exception as e:
        logger.error(f"Error storing embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/embeddings/{chunk_id}", response_model=Optional[DocumentEmbeddingRequest])
async def get_embedding(chunk_id: str, db_pool=Depends(get_db_pool)):
    """
    Retrieve embedding for a specific chunk
    """
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    chunk_id, file_id, document_id, content, embedding,
                    file_name, file_md5_checksum, published_date,
                    page_number, chunk_type, token_count, chroma_document
                FROM document_embeddings
                WHERE chunk_id = $1
            """,
                chunk_id,
            )

            if not row:
                return None

            # Convert PostgreSQL vector to list
            embedding_str = row["embedding"].strip("[]")
            embedding = [float(x) for x in embedding_str.split(",")]

            return DocumentEmbeddingRequest(
                chunk_id=row["chunk_id"],
                file_id=row["file_id"],
                document_id=row["document_id"],
                content=row["content"],
                embedding=embedding,
                file_name=row["file_name"],
                file_md5_checksum=row["file_md5_checksum"],
                published_date=(
                    row["published_date"].isoformat() if row["published_date"] else None
                ),
                page_number=row["page_number"],
                chunk_type=row["chunk_type"],
                token_count=row["token_count"],
                chroma_document=row["chroma_document"],
            )

    except Exception as e:
        logger.error(f"Error retrieving embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/embeddings/{chunk_id}")
async def delete_embedding(chunk_id: str, db_pool=Depends(get_db_pool)):
    """
    Delete embedding for a specific chunk
    """
    try:
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM document_embeddings
                WHERE chunk_id = $1
            """,
                chunk_id,
            )

            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Chunk not found")

            return {
                "status": "success",
                "message": f"Embedding deleted for {chunk_id}",
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
                SELECT
                    chunk_id, document_id, file_name, file_md5_checksum,
                    chunk_type, page_number, token_count, created_at, updated_at
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
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "file_name": row["file_name"],
                        "file_md5_checksum": row["file_md5_checksum"],
                        "chunk_type": row["chunk_type"],
                        "page_number": row["page_number"],
                        "token_count": row["token_count"],
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
