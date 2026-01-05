import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_community.embeddings import DeepInfraEmbeddings
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()


class VectorSearchRequest(BaseModel):
    query: str
    limit: int = 5
    threshold: float = 0.4
    regex_search: Optional[str] = None
    filter_metadata: Optional[Dict[str, Any]] = None

    search_mode: Literal["general", "most_recent"] = "general"


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
    published_date: Optional[str] = None


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


class ChunkIdsRequest(BaseModel):
    chunk_ids: List[str]


class ChunkByIdResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    file_name: str
    file_md5_checksum: str
    page_number: Optional[int] = None
    chunk_type: str
    token_count: Optional[int] = None
    published_date: Optional[str] = None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


def _parse_embedding_value(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        return [float(x) for x in s.split(",") if x.strip()]
    return []


def _mmr_select(
    query_emb: List[float],
    candidates: List[dict],
    k: int,
    lambd: float,
) -> List[dict]:
    # candidates: each dict must have:
    # - "similarity" (relevance score, already includes time-decay if enabled)
    # - "embedding" (list[float])
    if k <= 0 or not candidates:
        return []

    selected: List[dict] = []
    selected_embs: List[List[float]] = []

    remaining = candidates[:]

    while remaining and len(selected) < k:
        best = None
        best_score = -1e18

        for cand in remaining:
            rel = float(cand["similarity"])
            cand_emb = cand["embedding"]

            if not selected_embs:
                div = 0.0
            else:
                div = max(_cosine(cand_emb, e) for e in selected_embs)

            mmr_score = (lambd * rel) - ((1.0 - lambd) * div)
            if mmr_score > best_score:
                best_score = mmr_score
                best = cand

        selected.append(best)
        selected_embs.append(best["embedding"])
        remaining.remove(best)

    return selected


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
    # Generate embedding for the query
    try:
        embedding_list = await embeddings_model.aembed_query(request_data.query)
        query_embedding = embedding_list
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate embedding for query",
        ) from e

    if request_data.regex_search:
        try:
            re.compile(request_data.regex_search)
        except re.error as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid regex_search: {request_data.regex_search}. Underlying error: {type(e).__name__}: {e}",
            ) from e

    try:
        async with db_pool.acquire() as conn:
            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            if request_data.search_mode not in ("general", "most_recent"):
                raise HTTPException(
                    status_code=422,
                    detail="search_mode must be either 'general' or 'most_recent'",
                )

            use_time_decay = request_data.search_mode == "most_recent"
            use_mmr = request_data.search_mode in ("general", "most_recent")

            fetch_k = (
                max(int(request_data.limit) * 8, 40)
                if use_mmr
                else int(request_data.limit)
            )

            if use_time_decay:
                # Recency-aware ranking (then light de-dup via MMR with a high lambda).
                # The parameters below are intentionally not exposed to the caller.
                grace_days = 7
                cutoff_days = 180
                time_decay_lambda = 0.2

                # Build the query with a freshness factor derived from published_date.
                # - No penalty within grace window.
                # - Linear decay to 0 at cutoff.
                # - Exclude older than cutoff.
                query = """--sql
                    WITH scored AS (
                        SELECT
                            chunk_id,
                            document_id,
                            content,
                            embedding,
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
                                WHEN NOW() - published_date >= ($4 * INTERVAL '1 day') THEN (1.0 - $5)
                                ELSE 1.0 - (
                                    $5 * (
                                        (EXTRACT(EPOCH FROM (NOW() - published_date)) / 86400.0 - $3)
                                        / ($4 - $3)
                                    )
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
                        embedding,
                        file_name,
                        file_md5_checksum,
                        page_number,
                        chunk_type,
                        token_count,
                        published_date
                    FROM scored
                    WHERE (similarity * freshness_factor) > $2
                """
                params = [
                    embedding_str,
                    request_data.threshold,
                    grace_days,
                    cutoff_days,
                    time_decay_lambda,
                ]
            else:
                # Build the query (no time decay)
                query = """--sql
                    SELECT
                        chunk_id,
                        document_id,
                        content,
                        1 - (embedding <=> $1) as similarity,
                        embedding,
                        file_name,
                        file_md5_checksum,
                        page_number,
                        chunk_type,
                        token_count,
                        published_date
                    FROM document_embeddings
                    WHERE 1 - (embedding <=> $1) > $2
                """
                params = [embedding_str, request_data.threshold]

            if request_data.regex_search:
                query += " AND (content ~* ${} OR file_name ~* ${})".format(
                    len(params) + 1, len(params) + 2
                )
                params.append(request_data.regex_search)
                params.append(request_data.regex_search)

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
            params.append(fetch_k)

            rows = await conn.fetch(query, *params)

            # prepare candidate dicts (for optional mmr)
            candidates: List[dict] = []
            for row in rows:
                embedding_value = _parse_embedding_value(row.get("embedding"))
                candidates.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "content": row["content"],
                        "similarity": float(row["similarity"]),
                        "embedding": embedding_value,
                        "file_name": row["file_name"],
                        "file_md5_checksum": row["file_md5_checksum"],
                        "page_number": row["page_number"],
                        "chunk_type": row["chunk_type"],
                        "token_count": row["token_count"],
                        "published_date": (
                            row["published_date"].isoformat()
                            if row.get("published_date")
                            else None
                        ),
                    }
                )

            if use_mmr:
                mmr_lambda = 0.5 if request_data.search_mode == "general" else 0.9
                selected = _mmr_select(
                    query_emb=list(map(float, query_embedding)),
                    candidates=candidates,
                    k=request_data.limit,
                    lambd=float(mmr_lambda),
                )
            else:
                selected = candidates[: request_data.limit]

            return [
                VectorSearchResult(
                    chunk_id=c["chunk_id"],
                    document_id=c["document_id"],
                    content=c["content"],
                    similarity=float(c["similarity"]),
                    file_name=c["file_name"],
                    file_md5_checksum=c["file_md5_checksum"],
                    page_number=c["page_number"],
                    chunk_type=c["chunk_type"],
                    token_count=c["token_count"],
                    published_date=c.get("published_date"),
                )
                for c in selected
            ]

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


@router.post("/chunks/by-ids", response_model=List[ChunkByIdResult])
async def get_chunks_by_ids(
    request_data: ChunkIdsRequest, db_pool=Depends(get_db_pool)
):
    chunk_ids = request_data.chunk_ids
    if not chunk_ids:
        return []

    unique_chunk_ids = list(dict.fromkeys(chunk_ids))

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """--sql
                SELECT
                    t.chunk_id,
                    de.document_id,
                    de.content,
                    de.file_name,
                    de.file_md5_checksum,
                    de.page_number,
                    de.chunk_type,
                    de.token_count,
                    de.published_date
                FROM unnest($1::text[]) WITH ORDINALITY AS t(chunk_id, ord)
                JOIN document_embeddings de
                  ON de.chunk_id = t.chunk_id
                ORDER BY t.ord
                """,
                unique_chunk_ids,
            )

            results: List[ChunkByIdResult] = []
            for row in rows:
                results.append(
                    ChunkByIdResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        content=row["content"],
                        file_name=row["file_name"],
                        file_md5_checksum=row["file_md5_checksum"],
                        page_number=row["page_number"],
                        chunk_type=row["chunk_type"],
                        token_count=row["token_count"],
                        published_date=(
                            row["published_date"].isoformat()
                            if row["published_date"]
                            else None
                        ),
                    )
                )

            return results
    except Exception as e:
        logger.error(f"Error fetching chunks by ids: {e}")
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
