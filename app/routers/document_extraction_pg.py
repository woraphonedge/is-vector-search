import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from landingai_ade import LandingAIADE
from langchain_community.embeddings import DeepInfraEmbeddings
from pydantic import BaseModel, ConfigDict, Field

from app.models import DocumentMetadata
from app.utils.rag_utils import prepare_document

logger = logging.getLogger("is_vector_search")

router = APIRouter()

ade_client = LandingAIADE(apikey=os.getenv("VISION_AGENT_API_KEY"))


def _get_pg_storage_base_dir() -> Path:
    # Local disk storage for Postgres engine artifacts (uploaded file + parsed json)
    base = os.getenv("PG_FILE_STORAGE_DIR")
    if not base:
        base = "./pg_file_storage"
    return Path(base)


def _pg_files_dir() -> Path:
    return _get_pg_storage_base_dir() / "files"


def _pg_parsed_json_dir() -> Path:
    return _get_pg_storage_base_dir() / "parsed_json"


def _ensure_pg_storage_dirs() -> None:
    _pg_files_dir().mkdir(parents=True, exist_ok=True)
    _pg_parsed_json_dir().mkdir(parents=True, exist_ok=True)


def _file_path_for_upload(*, file_md5_checksum: str, original_filename: str) -> Path:
    ext = Path(original_filename).suffix
    if not ext:
        ext = ".bin"
    return _pg_files_dir() / f"{file_md5_checksum}{ext}"


def _find_uploaded_file_path(file_md5_checksum: str) -> Path | None:
    # Prefer the new layout: <base>/files/{checksum}.*
    for candidate in _pg_files_dir().glob(f"{file_md5_checksum}*"):
        if candidate.is_file():
            return candidate

    # Backward-compatible layout: <base>/{checksum}.*
    for candidate in _get_pg_storage_base_dir().glob(f"{file_md5_checksum}*"):
        if candidate.is_file():
            return candidate
    return None


def _find_parsed_json_path(file_md5_checksum: str) -> Path | None:
    # Prefer the new layout: <base>/parsed_json/{checksum}.json
    candidate = _pg_parsed_json_dir() / f"{file_md5_checksum}.json"
    if candidate.exists() and candidate.is_file():
        return candidate

    # Backward-compatible layout: <base>/{checksum}.json
    candidate = _get_pg_storage_base_dir() / f"{file_md5_checksum}.json"
    if candidate.exists() and candidate.is_file():
        return candidate

    return None


class CreateExtractionJobResponse(BaseModel):
    job_id: str
    file_reference: str


class ExtractionJobStatusResponse(BaseModel):
    job_id: str
    file_reference: str
    file_name: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    chunks_processed: Optional[int] = None
    pages_processed: Optional[int] = None


class DocumentMetadataWithJob(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        validate_by_alias=True,
        extra="forbid",
        from_attributes=True,
    )

    id: int | None = None
    file_md5_checksum: str = Field(alias="fileMd5Checksum")
    file_name: str = Field(alias="fileName")
    type: str
    category: str
    tags: List[str]
    parsed_at: datetime = Field(alias="parsedAt")
    published_date: datetime | None = Field(default=None, alias="publishedDate")
    user_id: str | None = Field(default=None, alias="userId")
    product_type: str | None = Field(default=None, alias="productType")
    service_type: str | None = Field(default=None, alias="serviceType")
    json_file_path: str = Field(alias="jsonFilePath")
    file_status: str | None = Field(default=None, alias="fileStatus")
    json_status: str | None = Field(default=None, alias="jsonStatus")
    created_at: datetime | None = Field(default=None, alias="createdAt")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")

    pg_job_id: str | None = Field(default=None, alias="pgJobId")
    pg_job_status: str | None = Field(default=None, alias="pgJobStatus")
    pg_job_error_message: str | None = Field(default=None, alias="pgJobErrorMessage")


async def get_db_pool(request: Request):
    if not hasattr(request.app.state, "db_pool"):
        raise HTTPException(
            status_code=503,
            detail="Database connection pool not initialized. Please ensure the application has started properly.",
        )
    return request.app.state.db_pool


@router.patch(
    "/documents/{file_reference}",
    response_model=DocumentMetadata,
    response_model_by_alias=True,
    response_model_exclude_none=False,
)
async def update_document_pg(
    file_reference: str,
    updates: Dict[str, Any],
    db_pool=Depends(get_db_pool),
):
    field_mapping = {
        "name": "file_name",
        "type": "type",
        "category": "category",
        "tags": "tags",
        "publishedDate": "published_date",
        "userId": "user_id",
        "productType": "product_type",
        "serviceType": "service_type",
        "fileStatus": "file_status",
        "jsonStatus": "json_status",
        "jsonFilePath": "json_file_path",
    }

    update_data: Dict[str, Any] = {}
    for api_field, db_field in field_mapping.items():
        if api_field not in updates:
            continue
        value = updates[api_field]

        if api_field == "tags":
            if isinstance(value, list):
                tag_list = [str(t).strip() for t in value if str(t).strip()]
            else:
                tag_list = [t.strip() for t in str(value).split(",") if t.strip()]
            update_data[db_field] = tag_list
        elif api_field == "publishedDate":
            if value:
                try:
                    if isinstance(value, str):
                        update_data[db_field] = datetime.fromisoformat(
                            value.replace("Z", "+00:00")
                        )
                    else:
                        update_data[db_field] = value
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=422, detail="Invalid publishedDate format"
                    ) from None
            else:
                update_data[db_field] = None
        else:
            update_data[db_field] = value

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields provided to update.")

    try:
        async with db_pool.acquire() as conn:
            existing = await conn.fetchrow(
                """
                SELECT * FROM file_metadata WHERE file_md5_checksum = $1
                """,
                file_reference,
            )
            if not existing:
                raise HTTPException(status_code=404, detail="Document not found")

            sets = []
            params: List[Any] = [file_reference]
            i = 2
            for col, val in update_data.items():
                if col == "tags":
                    sets.append(f"{col} = ${i}::jsonb")
                    params.append(json.dumps(val))
                else:
                    sets.append(f"{col} = ${i}")
                    params.append(val)
                i += 1
            sets.append("updated_at = NOW()")

            query = (
                "UPDATE file_metadata SET "
                + ", ".join(sets)
                + " WHERE file_md5_checksum = $1 RETURNING *"
            )

            row = await conn.fetchrow(query, *params)

        tags_val = row["tags"]
        tags_list: List[str] = []
        if isinstance(tags_val, list):
            tags_list = [str(x) for x in tags_val]
        elif tags_val is None:
            tags_list = []
        else:
            try:
                tags_list = list(tags_val)
            except Exception:
                tags_list = []

        parsed_at = row["parsed_at"] or row["created_at"]

        return DocumentMetadata(
            id=row["id"],
            file_md5_checksum=row["file_md5_checksum"],
            file_name=row["file_name"],
            type=row["type"] or "pdf",
            category=row["category"] or "resources",
            tags=tags_list,
            parsed_at=parsed_at,
            published_date=row["published_date"],
            user_id=row["user_id"],
            product_type=row["product_type"],
            service_type=row["service_type"],
            json_file_path=row["json_file_path"] or "",
            file_status=row["file_status"],
            json_status=row["json_status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update Postgres document: {e}")
        raise HTTPException(status_code=500, detail="Failed to update document") from e


@router.delete("/documents/{file_reference}")
async def delete_document_pg(file_reference: str, db_pool=Depends(get_db_pool)):
    try:
        stored_file = _find_uploaded_file_path(file_reference)
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                json_row = await conn.fetchrow(
                    "SELECT json_file_path FROM file_metadata WHERE file_md5_checksum = $1",
                    file_reference,
                )
                await conn.execute(
                    "DELETE FROM document_embeddings WHERE file_md5_checksum = $1",
                    file_reference,
                )
                await conn.execute(
                    "DELETE FROM extraction_jobs WHERE file_md5_checksum = $1",
                    file_reference,
                )
                await conn.execute(
                    "DELETE FROM file_metadata WHERE file_md5_checksum = $1",
                    file_reference,
                )
                await conn.execute(
                    "DELETE FROM files WHERE file_md5_checksum = $1",
                    file_reference,
                )

        if stored_file and stored_file.exists():
            try:
                stored_file.unlink()
            except Exception:
                logger.exception("Failed to delete stored file for %s", file_reference)

        json_path: Path | None = None
        if json_row and json_row.get("json_file_path"):
            json_path = Path(str(json_row["json_file_path"]))
            if not json_path.exists():
                json_path = None
        if not json_path:
            json_path = _find_parsed_json_path(file_reference)

        if json_path and json_path.exists():
            try:
                json_path.unlink()
            except Exception:
                logger.exception(
                    "Failed to delete stored parsed json for %s", file_reference
                )

        return {"status": "success", "deleted": file_reference}
    except Exception as e:
        logger.error(f"Failed to delete Postgres document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document") from e


async def get_embeddings_model():
    return DeepInfraEmbeddings(
        model_id="BAAI/bge-m3",
        query_instruction="",
        embed_instruction="",
        deepinfra_api_token=os.getenv("DEEP_INFRA_API_KEY"),
    )


@router.get(
    "/list_parsed_documents",
    response_model=List[DocumentMetadataWithJob],
    response_model_by_alias=True,
    response_model_exclude_none=False,
)
async def list_parsed_documents_pg(db_pool=Depends(get_db_pool)):
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    file_md5_checksum,
                    file_name,
                    type,
                    category,
                    tags,
                    parsed_at,
                    published_date,
                    user_id,
                    product_type,
                    service_type,
                    json_file_path,
                    file_status,
                    json_status,
                    created_at,
                    updated_at,
                    lj.job_id,
                    lj.job_status,
                    lj.error_message
                FROM file_metadata fm
                LEFT JOIN LATERAL (
                    SELECT
                        id::text AS job_id,
                        status AS job_status,
                        error_message
                    FROM extraction_jobs
                    WHERE file_md5_checksum = fm.file_md5_checksum
                    ORDER BY created_at DESC
                    LIMIT 1
                ) lj ON TRUE
                ORDER BY COALESCE(parsed_at, created_at) DESC
                """
            )

        results: List[DocumentMetadataWithJob] = []
        for row in rows:
            tags_val = row["tags"]
            tags_list: List[str] = []
            if isinstance(tags_val, list):
                tags_list = [str(x) for x in tags_val]
            elif tags_val is None:
                tags_list = []
            else:
                # asyncpg JSONB can come back as dict/str depending on config
                try:
                    tags_list = list(tags_val)
                except Exception:
                    tags_list = []

            parsed_at = row["parsed_at"] or row["created_at"]
            published_date = row["published_date"]

            pg_job_status = row["job_status"]
            if pg_job_status in {"completed", "processed"}:
                pg_job_status = "ready"

            results.append(
                DocumentMetadataWithJob(
                    id=row["id"],
                    file_md5_checksum=row["file_md5_checksum"],
                    file_name=row["file_name"],
                    type=row["type"] or "pdf",
                    category=row["category"] or "resources",
                    tags=tags_list,
                    parsed_at=parsed_at,
                    published_date=published_date,
                    user_id=row["user_id"],
                    product_type=row["product_type"],
                    service_type=row["service_type"],
                    json_file_path=row["json_file_path"] or "",
                    file_status=row["file_status"],
                    json_status=row["json_status"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    pg_job_id=row["job_id"],
                    pg_job_status=pg_job_status,
                    pg_job_error_message=row["error_message"],
                )
            )

        return results

    except Exception as e:
        logger.error(f"Failed to list Postgres documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents") from e


async def _call_landing_ai_parser(file_content: bytes) -> Dict[str, Any]:
    api_key = os.getenv("VISION_AGENT_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Please set the VISION_AGENT_API_KEY environment variable."
        )

    response = ade_client.parse(
        document=file_content,
        model="dpt-2",
    )
    return response.to_dict()


def _embedding_to_pgvector_str(embedding: List[float]) -> str:
    return f"[{','.join(map(str, embedding))}]"


async def _job_set_status(
    db_pool,
    job_id: uuid.UUID,
    status: str,
    error_message: Optional[str] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    chunks_processed: Optional[int] = None,
    pages_processed: Optional[int] = None,
):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE extraction_jobs
            SET status = $2,
                error_message = COALESCE($3, error_message),
                started_at = COALESCE($4, started_at),
                finished_at = COALESCE($5, finished_at),
                chunks_processed = COALESCE($6, chunks_processed),
                pages_processed = COALESCE($7, pages_processed)
            WHERE id = $1
            """,
            job_id,
            status,
            error_message,
            started_at,
            finished_at,
            chunks_processed,
            pages_processed,
        )


async def _ensure_files_row(
    db_pool,
    file_name: str,
    file_md5_checksum: str,
    published_date: Optional[datetime],
) -> int:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO files (file_name, file_md5_checksum, published_date)
            VALUES ($1, $2, $3)
            ON CONFLICT (file_md5_checksum)
            DO UPDATE SET
                file_name = EXCLUDED.file_name,
                published_date = EXCLUDED.published_date,
                processed_at = NOW()
            RETURNING id
            """,
            file_name,
            file_md5_checksum,
            published_date,
        )
        return int(row["id"])


async def _upsert_file_metadata(
    db_pool,
    *,
    file_md5_checksum: str,
    file_name: str,
    type: str,
    category: str,
    tags: List[str],
    parsed_at: Optional[datetime],
    published_date: Optional[datetime],
    user_id: Optional[str],
    product_type: Optional[str],
    service_type: Optional[str],
    json_file_path: Optional[str],
    file_status: Optional[str],
    json_status: Optional[str],
):
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO file_metadata (
                file_md5_checksum, file_name, type, category, tags,
                parsed_at, published_date, user_id, product_type, service_type,
                json_file_path, file_status, json_status
            ) VALUES (
                $1, $2, $3, $4, $5::jsonb,
                $6, $7, $8, $9, $10,
                $11, $12, $13
            )
            ON CONFLICT (file_md5_checksum)
            DO UPDATE SET
                file_name = EXCLUDED.file_name,
                type = EXCLUDED.type,
                category = EXCLUDED.category,
                tags = EXCLUDED.tags,
                parsed_at = EXCLUDED.parsed_at,
                published_date = EXCLUDED.published_date,
                user_id = EXCLUDED.user_id,
                product_type = EXCLUDED.product_type,
                service_type = EXCLUDED.service_type,
                json_file_path = EXCLUDED.json_file_path,
                file_status = EXCLUDED.file_status,
                json_status = EXCLUDED.json_status
            """,
            file_md5_checksum,
            file_name,
            type,
            category,
            json.dumps(tags),
            parsed_at,
            published_date,
            user_id,
            product_type,
            service_type,
            json_file_path,
            file_status,
            json_status,
        )


async def _run_extraction_job(
    *,
    db_pool,
    embeddings_model: DeepInfraEmbeddings,
    job_id: uuid.UUID,
    file_name: str,
    file_md5_checksum: str,
    file_path: str,
    type: str,
    category: str,
    tags: List[str],
    user_id: Optional[str],
    product_type: Optional[str],
    service_type: Optional[str],
    published_date: Optional[datetime],
):
    started_at = datetime.now()
    await _job_set_status(
        db_pool,
        job_id,
        status="running",
        started_at=started_at,
    )

    try:
        _ensure_pg_storage_dirs()
        stored_file_path = Path(file_path)
        if not stored_file_path.exists():
            raise FileNotFoundError(
                f"Stored file not found for {file_md5_checksum}: {stored_file_path}"
            )

        file_content = stored_file_path.read_bytes()
        parsed_json = await _call_landing_ai_parser(file_content)

        parsed_json_path = _pg_parsed_json_dir() / f"{file_md5_checksum}.json"
        parsed_json_path.write_text(json.dumps(parsed_json), encoding="utf-8")

        published_date_str = (
            published_date.isoformat() if published_date else datetime.now().isoformat()
        )
        documents = prepare_document(
            file_name=file_name,
            md5_checksum=file_md5_checksum,
            parsed_json=parsed_json,
            publishedDate=published_date_str,
            chunk_type="page",
        )

        file_id = await _ensure_files_row(
            db_pool,
            file_name=file_name,
            file_md5_checksum=file_md5_checksum,
            published_date=published_date,
        )

        await _upsert_file_metadata(
            db_pool,
            file_md5_checksum=file_md5_checksum,
            file_name=file_name,
            type=type,
            category=category,
            tags=tags,
            parsed_at=datetime.now(),
            published_date=published_date,
            user_id=user_id,
            product_type=product_type,
            service_type=service_type,
            json_file_path=str(parsed_json_path),
            file_status="ready",
            json_status="ready",
        )

        if not documents:
            await _job_set_status(
                db_pool,
                job_id,
                status="ready",
                finished_at=datetime.now(),
                chunks_processed=0,
                pages_processed=0,
            )
            return

        contents = [doc.page_content for doc in documents]
        embeddings = await embeddings_model.aembed_documents(contents)

        pages_processed = len({doc.metadata.get("page_number") for doc in documents})

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                for doc, embedding in zip(documents, embeddings, strict=True):
                    embedding_str = _embedding_to_pgvector_str(embedding)
                    await conn.execute(
                        """
                        INSERT INTO document_embeddings (
                            chunk_id, file_id, document_id, content, embedding,
                            file_name, file_md5_checksum, published_date,
                            page_number, chunk_type, token_count, chroma_document
                        ) VALUES (
                            $1, $2, $3, $4, $5,
                            $6, $7, $8,
                            $9, $10, $11, $12
                        )
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
                        doc.metadata.get("chunk_id"),
                        file_id,
                        doc.metadata.get("chunk_id"),
                        doc.page_content,
                        embedding_str,
                        doc.metadata.get("file_name"),
                        doc.metadata.get("file_md5_checksum"),
                        published_date,
                        doc.metadata.get("page_number"),
                        doc.metadata.get("chunk_type"),
                        doc.metadata.get("token_count"),
                        None,
                    )

        await _job_set_status(
            db_pool,
            job_id,
            status="ready",
            finished_at=datetime.now(),
            chunks_processed=len(documents),
            pages_processed=pages_processed,
        )

    except Exception as e:
        logger.error(f"Extraction job failed: {e}")
        try:
            await _upsert_file_metadata(
                db_pool,
                file_md5_checksum=file_md5_checksum,
                file_name=file_name,
                type=type,
                category=category,
                tags=tags,
                parsed_at=None,
                published_date=published_date,
                user_id=user_id,
                product_type=product_type,
                service_type=service_type,
                json_file_path=None,
                file_status="failed",
                json_status="failed",
            )
        except Exception:
            logger.exception("Failed to update file_metadata on job failure")
        await _job_set_status(
            db_pool,
            job_id,
            status="failed",
            error_message=str(e),
            finished_at=datetime.now(),
        )


@router.post("/jobs", response_model=CreateExtractionJobResponse)
async def create_extraction_job(
    file: UploadFile,
    type: str = Form(...),
    category: str = Form(...),
    tags: str = Form(...),
    publishedDate: str = Form(...),
    userId: str | None = Form(None),
    productType: str | None = Form(None),
    serviceType: str | None = Form(None),
    db_pool=Depends(get_db_pool),
    embeddings_model=Depends(get_embeddings_model),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    file_content = await file.read()
    file_md5_checksum = hashlib.md5(file_content).hexdigest()

    _ensure_pg_storage_dirs()
    stored_file_path = _file_path_for_upload(
        file_md5_checksum=file_md5_checksum, original_filename=file.filename
    )
    stored_file_path.write_bytes(file_content)

    tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

    published_date: Optional[datetime] = None
    try:
        published_date = datetime.fromisoformat(publishedDate.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(
            status_code=422, detail="Invalid publishedDate format"
        ) from None

    job_id = uuid.uuid4()

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO extraction_jobs (id, file_md5_checksum, file_name, status)
            VALUES ($1, $2, $3, $4)
            """,
            job_id,
            file_md5_checksum,
            file.filename,
            "queued",
        )

    await _upsert_file_metadata(
        db_pool,
        file_md5_checksum=file_md5_checksum,
        file_name=file.filename,
        type=type,
        category=category,
        tags=tags_list,
        parsed_at=None,
        published_date=published_date,
        user_id=userId,
        product_type=productType,
        service_type=serviceType,
        json_file_path=None,
        file_status="stored",
        json_status="queued",
    )

    asyncio.create_task(
        _run_extraction_job(
            db_pool=db_pool,
            embeddings_model=embeddings_model,
            job_id=job_id,
            file_name=file.filename,
            file_md5_checksum=file_md5_checksum,
            file_path=str(stored_file_path),
            type=type,
            category=category,
            tags=tags_list,
            user_id=userId,
            product_type=productType,
            service_type=serviceType,
            published_date=published_date,
        )
    )

    return CreateExtractionJobResponse(
        job_id=str(job_id), file_reference=file_md5_checksum
    )


@router.post("/jobs/retry/{file_reference}", response_model=CreateExtractionJobResponse)
async def retry_extraction_job(
    file_reference: str,
    db_pool=Depends(get_db_pool),
    embeddings_model=Depends(get_embeddings_model),
):
    stored_file = _find_uploaded_file_path(file_reference)
    if not stored_file:
        raise HTTPException(status_code=404, detail="Stored file not found for retry")

    async with db_pool.acquire() as conn:
        meta = await conn.fetchrow(
            "SELECT * FROM file_metadata WHERE file_md5_checksum = $1",
            file_reference,
        )

    if not meta:
        raise HTTPException(status_code=404, detail="Document metadata not found")

    job_id = uuid.uuid4()
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO extraction_jobs (id, file_md5_checksum, file_name, status) VALUES ($1, $2, $3, $4)",
            job_id,
            file_reference,
            meta["file_name"],
            "queued",
        )

    asyncio.create_task(
        _run_extraction_job(
            db_pool=db_pool,
            embeddings_model=embeddings_model,
            job_id=job_id,
            file_name=meta["file_name"],
            file_md5_checksum=file_reference,
            file_path=str(stored_file),
            type=meta["type"] or "pdf",
            category=meta["category"] or "resources",
            tags=(
                [str(x) for x in (meta["tags"] or [])]
                if isinstance(meta["tags"], list)
                else []
            ),
            user_id=meta["user_id"],
            product_type=meta["product_type"],
            service_type=meta["service_type"],
            published_date=meta["published_date"],
        )
    )

    return CreateExtractionJobResponse(
        job_id=str(job_id), file_reference=file_reference
    )


@router.get("/get_parsed_json/{file_reference}")
async def get_parsed_json_pg(file_reference: str, db_pool=Depends(get_db_pool)):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT json_file_path FROM file_metadata WHERE file_md5_checksum = $1",
            file_reference,
        )

    json_path: Path | None = None
    if row and row.get("json_file_path"):
        json_path = Path(str(row["json_file_path"]))
        if not json_path.exists():
            json_path = None

    if not json_path:
        json_path = _find_parsed_json_path(file_reference)

    if not json_path:
        raise HTTPException(status_code=404, detail="Parsed JSON not found")

    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read parsed JSON: {e}"
        ) from e


@router.get("/get_file/{file_reference}")
async def get_file_pg(file_reference: str):
    _ensure_pg_storage_dirs()
    stored_file = _find_uploaded_file_path(file_reference)
    if not stored_file:
        raise HTTPException(status_code=404, detail="Stored file not found")

    media_type, _ = mimetypes.guess_type(str(stored_file))
    return FileResponse(
        path=str(stored_file),
        media_type=media_type or "application/octet-stream",
        filename=stored_file.name,
    )


@router.get("/jobs/{job_id}", response_model=ExtractionJobStatusResponse)
async def get_extraction_job(job_id: str, db_pool=Depends(get_db_pool)):
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid job_id") from None

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                file_md5_checksum,
                file_name,
                status,
                error_message,
                created_at,
                started_at,
                finished_at,
                chunks_processed,
                pages_processed
            FROM extraction_jobs
            WHERE id = $1
            """,
            job_uuid,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return ExtractionJobStatusResponse(
        job_id=str(row["id"]),
        file_reference=row["file_md5_checksum"],
        file_name=row["file_name"],
        status=row["status"],
        error_message=row["error_message"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        chunks_processed=row["chunks_processed"],
        pages_processed=row["pages_processed"],
    )


@router.get(
    "/jobs/by-file/{file_reference}", response_model=List[ExtractionJobStatusResponse]
)
async def list_jobs_by_file(file_reference: str, db_pool=Depends(get_db_pool)):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                file_md5_checksum,
                file_name,
                status,
                error_message,
                created_at,
                started_at,
                finished_at,
                chunks_processed,
                pages_processed
            FROM extraction_jobs
            WHERE file_md5_checksum = $1
            ORDER BY created_at DESC
            """,
            file_reference,
        )

    return [
        ExtractionJobStatusResponse(
            job_id=str(row["id"]),
            file_reference=row["file_md5_checksum"],
            file_name=row["file_name"],
            status=row["status"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            chunks_processed=row["chunks_processed"],
            pages_processed=row["pages_processed"],
        )
        for row in rows
    ]
