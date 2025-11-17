import os
import tempfile
from typing import Any, Dict, List, Optional, Set

import httpx
import pandas as pd
from fastapi import HTTPException
from google import genai
from pydantic import BaseModel


class HermesFile(BaseModel):
    page_id: Optional[int] = None
    page_name: Optional[str] = None
    section_id: Optional[int] = None
    section_name: Optional[str] = None
    file_id: Optional[int] = None
    file_name: str
    file_url: str
    created_datetime: Optional[str] = None
    created_by: Optional[str] = None
    in_file_search: bool = False


class HermesUploadRequest(BaseModel):
    file_url: str
    file_name: str
    page_id: Optional[int] = None
    page_name: Optional[str] = None
    section_id: Optional[int] = None
    section_name: Optional[str] = None
    created_by: Optional[str] = None
    created_datetime: Optional[str] = None


def _normalize_text(value: Optional[str]) -> str:
    """Normalize text for robust matching.

    - Handle None safely
    - Strip leading/trailing whitespace
    - Lowercase
    - Replace dashes/underscores with spaces
    - Collapse multiple spaces
    - Strip common file extension suffixes
    """

    if value is None:
        return ""

    text = value.strip().lower()
    # Replace common separators with spaces
    text = text.replace("-", " ").replace("_", " ")
    # Strip simple extensions
    for ext in [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt"]:
        if text.endswith(ext):
            text = text[: -len(ext)]
            break
    # Collapse multiple spaces
    text = " ".join(text.split())
    return text


def list_hermes_files_logic(
    df: pd.DataFrame,
    genai_client: genai.Client,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    page_id: Optional[int] = None,
    section_id: Optional[int] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
) -> Dict[str, Any]:
    if df is None:
        raise HTTPException(
            status_code=500,
            detail="Hermes data not loaded. Check HERMES_PARQUET_PATH and restart the service.",
        )

    filtered = df.copy()

    # Precompute normalized columns for search
    filtered["_norm_file_name"] = filtered["file_name"].astype(str).map(_normalize_text)
    filtered["_norm_page_name"] = filtered["page_name"].astype(str).map(_normalize_text)
    filtered["_norm_section_name"] = (
        filtered["section_name"].astype(str).map(_normalize_text)
    )

    if q:
        q_norm = _normalize_text(q)
        filtered = filtered[
            filtered["_norm_file_name"].str.contains(q_norm)
            | filtered["_norm_page_name"].str.contains(q_norm)
            | filtered["_norm_section_name"].str.contains(q_norm)
        ]

    if page_id is not None:
        filtered = filtered[filtered["page_id"] == page_id]

    if section_id is not None:
        filtered = filtered[filtered["section_id"] == section_id]

    if created_from:
        filtered = filtered[filtered["created_datetime"] >= created_from]

    if created_to:
        filtered = filtered[filtered["created_datetime"] <= created_to]

    total = int(len(filtered))
    if page_size <= 0:
        page_size = 20
    if page <= 0:
        page = 1

    total_pages = max((total + page_size - 1) // page_size, 1) if total > 0 else 1
    if page > total_pages:
        page = total_pages

    start = (page - 1) * page_size
    end = start + page_size
    page_df = filtered.iloc[start:end]

    file_search_names: Set[str] = set()
    try:
        for f in genai_client.files.list():
            display_name = getattr(f, "display_name", None)
            if display_name:
                file_search_names.add(_normalize_text(str(display_name)))
    except Exception:
        file_search_names = set()

    rows: List[HermesFile] = []
    for _, row in page_df.iterrows():
        file_name = str(row.get("file_name", "")).strip()
        file_url = str(row.get("file_url", "")).strip()

        if not file_name or not file_url:
            continue

        in_file_search = _normalize_text(file_name) in file_search_names

        rows.append(
            HermesFile(
                page_id=row.get("page_id"),
                page_name=row.get("page_name"),
                section_id=row.get("section_id"),
                section_name=row.get("section_name"),
                file_id=row.get("file_id"),
                file_name=file_name,
                file_url=file_url,
                created_datetime=(
                    str(row.get("created_datetime"))
                    if row.get("created_datetime") is not None
                    else None
                ),
                created_by=row.get("created_by"),
                in_file_search=in_file_search,
            )
        )

    return {
        "files": rows,
        "page": page,
        "total_pages": total_pages,
        "total": total,
    }


def list_hermes_facets_logic(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None:
        raise HTTPException(
            status_code=500,
            detail="Hermes data not loaded. Check HERMES_PARQUET_PATH and restart the service.",
        )

    pages = (
        df[["page_id", "page_name", "page_description"]]
        .dropna(subset=["page_id"])
        .drop_duplicates()
        .sort_values(by=["page_id"])
    )
    sections = (
        df[["section_id", "section_name", "section_description"]]
        .dropna(subset=["section_id"])
        .drop_duplicates()
        .sort_values(by=["section_id"])
    )

    page_items = [
        {
            "id": int(row["page_id"]),
            "name": row.get("page_name") or "",
            "description": row.get("page_description") or "",
        }
        for _, row in pages.iterrows()
    ]

    section_items = [
        {
            "id": int(row["section_id"]),
            "name": row.get("section_name") or "",
            "description": row.get("section_description") or "",
        }
        for _, row in sections.iterrows()
    ]

    return {"pages": page_items, "sections": section_items}


async def hermes_upload_logic(
    request: HermesUploadRequest,
    genai_client: genai.Client,
) -> Dict[str, Any]:
    store_name = os.getenv("HERMES_FILE_SEARCH_STORE_NAME")
    if not store_name:
        raise HTTPException(
            status_code=500,
            detail="HERMES_FILE_SEARCH_STORE_NAME environment variable is not set",
        )

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, request.file_name)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(request.file_url, timeout=60.0)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Failed to download file from URL: {resp.text}",
                )
            with open(tmp_path, "wb") as f:
                f.write(resp.content)

        uploaded_file = genai_client.files.upload(
            file=tmp_path,
            config={"name": request.file_name},
        )

        metadata = [
            (
                {"key": "page_id", "numeric_value": request.page_id}
                if request.page_id is not None
                else None
            ),
            (
                {"key": "page_name", "string_value": request.page_name}
                if request.page_name is not None
                else None
            ),
            (
                {"key": "section_id", "numeric_value": request.section_id}
                if request.section_id is not None
                else None
            ),
            (
                {"key": "section_name", "string_value": request.section_name}
                if request.section_name is not None
                else None
            ),
            (
                {"key": "created_by", "string_value": request.created_by}
                if request.created_by is not None
                else None
            ),
            (
                {
                    "key": "created_datetime",
                    "string_value": request.created_datetime,
                }
                if request.created_datetime is not None
                else None
            ),
        ]

        metadata = [m for m in metadata if m is not None]

        op = genai_client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=uploaded_file.name,
            config={"custom_metadata": metadata} if metadata else None,
        )

        return {
            "status": "ok",
            "store_name": store_name,
            "file_name": uploaded_file.name,
            "operation": str(op.name) if hasattr(op, "name") else None,
        }
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass
