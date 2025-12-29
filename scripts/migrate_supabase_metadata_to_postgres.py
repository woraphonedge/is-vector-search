import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from dotenv import load_dotenv
from supabase import create_client


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _parse_tags(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p]
    return None


async def _connect_postgres() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "vector_search"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def _connect_supabase():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_KEY) must be set"
        )
    return create_client(supabase_url, supabase_key)


def _fetch_supabase_file_metadata(
    supabase, batch_size: int = 1000
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        resp = (
            supabase.table("file_metadata")
            .select("*")
            .order("parsed_at", desc=True)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < batch_size:
            break
        offset += batch_size
    return all_rows


async def _upsert_postgres(
    conn: asyncpg.Connection,
    *,
    file_md5_checksum: str,
    file_name: str,
    type: Optional[str],
    category: Optional[str],
    tags: Optional[List[str]],
    parsed_at: Optional[datetime],
    published_date: Optional[datetime],
    user_id: Optional[str],
    product_type: Optional[str],
    service_type: Optional[str],
    json_file_path: Optional[str],
    file_status: Optional[str],
    json_status: Optional[str],
    created_at: Optional[datetime],
    updated_at: Optional[datetime],
) -> Tuple[int, int]:
    tags_json = json.dumps(tags) if tags is not None else None

    # 1) Upsert into files (used by embeddings)
    file_row = await conn.fetchrow(
        """
        INSERT INTO files (file_name, file_md5_checksum, published_date, created_at, processed_at)
        VALUES ($1, $2, $3, COALESCE($4, NOW()), NOW())
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
        created_at,
    )
    file_id = int(file_row["id"])

    # 2) Upsert into file_metadata (mimic supabase)
    await conn.execute(
        """
        INSERT INTO file_metadata (
            file_md5_checksum, file_name, type, category, tags,
            parsed_at, published_date, user_id, product_type, service_type,
            json_file_path, file_status, json_status,
            created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5::jsonb,
            $6, $7, $8, $9, $10,
            $11, $12, $13,
            COALESCE($14, NOW()), COALESCE($15, NOW())
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
            json_status = EXCLUDED.json_status,
            created_at = EXCLUDED.created_at,
            updated_at = EXCLUDED.updated_at
        """,
        file_md5_checksum,
        file_name,
        type,
        category,
        tags_json,
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
    )

    return file_id, 1


async def main() -> None:
    load_dotenv()

    supabase = _connect_supabase()
    rows = _fetch_supabase_file_metadata(supabase)
    print(f"Fetched {len(rows)} rows from Supabase file_metadata")

    conn = await _connect_postgres()
    try:
        migrated = 0
        for row in rows:
            file_md5_checksum = row.get("file_md5_checksum")
            file_name = row.get("file_name")
            if not file_md5_checksum or not file_name:
                continue

            tags = _parse_tags(row.get("tags"))

            await _upsert_postgres(
                conn,
                file_md5_checksum=str(file_md5_checksum),
                file_name=str(file_name),
                type=row.get("type"),
                category=row.get("category"),
                tags=tags,
                parsed_at=_parse_dt(row.get("parsed_at")),
                published_date=_parse_dt(row.get("published_date")),
                user_id=row.get("user_id"),
                product_type=row.get("product_type"),
                service_type=row.get("service_type"),
                json_file_path=row.get("json_file_path"),
                file_status=row.get("file_status"),
                json_status=row.get("json_status"),
                created_at=_parse_dt(row.get("created_at")),
                updated_at=_parse_dt(row.get("updated_at")),
            )
            migrated += 1

        print(f"Migrated {migrated} rows into Postgres (files + file_metadata)")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
