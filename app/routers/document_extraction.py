import datetime
import hashlib
import json
import logging
import os
from typing import List

from fastapi import (
    APIRouter,
    Body,
    Depends,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from landingai_ade import LandingAIADE

from app.models import (
    ChatRequest,
    ChatResponse,
    DocumentMetadata,
    ExtractionResponse,
    SourceDocument,
)
from app.utils.rag_utils import (
    get_rag_response,
    process_and_insert_documents,
)
from app.utils.supabase_client import get_supabase_client
from app.utils.supabase_db import SupabaseDatabase

router = APIRouter()

logger = logging.getLogger("is_vector_search")

ade_client = LandingAIADE(apikey=os.getenv("VISION_AGENT_API_KEY"))


def get_supabase_db():
    """FastAPI dependency to get Supabase database instance."""
    return SupabaseDatabase()


@router.get(
    "/list_parsed_documents",
    response_model=List[DocumentMetadata],
    response_model_by_alias=True,
    response_model_exclude_none=False,
)
def list_parsed_documents(
    db: SupabaseDatabase = Depends(get_supabase_db),
):  # noqa: B008
    metadata_list = db.get_all_metadata()
    logger.info("Retrieved %d metadata records", len(metadata_list))
    res = [
        DocumentMetadata(
            id=metadata.id,
            file_md5_checksum=metadata.file_md5_checksum,
            file_name=metadata.file_name,
            type=metadata.type,
            category=metadata.category,
            tags=metadata.tags,
            parsed_at=metadata.parsed_at,
            published_date=metadata.published_date,
            user_id=metadata.user_id,
            product_type=metadata.product_type,
            service_type=metadata.service_type,
            json_file_path=metadata.json_file_path,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
        )
        for metadata in metadata_list
    ]
    print(f"API response: {res}")
    return res


@router.get("/get_parsed_json/{file_reference}")
def get_parsed_json(
    file_reference: str, db: SupabaseDatabase = Depends(get_supabase_db)
):  # noqa: B008
    metadata = db.get_metadata(file_reference)
    if metadata:
        try:
            # Get Supabase client
            supabase = get_supabase_client()

            # Extract filename from json_file_path (supabase://parsed-json/{filename}.json)
            json_path = metadata.json_file_path
            if json_path.startswith("supabase://"):
                bucket_name = "parsed-json"
                filename = json_path.replace("supabase://parsed-json/", "")

                # Download from Supabase Storage
                response = supabase.storage.from_(bucket_name).download(filename)

                if response:
                    return json.loads(response)
                else:
                    raise HTTPException(
                        status_code=404,
                        detail="Parsed JSON file not found in Supabase Storage.",
                    )
            else:
                # Fallback to local file for backward compatibility
                try:
                    with open(json_path, "r") as f:
                        return json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    raise HTTPException(
                        status_code=404,
                        detail="Parsed JSON file not found or is corrupt.",
                    ) from None

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error retrieving parsed JSON: {str(e)}"
            ) from e

    raise HTTPException(
        status_code=404, detail="File reference (MD5 checksum) not found."
    )


@router.get("/get_file/{file_reference}")
def get_file(file_reference: str):
    """Retrieve a stored file by its reference (MD5 checksum) from Supabase Storage."""
    try:
        supabase = get_supabase_client()

        # List files in documents bucket to find the correct extension
        files = supabase.storage.from_("documents").list()
        print(files)

        for file_info in files:
            if file_info["name"].startswith(file_reference):
                # Download the file
                response = supabase.storage.from_("documents").download(
                    file_info["name"]
                )

                if response:
                    # Determine content type based on extension
                    content_type = "application/octet-stream"
                    if file_info["name"].endswith(".pdf"):
                        content_type = "application/pdf"
                    elif file_info["name"].endswith(".txt"):
                        content_type = "text/plain"
                    elif file_info["name"].endswith(".docx"):
                        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    elif file_info["name"].endswith(".pptx"):
                        content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

                    return Response(
                        content=response,
                        media_type=content_type,
                        headers={
                            "Content-Disposition": f"attachment; filename={file_info['name']}"
                        },
                    )

        raise HTTPException(
            status_code=404, detail="File not found in Supabase Storage."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving file: {str(e)}"
        ) from e


@router.delete("/documents/{file_reference}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document_endpoint(
    file_reference: str, db: SupabaseDatabase = Depends(get_supabase_db)  # noqa: B008
):
    """Deletes a document and all its associated data from Supabase."""
    db.delete_metadata_and_files(file_reference)


@router.patch(
    "/documents/{file_reference}",
    response_model=DocumentMetadata,
    response_model_by_alias=True,
    response_model_exclude_none=False,
)
def update_document(
    file_reference: str,
    updates: dict = Body(...),
    db: SupabaseDatabase = Depends(get_supabase_db),  # noqa: B008
):
    """Update document metadata fields using file_reference as ID.

    Args:
        file_reference: The MD5 checksum of the document
        updates: Document metadata updates (all fields optional)

    Returns:
        Updated document metadata

    Example:
        ```json
        {
            "name": "Updated Document Name.pdf",
            "category": "research",
            "tags": ["important", "2024"],
            "publishedDate": "2024-01-15T10:00:00"
        }
        ```
    """
    # First check if document exists
    existing_metadata = db.get_metadata(file_reference)
    if not existing_metadata:
        raise HTTPException(
            status_code=404, detail="File reference (MD5 checksum) not found."
        )

    print(f"updates: {updates}")
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update.")
    try:
        updated_metadata = db.update_metadata(file_reference, updates)
        if updated_metadata:
            return updated_metadata
        else:
            raise HTTPException(
                status_code=404, detail="Failed to retrieve updated metadata."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update document: {str(e)}"
        ) from e


@router.post("/extract_document", response_model=ExtractionResponse)
async def extract_document_from_api(
    file: UploadFile,
    db: SupabaseDatabase = Depends(get_supabase_db),  # noqa: B008
    type: str = Form(...),
    category: str = Form(...),
    tags: str = Form(...),  # Expects a comma-separated string
    userId: str | None = Form(None),
    productType: str | None = Form(None),
    serviceType: str | None = Form(None),
    publishedDate: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    file_content = await file.read()
    md5_checksum = hashlib.md5(file_content).hexdigest()

    # Prepare tags and initial json file path
    tags_list = [tag.strip() for tag in tags.split(",")]
    json_file_path = f"supabase://parsed-json/{md5_checksum}.json"

    # 1) Insert initial metadata with statuses = "queuing"
    initial_metadata = DocumentMetadata(
        file_md5_checksum=md5_checksum,
        file_name=file.filename,
        type=type,
        category=category,
        tags=tags_list,
        parsed_at=datetime.datetime.now(),
        published_date=datetime.datetime.fromisoformat(publishedDate),
        user_id=userId,
        product_type=productType,
        service_type=serviceType,
        json_file_path=json_file_path,
        file_status="queuing",
        json_status="queuing",
    )
    db.add_metadata(initial_metadata)

    # 2) Upload original file to Supabase Storage (documents bucket)
    file_extension = os.path.splitext(file.filename)[1]
    stored_file_name = f"{md5_checksum}{file_extension}"
    document_path = f"{stored_file_name}"
    db.supabase.storage.from_("documents").upload(
        document_path,
        file_content,
        {"content-type": file.content_type or "application/pdf"},
    )
    # Update file_status to uploaded
    db.update_metadata(md5_checksum, {"fileStatus": "uploaded"})
    # Insert file_storage record
    try:
        db.add_file_storage(
            {
                "file_md5_checksum": md5_checksum,
                "storage_path": document_path,
                "file_name": file.filename,
                "file_size": len(file_content),
                "content_type": file.content_type or "application/pdf",
                "bucket_name": "documents",
                "metadata": {
                    "type": type,
                    "category": category,
                    "userId": userId,
                    "productType": productType,
                    "serviceType": serviceType,
                },
            }
        )
    except Exception as e:
        print(f"Warning: Failed to insert file_storage record: {e}")

    try:
        parsed_json = await call_landing_ai_parser(file_content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse document with external API: {e}"
        ) from e

    # 3) Save parsed JSON to Supabase Storage
    json_content = json.dumps(parsed_json, indent=4)
    json_path = f"{md5_checksum}.json"

    try:
        db.supabase.storage.from_("parsed-json").upload(
            json_path,
            json_content.encode("utf-8"),
            {"content-type": "application/json"},
        )
        # Update json_file_path and mark json_status as uploaded
        db.update_metadata(
            md5_checksum, {"jsonFilePath": json_file_path, "jsonStatus": "uploaded"}
        )
    except Exception as e:
        print(f"Warning: Failed to upload JSON to Supabase Storage: {e}")

    process_and_insert_documents(
        parsed_json, file.filename, md5_checksum, publishedDate
    )
    return ExtractionResponse(
        message="Document processed successfully.",
        file_reference=md5_checksum,
        parsed_json=parsed_json,
    )


async def call_landing_ai_parser(file_content: bytes) -> dict:
    api_key = os.getenv("VISION_AGENT_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Please set the VISION_AGENT_API_KEY environment variable."
        )

    try:
        # Use the new landingai_ade client
        # The client accepts bytes directly for document parameter
        response = ade_client.parse(
            document=file_content,
            model="dpt-2",
        )
        # Convert Pydantic response to dict
        return response.to_dict()
    except Exception as e:
        print(f"Error calling Landing AI API: {e}")
        raise


@router.post("/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    if not request.query or not request.file_reference:
        raise HTTPException(
            status_code=400, detail="Query and file_reference are required."
        )

    try:
        rag_response = await get_rag_response(request)

        sources = [
            SourceDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in rag_response["context"]
        ]

        return ChatResponse(answer=rag_response["answer"], sources=sources)
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get chat response."
        ) from e
