import asyncio
import os
import tempfile

from fastapi import APIRouter, Depends, File, Request, UploadFile
from google.genai import types
from pydantic import BaseModel

router = APIRouter()


class FileSearchUploadResponse(BaseModel):
    store_name: str
    operation_done: bool


class FileSearchQueryRequest(BaseModel):
    query: str
    store_name: str = os.getenv("FILE_SEARCH_STORE_NAME")
    model: str = "gemini-2.5-flash"


def get_genai_client(request: Request):
    """Dependency to get genai_client from app state"""
    return request.app.state.genai_client


def get_hermes_df(request: Request):
    """Dependency to get hermes_df from app state"""
    return getattr(request.app.state, "hermes_df", None)


@router.post("/file-search/upload", response_model=FileSearchUploadResponse)
async def upload_file_to_store(
    file: UploadFile = File(...), genai_client=Depends(get_genai_client)
):
    """Upload a file to a new file search store"""
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    store = genai_client.file_search_stores.create()

    upload_op = genai_client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store.name,
        file=tmp_path,
    )

    while not upload_op.done:
        await asyncio.sleep(5)
        upload_op = genai_client.operations.get(upload_op)

    return FileSearchUploadResponse(
        store_name=store.name, operation_done=upload_op.done
    )


@router.post("/file-search/query")
async def query_file_search(
    request: FileSearchQueryRequest, genai_client=Depends(get_genai_client)
):
    """Query a file search store"""
    response = genai_client.models.generate_content(
        model=request.model,
        contents=request.query,
        config=types.GenerateContentConfig(
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[request.store_name]
                    )
                )
            ]
        ),
    )

    text = getattr(response, "text", None)

    grounding_metadata = None
    if response.candidates:
        grounding_metadata = response.candidates[0].grounding_metadata

    sources = []
    if grounding_metadata and grounding_metadata.grounding_chunks:
        sources = list(
            {c.retrieved_context.title for c in grounding_metadata.grounding_chunks}
        )

    return {
        "text": text,
        "sources": sources,
    }


@router.get("/file-search/files")
async def list_files(genai_client=Depends(get_genai_client)):
    """List all uploaded files"""
    files = []
    for f in genai_client.files.list():
        files.append(
            {
                "name": getattr(f, "name", None),
                "display_name": getattr(f, "display_name", None),
            }
        )

    return {"files": files}


@router.get("/file-search/hermes-files")
async def list_hermes_files(
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
    page_id: int | None = None,
    section_id: int | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    df=Depends(get_hermes_df),
    genai_client=Depends(get_genai_client),
) -> dict:
    """List Hermes files with filtering options"""
    from app.hermes import list_hermes_files_logic

    return list_hermes_files_logic(
        df=df,
        genai_client=genai_client,
        q=q,
        page=page,
        page_size=page_size,
        page_id=page_id,
        section_id=section_id,
        created_from=created_from,
        created_to=created_to,
    )


@router.get("/file-search/hermes-facets")
async def list_hermes_facets(df=Depends(get_hermes_df)) -> dict:
    """List Hermes facets"""
    from app.hermes import list_hermes_facets_logic

    return list_hermes_facets_logic(df)


@router.post("/file-search/hermes-upload")
async def hermes_upload(request, genai_client=Depends(get_genai_client)) -> dict:
    """Upload to Hermes"""
    from app.hermes import hermes_upload_logic

    return await hermes_upload_logic(request=request, genai_client=genai_client)
