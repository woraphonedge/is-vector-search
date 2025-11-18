import os
import tempfile
import time
from contextlib import asynccontextmanager

import httpx
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

from app.hermes import (
    HermesUploadRequest,
    hermes_upload_logic,
    list_hermes_facets_logic,
    list_hermes_files_logic,
)
from app.routers import document_extraction
from app.utils.supabase_db import initialize_env

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Hermes parquet once at startup and cache in app.state."""
    hermes_path = os.getenv(
        "HERMES_PARQUET_PATH",
        "/Users/home/projects/mcp-server/data/hermes_files_conso.parquet",
    )

    try:
        app.state.hermes_df = pd.read_parquet(hermes_path)
    except Exception:  # pragma: no cover - runtime safety
        # Cache failure; endpoint will surface this if accessed
        app.state.hermes_df = None

    # Initialize Supabase, Chroma, and related globals used by document_extraction
    initialize_env()

    # Store genai_client in app.state for router access
    app.state.genai_client = genai_client

    # Create or reuse a shared file search store for document extraction
    store_name = os.getenv("FILE_SEARCH_STORE_NAME")
    if store_name:
        try:
            # Try to use existing store from environment
            app.state.file_search_store_name = store_name
            print(f"Using existing file search store: {store_name}")
        except Exception as e:
            print(f"Failed to use existing store {store_name}: {e}")
            app.state.file_search_store_name = None
    else:
        try:
            # Create new store and save name to environment
            store = genai_client.file_search_stores.create()
            app.state.file_search_store_name = store.name
            print(f"Created new file search store: {store.name}")

            # Save to .env file for persistence (avoid duplicates)
            env_file = ".env"
            if os.path.exists(env_file):
                with open(env_file, "r") as f:
                    content = f.read()
                if "FILE_SEARCH_STORE_NAME=" not in content:
                    with open(env_file, "a") as f:
                        f.write(f"\nFILE_SEARCH_STORE_NAME={store.name}")
            else:
                with open(env_file, "w") as f:
                    f.write(f"FILE_SEARCH_STORE_NAME={store.name}")

        except Exception as e:
            print(f"Failed to create file search store: {e}")
            app.state.file_search_store_name = None

    yield


app = FastAPI(title="Google File Search API", lifespan=lifespan)

genai_client = genai.Client()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include document extraction router
app.include_router(document_extraction.router, prefix="/documents", tags=["documents"])


# Request model
class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class FileSearchUploadResponse(BaseModel):
    store_name: str
    operation_done: bool


class FileSearchQueryRequest(BaseModel):
    query: str
    store_name: str = os.getenv("FILE_SEARCH_STORE_NAME")
    model: str = "gemini-2.5-flash"


# Mock Gemini API URL (replace with actual URL when available)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Google File Search API. Use /search to search files."
    }


@app.post("/search")
async def search_files(request: SearchRequest):
    """
    Search files using the Gemini API
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY environment variable not set"
        )

    headers = {
        "Content-Type": "application/json",
    }

    # Prepare the request payload for Gemini API
    payload = {
        "contents": [
            {"parts": [{"text": f"Search for files related to: {request.query}"}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 800,
            "temperature": 0.9,
            "topP": 1,
            "topK": 40,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers=headers,
                json=payload,
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from Gemini API: {response.text}",
                )

            # Process and return the response
            result = response.json()
            return {
                "query": request.query,
                "results": result.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No results found"),
                "status": "success",
            }

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail=f"Error connecting to Gemini API: {str(e)}"
        ) from e


@app.post("/file-search/upload", response_model=FileSearchUploadResponse)
async def upload_file_to_store(file: UploadFile = File(...)):
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
        time.sleep(5)
        upload_op = genai_client.operations.get(upload_op)

    return FileSearchUploadResponse(
        store_name=store.name, operation_done=upload_op.done
    )


@app.post("/file-search/query")
async def query_file_search(request: FileSearchQueryRequest):
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


@app.get("/file-search/files")
async def list_files():
    files = []
    for f in genai_client.files.list():
        files.append(
            {
                "name": getattr(f, "name", None),
                "display_name": getattr(f, "display_name", None),
            }
        )

    return {"files": files}


@app.get("/file-search/hermes-files")
async def list_hermes_files(
    q: str | None = None,
    page: int = 1,
    page_size: int = 20,
    page_id: int | None = None,
    section_id: int | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> dict:
    df = getattr(app.state, "hermes_df", None)
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


@app.get("/file-search/hermes-facets")
async def list_hermes_facets() -> dict:
    df = getattr(app.state, "hermes_df", None)
    return list_hermes_facets_logic(df)


@app.post("/file-search/hermes-upload")
async def hermes_upload(request: HermesUploadRequest) -> dict:
    return await hermes_upload_logic(request=request, genai_client=genai_client)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)


if __name__ == "__main__":
    main()
