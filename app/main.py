import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai

# Load environment variables before importing modules that depend on them
load_dotenv()

from app.lifespan import lifespan
from app.routers import (
    document_extraction,
    document_extraction_pg,
    file_search,
    search,
    vector_search,
)

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("is_vector_search")


app = FastAPI(title="Google File Search API", lifespan=lifespan)

genai_client = genai.Client()

# Store genai_client in app.state for router access
app.state.genai_client = genai_client


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(document_extraction.router, prefix="/documents", tags=["documents"])
app.include_router(
    document_extraction_pg.router, prefix="/documents-pg", tags=["documents-pg"]
)
app.include_router(search.router, tags=["search"])
app.include_router(file_search.router, tags=["file-search"])
app.include_router(vector_search.router, prefix="/pgvector", tags=["pgvector"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to Google File Search API. Use /search to search files."
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8200)


if __name__ == "__main__":
    main()
