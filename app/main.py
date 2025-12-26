import logging
import os

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from app.lifespan import lifespan
from app.routers import document_extraction, file_search, search, vector_search

# Load environment variables before importing modules that depend on them
load_dotenv()

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


# Initialize PostgreSQL connection pool
@app.on_event("startup")
async def startup():
    try:
        app.state.db_pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "vector_search"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            min_size=5,
            max_size=20,
        )
        logger.info("PostgreSQL connection pool created successfully")
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL connection pool: {e}")
        # Continue without PostgreSQL - vector search endpoints will fail


@app.on_event("shutdown")
async def shutdown():
    if hasattr(app.state, "db_pool"):
        await app.state.db_pool.close()
        logger.info("PostgreSQL connection pool closed")


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
