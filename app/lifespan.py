import logging
import os
from contextlib import asynccontextmanager

import asyncpg
import pandas as pd
from fastapi import FastAPI
from langchain_community.embeddings import DeepInfraEmbeddings
from openai import AsyncOpenAI

logger = logging.getLogger("is_vector_search")


class EmbeddingsProvider:
    def __init__(
        self,
        openrouter_client: AsyncOpenAI | None,
        openrouter_model: str | None,
        openrouter_extra_headers: dict[str, str] | None,
        deepinfra_model: DeepInfraEmbeddings | None,
    ):
        self._openrouter_client = openrouter_client
        self._openrouter_model = openrouter_model
        self._openrouter_extra_headers = openrouter_extra_headers
        self._deepinfra_model = deepinfra_model

    async def aembed_query(self, text: str):
        if self._openrouter_client and self._openrouter_model:
            try:
                resp = await self._openrouter_client.embeddings.create(
                    model=self._openrouter_model,
                    input=text,
                    encoding_format="float",
                    extra_headers=self._openrouter_extra_headers,
                )
                return resp.data[0].embedding
            except Exception as e:
                logger.warning(f"OpenRouter embeddings failed, falling back: {e}")

        if self._deepinfra_model:
            return self._deepinfra_model.embed_query(text)

        raise RuntimeError("No embeddings provider configured")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load Hermes parquet once at startup and cache in app.state."""
    hermes_path = os.getenv(
        "HERMES_PARQUET_PATH",
        "./data/hermes_files_conso.parquet",
    )

    try:
        app.state.hermes_df = pd.read_parquet(hermes_path)
        logger.info(f"Successfully loaded Hermes parquet from {hermes_path}")
    except Exception as e:  # pragma: no cover - runtime safety
        # Cache failure; endpoint will surface this if accessed
        app.state.hermes_df = None
        logger.error(f"Failed to load Hermes parquet: {e}")

    # Store genai_client in app.state for router access
    # Note: genai_client should be created before passing to lifespan
    if hasattr(app.state, "genai_client"):
        logger.info("genai_client already set in app.state")
    else:
        logger.warning("genai_client not found in app.state")

    # Initialize embeddings model once at startup
    try:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_client = (
            AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            if openrouter_api_key
            else None
        )

        openrouter_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "baai/bge-m3")

        openrouter_headers: dict[str, str] = {}
        openrouter_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if openrouter_referer:
            openrouter_headers["HTTP-Referer"] = openrouter_referer
        openrouter_title = os.getenv("OPENROUTER_X_TITLE")
        if openrouter_title:
            openrouter_headers["X-Title"] = openrouter_title

        deepinfra_api_key = os.getenv("DEEP_INFRA_API_KEY")
        deepinfra_model = (
            DeepInfraEmbeddings(
                model_id="BAAI/bge-m3",
                query_instruction="",
                embed_instruction="",
                deepinfra_api_token=deepinfra_api_key,
            )
            if deepinfra_api_key
            else None
        )

        app.state.embeddings_model = EmbeddingsProvider(
            openrouter_client=openrouter_client,
            openrouter_model=openrouter_model,
            openrouter_extra_headers=openrouter_headers or None,
            deepinfra_model=deepinfra_model,
        )
        logger.info("Embeddings provider initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings provider: {e}")
        app.state.embeddings_model = None

    # Create or reuse a shared file search store for document extraction
    store_name = os.getenv("FILE_SEARCH_STORE_NAME")
    if store_name:
        try:
            # Try to use existing store from environment
            app.state.file_search_store_name = store_name
            logger.info(f"Using existing file search store: {store_name}")
        except Exception as e:
            logger.error(f"Failed to use existing store {store_name}: {e}")
            app.state.file_search_store_name = None
    else:
        genai_client = getattr(app.state, "genai_client", None)
        if genai_client:
            try:
                # Create new store and save name to environment
                store = genai_client.file_search_stores.create()
                app.state.file_search_store_name = store.name
                logger.info(f"Created new file search store: {store.name}")

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
                logger.error(f"Failed to create file search store: {e}")
                app.state.file_search_store_name = None
        else:
            logger.warning("No genai_client available to create file search store")

    # Initialize PostgreSQL connection pool
    try:
        pg_host = os.getenv("POSTGRES_HOST")
        pg_port = os.getenv("POSTGRES_PORT")
        pg_db = os.getenv("POSTGRES_DB")
        pg_user = os.getenv("POSTGRES_USER")
        pg_password = os.getenv("POSTGRES_PASSWORD")

        missing = [
            name
            for name, val in (
                ("POSTGRES_HOST", pg_host),
                ("POSTGRES_PORT", pg_port),
                ("POSTGRES_DB", pg_db),
                ("POSTGRES_USER", pg_user),
                ("POSTGRES_PASSWORD", pg_password),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                "Missing required Postgres env vars: " + ", ".join(missing)
            )

        app.state.db_pool = await asyncpg.create_pool(
            host=pg_host,
            port=int(pg_port),
            database=pg_db,
            user=pg_user,
            password=pg_password,
            min_size=5,
            max_size=20,
        )
        logger.info("PostgreSQL connection pool created successfully")
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL connection pool: {e}")
        # Continue without PostgreSQL - vector search endpoints will fail

    yield

    # Cleanup: Close PostgreSQL connection pool
    if hasattr(app.state, "db_pool"):
        await app.state.db_pool.close()
        logger.info("PostgreSQL connection pool closed")
