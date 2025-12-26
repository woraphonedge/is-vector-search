import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI

from app.utils.supabase_db import initialize_env

logger = logging.getLogger("is_vector_search")


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

    # Initialize Supabase, Chroma, and related globals used by document_extraction
    initialize_env()

    # Store genai_client in app.state for router access
    # Note: genai_client should be created before passing to lifespan
    if hasattr(app.state, "genai_client"):
        logger.info("genai_client already set in app.state")
    else:
        logger.warning("genai_client not found in app.state")

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

    yield
