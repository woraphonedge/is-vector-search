# IS Vector Search

A FastAPI service providing document extraction, vector search, and RAG (Retrieval-Augmented Generation) chat capabilities with Google File Search integration.

## Features

- **Document extraction**: Parse and store documents with metadata
- **Vector search**: Semantic search using PostgreSQL + pgvector
- **RAG chat**: Chat with documents using LLM with retrieved context
- **Google File Search**: Upload and query files via Google Gemini API
- **Hermes integration**: Search and manage Hermes document corpus
- **Supabase backend**: Metadata and file storage with Supabase

## Architecture

- **FastAPI**: REST API framework
- **PostgreSQL + pgvector**: Vector database for semantic search
- **LangChain**: RAG pipeline and LLM orchestration
- **OpenAI**: LLM for chat responses
- **DeepInfra**: Embeddings model
- **Supabase**: Database and storage
- **Google Genai**: File Search API integration

## Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Supabase project (optional, for production)

### 1. Clone and install dependencies

```bash
git clone <repository-url>
cd is-vector-search
uv sync
```

### 2. Environment variables

Create a `.env` file in the project root:

```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
DEEP_INFRA_API_KEY=your_deep_infra_api_key_here

# Supabase (required)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_SECRET_KEY=your_supabase_secret_key

## Postgres (pgvector)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=vector_search
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Optional: File search store persistence
FILE_SEARCH_STORE_NAME=your_store_name

# Optional: Use Supabase Storage for parquet files
USE_SUPABASE_AS_SOURCE=True
SUPABASE_STORAGE_BUCKET=mcp
SUPABASE_STORAGE_PREFIX=data

# Optional: Hermes parquet path
HERMES_PARQUET_PATH=/path/to/hermes_files_conso.parquet
```

### 3. Run the application

```bash
# Development
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8200

# Or using the script
uv run python -m app.main
```

The API will be available at `http://localhost:8200`

## API Endpoints

### Core Search

- `GET /` - Welcome message
- `POST /search` - Search files using Gemini API (legacy)

### Document Management

- `POST /documents/extract` - Extract and parse documents
- `GET /documents` - List all documents
- `GET /documents/{file_reference}` - Get document metadata
- `PUT /documents/{file_reference}` - Update document metadata
- `DELETE /documents/{file_reference}` - Delete document and associated data

### Chat & RAG

- `POST /documents/chat` - Chat with documents using RAG

### Google File Search

- `POST /file-search/upload` - Upload file to Google File Search store
- `POST /file-search/query` - Query uploaded files with Gemini
- `GET /file-search/files` - List uploaded files

### Hermes Integration

- `GET /file-search/hermes-files` - Search Hermes document corpus
- `GET /file-search/hermes-facets` - Get Hermes search facets
- `POST /file-search/hermes-upload` - Upload to Hermes corpus

## Usage Examples

### Document Extraction

```bash
curl -X POST "http://localhost:8200/documents/extract" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Chat with Documents

```bash
curl -X POST "http://localhost:8200/documents/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key financial metrics?",
    "file_reference": "document_md5_hash"
  }'
```

### Upload File to Google File Search

```bash
# Upload file
curl -X POST "http://localhost:8200/file-search/upload" \
  -F "file=@document.pdf"

# Query files
curl -X POST "http://localhost:8200/file-search/query" \
  -H "Content-Type: application/json" \
  -d '{
    "store_name": "stores/your-store-name",
    "query": "Summarize the financial outlook",
    "model": "gemini-2.5-flash"
  }'
```

## Development

### Code Quality

```bash
# Lint and format
uv run ruff check .
uv run ruff format .
```

### Testing

```bash
# Run tests (if available)
uv run pytest
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# View logs
docker-compose logs -f
```

## Configuration

### PostgreSQL + pgvector

- Default port: `5432` (via docker-compose)
- Embeddings table: `document_embeddings`

### Embeddings

- Model: `BAAI/bge-m3` (via DeepInfra)
- Token limit: `4096` tokens per chunk

### LLM

- Model: `gpt-4.1-mini` (via OpenAI)
- Temperature: `0.1`

## Troubleshooting

### Common Issues

1. **Postgres connection failed**: Ensure Postgres container is running and healthy
2. **Supabase errors**: Verify SUPABASE_URL and SUPABASE_KEY are correct
3. **Embedding failures**: Check DEEP_INFRA_API_KEY is valid
4. **LLM errors**: Verify OPENAI_API_KEY has sufficient credits

### Debug Mode

Enable debug logging by setting:

```bash
export PYTHONPATH=.
uv run python -m app.main --log-level debug
```

## License

[Add your license information here]
