-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Mimic Supabase file_metadata table used by the legacy document extraction API
CREATE TABLE IF NOT EXISTS file_metadata (
    id SERIAL PRIMARY KEY,
    file_md5_checksum VARCHAR(64) UNIQUE NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    type TEXT,
    category TEXT,
    tags JSONB,
    parsed_at TIMESTAMP WITH TIME ZONE,
    published_date TIMESTAMP WITH TIME ZONE,
    user_id TEXT,
    product_type TEXT,
    service_type TEXT,
    json_file_path TEXT,
    file_status TEXT,
    json_status TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_file_metadata_checksum ON file_metadata(file_md5_checksum);

CREATE INDEX IF NOT EXISTS idx_file_metadata_parsed_at ON file_metadata(parsed_at);

-- Long-running extraction job tracking
CREATE TABLE IF NOT EXISTS extraction_jobs (
    id UUID PRIMARY KEY,
    file_md5_checksum VARCHAR(64) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    chunks_processed INTEGER,
    pages_processed INTEGER
);

CREATE INDEX IF NOT EXISTS idx_extraction_jobs_checksum ON extraction_jobs(file_md5_checksum);

CREATE INDEX IF NOT EXISTS idx_extraction_jobs_created_at ON extraction_jobs(created_at);

-- Create a table for file metadata (matching Chroma schema)
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_md5_checksum VARCHAR(64) UNIQUE NOT NULL,
    published_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create a table for document embeddings (mimicking Chroma metadata structure)
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    document_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    -- OpenAI embedding dimension
    file_name VARCHAR(255) NOT NULL,
    file_md5_checksum VARCHAR(64) NOT NULL,
    published_date TIMESTAMP WITH TIME ZONE,
    page_number INTEGER,
    chunk_type VARCHAR(50) NOT NULL,
    token_count INTEGER,
    chroma_document TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better search performance
CREATE INDEX IF NOT EXISTS idx_document_embeddings_chunk_id ON document_embeddings(chunk_id);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_file_id ON document_embeddings(file_id);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_document_id ON document_embeddings(document_id);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_chunk_type ON document_embeddings(chunk_type);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_page_number ON document_embeddings(page_number);

CREATE INDEX IF NOT EXISTS idx_document_embeddings_token_count ON document_embeddings(token_count);

CREATE INDEX IF NOT EXISTS idx_files_checksum ON files(file_md5_checksum);

CREATE INDEX IF NOT EXISTS idx_files_name ON files(file_name);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_document_embeddings_embedding ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Create or update function to automatically update updated_at
CREATE
OR REPLACE FUNCTION update_updated_at_column() RETURNS TRIGGER AS $ $ BEGIN NEW.updated_at = NOW();

RETURN NEW;

END;

-- $ $ language 'plpgsql';

-- Create trigger for document_embeddings
CREATE TRIGGER update_document_embeddings_updated_at BEFORE
UPDATE
    ON document_embeddings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for file_metadata
CREATE TRIGGER update_file_metadata_updated_at BEFORE
UPDATE
    ON file_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for extraction_jobs
CREATE TRIGGER update_extraction_jobs_updated_at BEFORE
UPDATE
    ON extraction_jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create a view for easy access to document with embeddings
CREATE
OR REPLACE VIEW documents_with_embeddings AS
SELECT
    f.id as file_id,
    f.file_name,
    f.file_md5_checksum,
    f.published_date,
    f.created_at as file_created_at,
    de.id as embedding_id,
    de.chunk_id,
    de.document_id,
    de.content,
    de.page_number,
    de.chunk_type,
    de.token_count,
    de.chroma_document,
    de.created_at as embedding_created_at,
    de.updated_at
FROM
    files f
    LEFT JOIN document_embeddings de ON f.id = de.file_id;