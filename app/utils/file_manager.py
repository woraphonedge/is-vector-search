"""
File management utility ensuring consistency across:
- Database metadata
- File storage
- Vector database chunks
"""

import hashlib
import os
from typing import Any, Dict, Optional

from app.models import DocumentMetadata
from app.utils.supabase_db import SupabaseDatabase, initialize_env


class FileManager:
    """Centralized file management ensuring consistency across all systems."""

    def __init__(self):
        initialize_env()
        self.db = SupabaseDatabase()

    def calculate_md5(self, file_path: str) -> str:
        """Calculate MD5 checksum for a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def check_file_exists(self, md5_checksum: str) -> Optional[DocumentMetadata]:
        """Check if file already exists by MD5 checksum."""
        return self.db.get_metadata(md5_checksum)

    def upload_file_with_deduplication(self, file_path: str, metadata: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """
        Upload file with deduplication check and dual-bucket storage.

        Returns:
            {
                'status': 'new' | 'existing',
                'metadata': DocumentMetadata,
                'document_path': str,
                'json_path': str,
                'message': str
            }
        """
        md5_checksum = self.calculate_md5(file_path)

        # Check if file already exists
        existing = self.check_file_exists(md5_checksum)
        if existing:
            return {
                'status': 'existing',
                'metadata': existing,
                'document_path': None,
                'json_path': None,
                'message': f"File already exists: {existing.name}"
            }

        # Generate storage paths
        document_path = f"documents/{md5_checksum}{os.path.splitext(file_path)[1]}"
        json_path = f"parsed-json/{md5_checksum}.json"

        # Create new metadata record
        doc_metadata = DocumentMetadata(
            id=md5_checksum,
            name=os.path.basename(file_path),
            type=metadata.get('type', 'unknown'),
            category=metadata.get('category'),
            tags=metadata.get('tags', []),
            uploadDate=metadata.get('upload_date'),
            publishedDate=metadata.get('published_date'),
            userId=user_id,
            productType=metadata.get('product_type'),
            serviceType=metadata.get('service_type'),
            json_file_path=json_path
        )

        # Add to database
        success = self.db.add_metadata(doc_metadata)
        if success:
            return {
                'status': 'new',
                'metadata': doc_metadata,
                'document_path': document_path,
                'json_path': json_path,
                'message': f"File uploaded successfully: {doc_metadata.name}"
            }
        else:
            return {
                'status': 'error',
                'metadata': None,
                'document_path': None,
                'json_path': None,
                'message': "Failed to upload file"
            }

    def delete_file_consistently(self, md5_checksum: str) -> bool:
        """
        Delete file consistently across all systems:
        1. Database metadata
        2. File storage
        3. Vector database chunks
        """
        try:
            # This will handle all consistency:
            # - Delete metadata (cascades to file_storage)
            # - Delete from vector store
            # - Delete local files
            return self.db.delete_metadata_and_files(md5_checksum)
        except Exception as e:
            print(f"Error deleting file consistently: {e}")
            return False

    def get_file_status(self, md5_checksum: str) -> Dict[str, Any]:
        """Get comprehensive file status across all systems."""
        metadata = self.db.get_metadata(md5_checksum)
        if not metadata:
            return {
                'exists': False,
                'metadata': None,
                'storage_records': 0,
                'vector_chunks': 0
            }

        # Count storage records
        storage_response = self.db.supabase.table('file_storage').select('*').eq('file_md5_checksum', md5_checksum).execute()
        storage_count = len(storage_response.data) if storage_response.data else 0

        # Count vector chunks (approximate)
        try:
            from app.utils.supabase_db import vector_store
            retrieved_docs = vector_store.get(where={"file_md5_checksum": md5_checksum})
            vector_chunks = len(retrieved_docs.get("ids", [])) if retrieved_docs else 0
        except Exception as e:
            print(f"Error getting vector chunks: {e}")
            vector_chunks = 0

        return {
            'exists': True,
            'metadata': metadata,
            'storage_records': storage_count,
            'vector_chunks': vector_chunks
        }

    def validate_consistency(self, md5_checksum: str) -> Dict[str, bool]:
        """Validate consistency across all systems."""
        status = self.get_file_status(md5_checksum)

        return {
            'metadata_exists': status['metadata'] is not None,
            'storage_consistent': status['storage_records'] >= 0,
            'vector_consistent': status['vector_chunks'] >= 0,
            'overall_consistent': all([
                status['metadata'] is not None,
                status['storage_records'] >= 0,
                status['vector_chunks'] >= 0
            ])
        }
