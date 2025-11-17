"""
Singleton Supabase client for efficient initialization across the application.
"""

import os
from typing import Optional

from supabase import Client, create_client


class SupabaseSingleton:
    """Singleton class for Supabase client initialization."""

    _instance: Optional['SupabaseSingleton'] = None
    _client: Optional[Client] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseSingleton, cls).__new__(cls)
        return cls._instance

    def get_client(self) -> Client:
        """Get or create the Supabase client."""
        if self._client is None:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")

            if not supabase_url or not supabase_key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

            self._client = create_client(supabase_url, supabase_key)

        return self._client

    def reset_client(self):
        """Reset the client (useful for testing)."""
        self._client = None


# Global instance
supabase_singleton = SupabaseSingleton()


def get_supabase_client() -> Client:
    """Convenience function to get the Supabase client."""
    return supabase_singleton.get_client()


def init_supabase():
    """Initialize Supabase client and verify connection."""
    try:
        client = get_supabase_client()
        # Test connection
        client.auth.get_user()
        print("✅ Supabase client initialized successfully")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize Supabase client: {e}")
        raise
