#!/usr/bin/env python3
"""
Simple script to run the Chroma to PostgreSQL migration
"""
import asyncio
import os
import sys

# Add parent directory to path to import migration module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.migrate_chroma_to_postgres import main

if __name__ == "__main__":
    print("=" * 60)
    print("Chroma to PostgreSQL Migration Script")
    print("=" * 60)
    print()
    print("This script will migrate all data from Chroma SQLite to PostgreSQL.")
    print("Make sure PostgreSQL is running and accessible.")
    print()

    response = input("Do you want to continue? (y/N): ")
    if response.lower() != "y":
        print("Migration cancelled.")
        sys.exit(0)

    print()
    print("Starting migration...")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMigration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nMigration failed with error: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)
