"""Chroma configuration utilities."""

import os
from typing import Optional

def get_persist_directory() -> str:
    """Get the directory for persisting Chroma database.
    
    Returns:
        str: Directory path for Chroma persistence
    """
    return os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db") 