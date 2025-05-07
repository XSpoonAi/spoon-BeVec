"""Pinecone configuration management."""

import os
from typing import Optional

def get_api_key() -> Optional[str]:
    """Get Pinecone API key from environment.
    
    Returns:
        API key if set in environment, None otherwise
    """
    return os.getenv("PINECONE_API_KEY")

def get_environment() -> Optional[str]:
    """Get Pinecone environment from environment.
    
    Returns:
        Environment if set in environment, None otherwise
    """
    return os.getenv("PINECONE_ENVIRONMENT") 