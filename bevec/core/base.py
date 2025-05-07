"""Base classes for vector database clients."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol

class VectorClient(Protocol):
    """Protocol defining vector database client interface."""
    
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors into the database.
        
        Args:
            vectors: List of dictionaries containing vector data
                Each vector should have: id, values, and metadata
        """
        ...
    
    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Query similar vectors from the database.
        
        Args:
            vector: Query vector
            top_k: Number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing query results
            Each result has: id, score, and metadata
        """
        ... 