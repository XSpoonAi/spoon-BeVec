"""Test base vector client protocol."""

import pytest
from typing import Any, Dict, List

from bevec.core.base import VectorClient

class TestVectorClient(VectorClient):
    """Test implementation of VectorClient protocol."""
    
    def __init__(self):
        """Initialize test client."""
        self.vectors: List[Dict[str, Any]] = []
    
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Test upsert implementation."""
        self.vectors.extend(vectors)
    
    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Test query implementation."""
        return self.vectors[:top_k]

def test_vector_client_protocol():
    """Test VectorClient protocol implementation."""
    client = TestVectorClient()
    
    # Test data
    test_vectors = [
        {
            "id": "1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"text": "test1"}
        },
        {
            "id": "2",
            "values": [0.4, 0.5, 0.6],
            "metadata": {"text": "test2"}
        }
    ]
    
    # Test upsert
    client.upsert(test_vectors)
    assert len(client.vectors) == 2
    assert client.vectors[0]["id"] == "1"
    assert client.vectors[1]["id"] == "2"
    
    # Test query
    query_vector = [0.1, 0.2, 0.3]
    results = client.query(query_vector, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "1"
    
    # Test query with larger top_k
    results = client.query(query_vector, top_k=10)
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "2"
    
    # Test query with smaller top_k
    results = client.query(query_vector, top_k=0)
    assert len(results) == 0 