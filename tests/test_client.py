"""Test vector client protocol."""

import pytest
import numpy as np
from typing import Any, Dict, List, Optional

from bevec.client import VectorClient

class MockVectorClient(VectorClient):
    """Mock implementation of VectorClient for testing."""
    
    def __init__(self):
        """Initialize mock client."""
        self.vectors: Dict[str, List[Dict[str, Any]]] = {}
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Mock upsert implementation."""
        if namespace not in self.vectors:
            self.vectors[namespace] = []
        self.vectors[namespace].extend(vectors)
    
    def query(
        self,
        vector: List[float],
        namespace: Optional[str] = None,
        top_k: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Mock query implementation."""
        if namespace not in self.vectors:
            return []
        
        # Simple cosine similarity calculation
        results = []
        for vec in self.vectors[namespace]:
            similarity = np.dot(vector, vec["values"]) / (
                np.linalg.norm(vector) * np.linalg.norm(vec["values"])
            )
            results.append({
                "id": vec["id"],
                "score": float(similarity),
                "metadata": vec["metadata"]
            })
        
        # Sort by similarity and return top_k results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

def test_vector_client_protocol():
    """Test VectorClient protocol implementation."""
    client = MockVectorClient()
    
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
    client.upsert(test_vectors, namespace="test")
    assert len(client.vectors["test"]) == 2
    assert client.vectors["test"][0]["id"] == "1"
    assert client.vectors["test"][1]["id"] == "2"
    
    # Test query
    query_vector = [0.1, 0.2, 0.3]
    results = client.query(query_vector, namespace="test", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert results[0]["score"] > 0.9  # Should be very similar
    
    # Test empty namespace
    results = client.query(query_vector, namespace="nonexistent")
    assert len(results) == 0
    
    # Test with numpy array
    query_vector_np = np.array([0.1, 0.2, 0.3])
    results = client.query(query_vector_np, namespace="test", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "1" 