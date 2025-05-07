"""Test vector database clients."""

import os
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from bevec.providers.pinecone import Pinecone
from bevec.providers.chroma import Chroma

# Test data
test_vectors = [
    {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"text": "test1"}},
    {"id": "2", "values": [0.4, 0.5, 0.6], "metadata": {"text": "test2"}}
]

test_query_vector = [0.1, 0.2, 0.3]

def test_pinecone_client():
    """Test Pinecone client functionality."""
    # Mock Pinecone client
    mock_index = MagicMock()
    mock_index.upsert.return_value = None
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "1",
                "score": 0.9,
                "metadata": {"text": "test1"}
            }
        ]
    }

    # Create a mock for list_indexes response
    mock_list_indexes_response = MagicMock()
    mock_list_indexes_response.names = lambda: ["test-index"]

    # Create the main Pinecone mock
    mock_pinecone = MagicMock()
    mock_pinecone.Index.return_value = mock_index
    mock_pinecone.list_indexes.return_value = mock_list_indexes_response
    mock_pinecone.create_index.return_value = None
    mock_pinecone.delete_index.return_value = None

    with patch("bevec.providers.pinecone.adapter._Pinecone", return_value=mock_pinecone):
        # Initialize client
        client = Pinecone.init(api_key="test-key")

        # Test list indexes
        assert client.list_indexes() == ["test-index"]

        # Test create index
        client.create_index("new-index", dimension=3, metric="cosine")
        mock_pinecone.create_index.assert_called_once_with(
            name="new-index",
            dimension=3,
            metric="cosine"
        )

        # Test delete index
        client.delete_index("test-index")
        mock_pinecone.delete_index.assert_called_once_with(name="test-index")

        # Test upsert vectors
        index = client.Index("test-index")
        index.upsert(test_vectors)
        mock_index.upsert.assert_called_once_with(vectors=[
            ("1", [0.1, 0.2, 0.3], {"text": "test1"}),
            ("2", [0.4, 0.5, 0.6], {"text": "test2"})
        ])

        # Test query
        results = index.query(test_query_vector, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.9
        assert results[0]["metadata"] == {"text": "test1"}

@pytest.fixture
def chroma_client():
    """Fixture to create and clean up a Chroma client."""
    # Create a temporary directory for testing
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp()
    os.environ["CHROMA_PERSIST_DIRECTORY"] = test_dir
    
    # Mock Chroma client
    mock_collection = MagicMock()
    mock_collection.upsert.return_value = None
    mock_collection.query.return_value = {
        "ids": [["1"]],
        "metadatas": [[{"text": "test1"}]],
        "distances": [[0.1]]
    }

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = ValueError("Collection not found")
    mock_client.create_collection.return_value = mock_collection
    mock_client.delete_collection.return_value = None

    with patch("chromadb.PersistentClient", return_value=mock_client):
        client = Chroma.init()
        yield client
    
    # Clean up
    shutil.rmtree(test_dir)

def test_chroma_client(chroma_client):
    """Test Chroma client functionality."""
    # Test collection operations
    collection = chroma_client.get_or_create_collection("test-collection")

    # Test upsert vectors
    collection.upsert(test_vectors)
    
    # Test query
    results = collection.query(test_query_vector, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "1"
    assert "score" in results[0]
    assert "metadata" in results[0]

    # Test delete collection
    chroma_client.delete_collection("test-collection")

def test_vector_operations():
    """Test vector operations."""
    # Test vector format
    for vector in test_vectors:
        assert "id" in vector
        assert "values" in vector
        assert "metadata" in vector
        assert len(vector["values"]) == 3

    # Test similarity calculation
    v1 = np.array(test_vectors[0]["values"])
    v2 = np.array(test_vectors[1]["values"])
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    assert 0 <= similarity <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 