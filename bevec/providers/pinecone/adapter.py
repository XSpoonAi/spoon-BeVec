"""Pinecone vector database adapter."""

from typing import Any, Dict, List, Optional

from pinecone import Pinecone as _Pinecone

from ...core.base import VectorClient
from ...core.exceptions import (
    ConfigurationError,
    ProviderError,
    ValidationError,
    VectorOperationError
)
from ...core.registry import register_provider
from .config import get_api_key

class PineconeIndex(VectorClient):
    """Pinecone index adapter implementing VectorClient protocol."""
    
    def __init__(self, index: Any):
        """Initialize Pinecone index adapter.
        
        Args:
            index: Pinecone index instance
        """
        self._index = index

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors into Pinecone.
        
        Args:
            vectors: List of dictionaries containing vector data
                Each vector should have: id, values, and metadata
                
        Raises:
            ValidationError: If vectors are not properly formatted
            VectorOperationError: If upsert operation fails
        """
        if not vectors:
            raise ValidationError("Vectors list cannot be empty")
        
        formatted_vectors = []
        for i, vector in enumerate(vectors):
            try:
                if not isinstance(vector, dict):
                    raise ValidationError(f"Vector at index {i} must be a dictionary")
                
                if "id" not in vector:
                    raise ValidationError(f"Vector at index {i} missing 'id' field")
                if "values" not in vector:
                    raise ValidationError(f"Vector at index {i} missing 'values' field")
                if "metadata" not in vector:
                    raise ValidationError(f"Vector at index {i} missing 'metadata' field")
                
                if not isinstance(vector["values"], list):
                    raise ValidationError(f"Vector values at index {i} must be a list")
                
                formatted_vectors.append(
                    (vector["id"], vector["values"], vector["metadata"])
                )
            except Exception as e:
                raise ValidationError(f"Error formatting vector at index {i}: {str(e)}")
        
        try:
            self._index.upsert(vectors=formatted_vectors)
        except Exception as e:
            raise VectorOperationError(f"Failed to upsert vectors: {str(e)}")

    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Query similar vectors from Pinecone.
        
        Args:
            vector: Query vector
            top_k: Number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing query results
            Each result has: id, score, and metadata
            
        Raises:
            ValidationError: If query parameters are invalid
            VectorOperationError: If query operation fails
        """
        if not isinstance(vector, list):
            raise ValidationError("Query vector must be a list")
        
        if not vector:
            raise ValidationError("Query vector cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in vector):
            raise ValidationError("Query vector must contain only numbers")
        
        if top_k < 1:
            raise ValidationError("top_k must be greater than 0")
        
        try:
            results = self._index.query(vector=vector, top_k=top_k)
            return results["matches"]
        except Exception as e:
            raise VectorOperationError(f"Failed to query vectors: {str(e)}")

@register_provider("pinecone")
class Pinecone:
    """Pinecone vector database client with native SDK compatibility."""
    
    @classmethod
    def init(cls, api_key: Optional[str] = None) -> "Pinecone":
        """Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            
        Returns:
            Pinecone client instance
            
        Raises:
            ConfigurationError: If API key is not provided or invalid
        """
        try:
            api_key = api_key or get_api_key()
            if not api_key:
                raise ConfigurationError("Pinecone API key not found")
            return cls(api_key=api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Pinecone client: {str(e)}")
    
    def __init__(self, api_key: str):
        """Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            
        Raises:
            ConfigurationError: If client initialization fails
        """
        try:
            self._client = _Pinecone(api_key=api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to create Pinecone client: {str(e)}")

    def list_indexes(self) -> List[str]:
        """List all available indexes.
        
        Returns:
            List of index names
            
        Raises:
            ProviderError: If listing indexes fails
        """
        try:
            return self._client.list_indexes().names()
        except Exception as e:
            raise ProviderError(f"Failed to list indexes: {str(e)}")

    def create_index(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
    ) -> None:
        """Create a new index.
        
        Args:
            name: Name of the index
            dimension: Dimension of vectors
            metric: Distance metric (default: "cosine")
            
        Raises:
            ValidationError: If parameters are invalid
            ProviderError: If index creation fails
        """
        if not name:
            raise ValidationError("Index name cannot be empty")
        
        if dimension < 1:
            raise ValidationError("Dimension must be greater than 0")
        
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        if metric not in valid_metrics:
            raise ValidationError(f"Metric must be one of: {', '.join(valid_metrics)}")
        
        try:
            self._client.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
            )
        except Exception as e:
            raise ProviderError(f"Failed to create index: {str(e)}")

    def delete_index(self, name: str) -> None:
        """Delete an index.
        
        Args:
            name: Name of the index to delete
            
        Raises:
            ValidationError: If index name is invalid
            ProviderError: If index deletion fails
        """
        if not name:
            raise ValidationError("Index name cannot be empty")
        
        try:
            self._client.delete_index(name=name)
        except Exception as e:
            raise ProviderError(f"Failed to delete index: {str(e)}")

    def Index(self, name: str) -> PineconeIndex:
        """Get an index instance.
        
        Args:
            name: Name of the index
            
        Returns:
            PineconeIndex: An index instance
            
        Raises:
            ValidationError: If index name is invalid
            ProviderError: If getting index fails
        """
        if not name:
            raise ValidationError("Index name cannot be empty")
        
        try:
            return PineconeIndex(self._client.Index(name))
        except Exception as e:
            raise ProviderError(f"Failed to get index: {str(e)}") 