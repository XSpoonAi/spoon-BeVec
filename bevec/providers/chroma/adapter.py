"""Chroma vector database adapter."""

from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings

from ...core.base import VectorClient
from ...core.exceptions import (
    ConfigurationError,
    ProviderError,
    ValidationError,
    VectorOperationError
)
from ...core.registry import register_provider
from .config import get_persist_directory

class ChromaCollection(VectorClient):
    """Chroma collection adapter implementing VectorClient protocol."""
    
    def __init__(self, collection: chromadb.Collection):
        """Initialize Chroma collection adapter.
        
        Args:
            collection: Chroma collection instance
        """
        self._collection = collection
    
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors into Chroma.
        
        Args:
            vectors: List of dictionaries containing vector data
                Each vector should have: id, values, and metadata
                
        Raises:
            ValidationError: If vectors are not properly formatted
            VectorOperationError: If upsert operation fails
        """
        if not vectors:
            raise ValidationError("Vectors list cannot be empty")
        
        ids = []
        embeddings = []
        metadatas = []
        
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
                
                ids.append(vector["id"])
                embeddings.append(vector["values"])
                metadatas.append(vector["metadata"])
            except Exception as e:
                raise ValidationError(f"Error formatting vector at index {i}: {str(e)}")
            
        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            raise VectorOperationError(f"Failed to upsert vectors: {str(e)}")
    
    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Query similar vectors from Chroma.
        
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
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["metadatas", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity score
                })
                
            return formatted_results
        except Exception as e:
            raise VectorOperationError(f"Failed to query vectors: {str(e)}")

class ChromaClient:
    """Chroma client with native SDK compatibility."""
    
    def __init__(self, client: chromadb.Client):
        """Initialize Chroma client.
        
        Args:
            client: Chroma persistent client instance
        """
        self._client = client
    
    def get_or_create_collection(self, name: str) -> ChromaCollection:
        """Get or create a collection.
        
        Args:
            name: Name of the collection
            
        Returns:
            ChromaCollection: A collection instance
            
        Raises:
            ValidationError: If collection name is invalid
            ProviderError: If collection operation fails
        """
        if not name:
            raise ValidationError("Collection name cannot be empty")
        
        try:
            try:
                collection = self._client.get_collection(name=name)
            except ValueError:
                # Collection doesn't exist, create it
                collection = self._client.create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity by default
                )
            return ChromaCollection(collection)
        except Exception as e:
            raise ProviderError(f"Failed to get or create collection: {str(e)}")
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection.
        
        Args:
            name: Name of the collection to delete
            
        Raises:
            ValidationError: If collection name is invalid
            ProviderError: If collection deletion fails
        """
        if not name:
            raise ValidationError("Collection name cannot be empty")
        
        try:
            try:
                self._client.delete_collection(name=name)
            except ValueError:
                # Collection doesn't exist, ignore
                pass
        except Exception as e:
            raise ProviderError(f"Failed to delete collection: {str(e)}")

@register_provider("chroma")
class Chroma:
    """Chroma vector database client with native SDK compatibility."""
    
    @classmethod
    def init(
        cls,
        persist_directory: Optional[str] = None,
        **kwargs: Any
    ) -> ChromaClient:
        """Initialize Chroma client (compatible with original SDK).
        
        Args:
            persist_directory: Directory to persist the database (defaults to CHROMA_PERSIST_DIRECTORY env var)
            **kwargs: Additional Chroma initialization arguments
            
        Returns:
            Chroma client instance
            
        Raises:
            ConfigurationError: If client initialization fails
        """
        try:
            persist_directory = persist_directory or get_persist_directory()
            if not persist_directory:
                raise ConfigurationError("Persist directory not found")
            
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True  # Allow resetting for testing
                )
            )
            return ChromaClient(client)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Chroma client: {str(e)}")
    
    @classmethod
    def PersistentClient(
        cls,
        path: Optional[str] = None,
        **kwargs: Any
    ) -> ChromaClient:
        """Get Chroma persistent client (compatible with original SDK).
        
        Args:
            path: Directory to persist the database (defaults to CHROMA_PERSIST_DIRECTORY env var)
            **kwargs: Additional Chroma initialization arguments
            
        Returns:
            Chroma client instance
            
        Raises:
            ConfigurationError: If client initialization fails
        """
        try:
            path = path or get_persist_directory()
            if not path:
                raise ConfigurationError("Persist directory not found")
            
            client = chromadb.PersistentClient(
                path=path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True  # Allow resetting for testing
                )
            )
            return ChromaClient(client)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Chroma client: {str(e)}") 