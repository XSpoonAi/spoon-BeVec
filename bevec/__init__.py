"""BeVec - A unified vector database client library.

This package provides a unified interface for working with different vector
database providers like Pinecone and Chroma.

Example usage:

    # Using Pinecone
    from bevec import Pinecone
    
    # Initialize client
    pc = Pinecone(api_key="your-api-key")
    
    # Get index
    index = pc.Index("your-index")
    
    # Upsert vectors
    index.upsert(vectors=[
        ("id1", [0.1, 0.2, 0.3], {"text": "hello"}),
        ("id2", [0.4, 0.5, 0.6], {"text": "world"}),
    ])
    
    # Query vectors
    results = index.query(
        vector=[0.1, 0.2, 0.3],
        top_k=2,
        include_metadata=True
    )
    
    # Using Chroma
    from bevec import Chroma
    
    # Initialize client
    client = Chroma.PersistentClient(path="./chroma_db")
    
    # Get collection
    collection = client.get_or_create_collection("my_collection")
    
    # Upsert vectors
    collection.upsert(
        ids=["id1", "id2"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadatas=[{"text": "hello"}, {"text": "world"}]
    )
    
    # Query vectors
    results = collection.query(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=2,
        include=["metadatas", "distances"]
    )
"""

from .providers.pinecone.adapter import Pinecone
from .providers.chroma.adapter import Chroma

__version__ = "0.1.0"

__all__ = [
    "Pinecone",
    "Chroma",
] 