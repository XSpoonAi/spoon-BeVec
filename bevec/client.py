from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np

class VectorClient(Protocol):
    """Protocol defining the interface for vector database clients."""
    
    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Upsert vectors into the database.
        
        Args:
            vectors: List of dictionaries containing vector data
            namespace: Optional namespace/collection name
            **kwargs: Additional provider-specific arguments
        """
        ...
    
    def query(
        self,
        vector: Union[List[float], np.ndarray],
        namespace: Optional[str] = None,
        top_k: int = 10,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Query similar vectors from the database.
        
        Args:
            vector: Query vector
            namespace: Optional namespace/collection name
            top_k: Number of results to return
            **kwargs: Additional provider-specific arguments
            
        Returns:
            List of dictionaries containing query results
        """
        ... 