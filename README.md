# BeVec

BeVec is a unified vector database client library that supports Pinecone and Chroma databases. It provides a consistent interface while maintaining compatibility with the original SDKs.

## Features

- Unified vector operation interface
- Full compatibility with original SDKs
- Type safety and comprehensive input validation
- Simple and intuitive API
- Support for Pinecone and Chroma databases
- Comprehensive error handling with custom exceptions
- Detailed logging and debugging support
- High test coverage with mock implementations
- Consistent behavior across different providers

## Installation

```bash
pip install bevec
```

## Quick Start

### Pinecone

```python
from bevec import Pinecone

# Initialize client (using environment variable)
pc = Pinecone.init()  # Uses PINECONE_API_KEY environment variable

# Or initialize with explicit API key
pc = Pinecone.init(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine"  # Available metrics: cosine, euclidean, dotproduct
)

# Get index
index = pc.Index("my-index")

# Insert vectors
index.upsert(vectors=[
    {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"text": "hello"}},
    {"id": "2", "values": [0.4, 0.5, 0.6], "metadata": {"text": "world"}},
])

# Query vectors
results = index.query(
    vector=[0.1, 0.2, 0.3],
    top_k=2
)

# List all indexes
indexes = pc.list_indexes()

# Delete index
pc.delete_index("my-index")
```

### Chroma

```python
from bevec import Chroma

# Initialize client (using environment variable)
client = Chroma.init()  # Uses CHROMA_PERSIST_DIRECTORY environment variable

# Or initialize with explicit path
client = Chroma.init(persist_directory="./chroma_db")

# Get collection
collection = client.get_or_create_collection("my_collection")

# Insert vectors
collection.upsert(vectors=[
    {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"text": "hello"}},
    {"id": "2", "values": [0.4, 0.5, 0.6], "metadata": {"text": "world"}},
])

# Query vectors
results = collection.query(
    vector=[0.1, 0.2, 0.3],
    top_k=2
)

# Delete collection
client.delete_collection("my_collection")
```

## Error Handling

BeVec provides a comprehensive error handling system with custom exceptions:

### Exception Hierarchy

- `BeVecError`: Base exception for all BeVec errors
  - `ProviderError`: Provider-related errors (e.g., API errors)
  - `ConfigurationError`: Configuration issues (e.g., missing API keys)
  - `VectorOperationError`: Vector operation failures
  - `ValidationError`: Input validation errors

### Example Usage

```python
from bevec import Pinecone
from bevec.core.exceptions import ValidationError, ConfigurationError

try:
    pc = Pinecone.init()
    index = pc.Index("my-index")
    
    # This will raise ValidationError if vectors are not properly formatted
    index.upsert(vectors=[
        {"id": "1", "values": [0.1, 0.2, 0.3]}  # Missing metadata
    ])
except ValidationError as e:
    print(f"Invalid input: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## API Documentation

### Pinecone Client

#### `Pinecone(api_key: str)`
Initialize a Pinecone client.

**Parameters:**
- `api_key` (str): Your Pinecone API key

**Returns:**
- `Pinecone`: A Pinecone client instance

#### `Pinecone.list_indexes() -> List[str]`
List all available indexes.

**Returns:**
- `List[str]`: List of index names

#### `Pinecone.create_index(name: str, dimension: int, metric: str = "cosine") -> None`
Create a new index.

**Parameters:**
- `name` (str): Name of the index
- `dimension` (int): Dimension of vectors
- `metric` (str): Distance metric (default: "cosine")

#### `Pinecone.delete_index(name: str) -> None`
Delete an index.

**Parameters:**
- `name` (str): Name of the index to delete

#### `Pinecone.Index(name: str) -> PineconeIndex`
Get an index instance.

**Parameters:**
- `name` (str): Name of the index

**Returns:**
- `PineconeIndex`: An index instance

### Chroma Client

#### `Chroma.PersistentClient(path: str) -> ChromaClient`
Initialize a persistent Chroma client.

**Parameters:**
- `path` (str): Path to persist the database

**Returns:**
- `ChromaClient`: A Chroma client instance

#### `ChromaClient.get_or_create_collection(name: str) -> ChromaCollection`
Get or create a collection.

**Parameters:**
- `name` (str): Name of the collection

**Returns:**
- `ChromaCollection`: A collection instance

#### `ChromaClient.delete_collection(name: str) -> None`
Delete a collection.

**Parameters:**
- `name` (str): Name of the collection to delete

### Vector Operations

Both Pinecone and Chroma collections support the following operations:

#### `upsert(vectors: List[Dict[str, Any]]) -> None`
Insert or update vectors.

**Parameters:**
- `vectors` (List[Dict[str, Any]]): List of vectors to insert/update
  - Each vector should have: `id`, `values`, and `metadata`

#### `query(vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]`
Query similar vectors.

**Parameters:**
- `vector` (List[float]): Query vector
- `top_k` (int): Number of results to return (default: 10)

**Returns:**
- `List[Dict[str, Any]]`: List of similar vectors with scores and metadata

## Environment Variables

### Pinecone
- `PINECONE_API_KEY`: Your Pinecone API key

### Chroma
- `CHROMA_PERSIST_DIRECTORY`: Directory to persist Chroma database

## Development

### Install Development Dependencies

```bash
pip install -e ".[test]"
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=bevec tests/

# Run specific test file
pytest tests/test_client.py
```

### Test Components

BeVec includes comprehensive test suites:

- `test_client.py`: Tests for VectorClient protocol
- `test_base.py`: Tests for base VectorClient implementation
- `test_registry.py`: Tests for provider registry system
- Provider-specific tests for Pinecone and Chroma adapters

### Mock Implementations

The test suite includes mock implementations that can be used for testing your applications:

```python
from bevec.client import VectorClient
from typing import List, Dict, Any

class MockVectorClient(VectorClient):
    def __init__(self):
        self.vectors = {}
    
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        # Implementation for testing
        pass
    
    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        # Implementation for testing
        pass
```

## Best Practices

### Configuration Management

1. Use environment variables for sensitive information:
   - `PINECONE_API_KEY` for Pinecone API key
   - `CHROMA_PERSIST_DIRECTORY` for Chroma database location

2. Initialize clients using the `init()` class method:
   ```python
   # Preferred way
   pc = Pinecone.init()
   chroma = Chroma.init()
   ```

### Error Handling

1. Always wrap vector operations in try-except blocks
2. Handle specific exceptions for better error management
3. Validate inputs before operations
4. Check return values and handle edge cases

### Vector Operations

1. Ensure vector dimensions match index configuration
2. Properly format metadata for better searchability
3. Use appropriate batch sizes for upsert operations
4. Consider using namespaces for better organization

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.