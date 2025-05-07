"""Test provider registry."""

import pytest
from typing import Any, Dict, List

from bevec.core.registry import (
    ProviderRegistry,
    register_provider,
    get_provider,
    list_providers,
    registry
)
from bevec.core.base import VectorClient

class TestProvider(VectorClient):
    """Test provider implementation."""
    
    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        """Test upsert implementation."""
        pass
    
    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Test query implementation."""
        return []

@register_provider("test")
class RegisteredProvider(TestProvider):
    """Registered test provider."""
    pass

def test_provider_registry():
    """Test provider registry functionality."""
    # Test registry initialization
    registry = ProviderRegistry()
    assert len(registry._providers) == 0
    
    # Test provider registration
    @registry.register("test1")
    class TestProvider1(TestProvider):
        pass
    
    assert "test1" in registry._providers
    assert registry._providers["test1"] == TestProvider1
    
    # Test provider retrieval
    provider = registry.get("test1")
    assert provider == TestProvider1
    
    # Test provider not found
    with pytest.raises(KeyError):
        registry.get("nonexistent")

def test_global_registry():
    """Test global registry functionality."""
    # Test provider registration
    assert "test" in registry._providers
    assert registry._providers["test"] == RegisteredProvider
    
    # Test provider retrieval
    provider = get_provider("test")
    assert provider == RegisteredProvider
    
    # Test provider not found
    with pytest.raises(ValueError):
        get_provider("nonexistent")
    
    # Test list providers
    providers = list_providers()
    assert "test" in providers
    
    # Test case insensitivity
    provider = get_provider("TEST")
    assert provider == RegisteredProvider

def test_provider_implementation():
    """Test provider implementation."""
    provider = RegisteredProvider()
    
    # Test upsert
    vectors = [
        {"id": "1", "values": [0.1, 0.2, 0.3], "metadata": {"text": "test1"}}
    ]
    provider.upsert(vectors)  # Should not raise any errors
    
    # Test query
    results = provider.query([0.1, 0.2, 0.3])
    assert isinstance(results, list)
    assert len(results) == 0 