"""Provider registry for vector database clients."""

from typing import Any, Callable, Dict, Type, List

from .base import VectorClient

class ProviderRegistry:
    """Registry for vector database providers."""
    
    def __init__(self):
        """Initialize empty registry."""
        self._providers: Dict[str, Callable[..., Any]] = {}
    
    def register(self, name: str) -> Callable[[Type], Type]:
        """Register a provider class.
        
        Args:
            name: Provider name
            
        Returns:
            Decorator function
        """
        def decorator(cls: Type) -> Type:
            self._providers[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Callable[..., Any]:
        """Get provider class by name.
        
        Args:
            name: Provider name
            
        Returns:
            Provider class
            
        Raises:
            KeyError: If provider not found
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found")
        return self._providers[name]

# Global registry instance
registry = ProviderRegistry()

def register_provider(name: str) -> Callable[[Type], Type]:
    """Register a provider class with the global registry.
    
    Args:
        name: Provider name
        
    Returns:
        Decorator function
    """
    return registry.register(name)

def get_provider(name: str) -> Type[VectorClient]:
    """Get a registered provider class.
    
    Args:
        name: Name of the provider
        
    Returns:
        Provider class
        
    Raises:
        ValueError: If provider is not registered
    """
    name = name.lower()
    if name not in registry._providers:
        raise ValueError(f"Unsupported provider: {name}")
    return registry._providers[name]

def list_providers() -> List[str]:
    """List all registered providers.
    
    Returns:
        List of provider names
    """
    return list(registry._providers.keys()) 