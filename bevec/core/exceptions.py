"""Custom exceptions for BeVec library."""

class BeVecError(Exception):
    """Base exception for BeVec library."""
    pass

class ProviderError(BeVecError):
    """Exception raised for provider-related errors."""
    pass

class ConfigurationError(BeVecError):
    """Exception raised for configuration-related errors."""
    pass

class VectorOperationError(BeVecError):
    """Exception raised for vector operation errors."""
    pass

class ValidationError(BeVecError):
    """Exception raised for validation errors."""
    pass 