"""
Vectro+ Python API

High-performance vector compression and search toolkit with Python bindings.

This module provides Python access to the high-performance Rust implementation
of Vectro+, enabling efficient vector compression, quantization, and search
operations.

Example:
    >>> import vectro_plus as vp
    >>> import numpy as np
    >>> 
    >>> # Create sample embeddings
    >>> vectors = np.random.randn(1000, 128).astype(np.float32)
    >>> ids = [f"vec_{i}" for i in range(1000)]
    >>> 
    >>> # Compress and create search indices
    >>> search_index, quantized_index = vp.compress_embeddings(vectors, ids)
    >>> 
    >>> # Search for similar vectors
    >>> query = np.random.randn(128).astype(np.float32)
    >>> indices, similarities = search_index.search_vector(query, top_k=10)
    >>> 
    >>> # Analyze compression quality
    >>> quality = vp.analyze_compression_quality(vectors, quantized_index)
    >>> print(f"Compression ratio: {quality['compression_ratio']:.2f}x")
    >>> print(f"Average similarity: {quality['average_similarity']:.4f}")
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# Import the Rust extension
try:
    from .vectro_py import (
        PyEmbedding as Embedding,
        PyEmbeddingDataset as EmbeddingDataset,
        PySearchIndex as SearchIndex,
        PyQuantizedIndex as QuantizedIndex,
        compress_embeddings,
        analyze_compression_quality,
        benchmark_search_performance,
        __version__,
        __author__,
        __description__,
    )
    _rust_available = True
except ImportError as e:
    print(f"Warning: Rust extension not available: {e}")
    print("Falling back to Python-only implementation")
    _rust_available = False
    __version__ = "1.5.0"
    __author__ = "Wesley Scholl"
    __description__ = "Python bindings for Vectro+ (fallback mode)"

# Re-export main classes and functions
__all__ = [
    # Core classes
    "Embedding",
    "EmbeddingDataset", 
    "SearchIndex",
    "QuantizedIndex",
    # Main functions
    "compress_embeddings",
    "create_index",
    "create_quantized_index",
    "search_similar",
    "batch_search",
    # Analysis functions
    "analyze_compression_quality",
    "benchmark_search_performance",
    "generate_quality_report",
    # Utility functions
    "load_embeddings_from_array",
    "save_index",
    "load_index",
    # Package info
    "__version__",
    "__author__",
    "__description__",
]


class VectroConfig:
    """Configuration class for Vectro+ operations."""
    
    def __init__(self, 
                 compression_method: str = "quantized",
                 quantization_bits: int = 8,
                 search_threads: Optional[int] = None,
                 memory_map: bool = True):
        """
        Initialize Vectro+ configuration.
        
        Args:
            compression_method: Either "quantized" or "regular"
            quantization_bits: Number of bits for quantization (8 or 16)
            search_threads: Number of threads for search operations (None for auto)
            memory_map: Whether to use memory mapping for large datasets
        """
        self.compression_method = compression_method
        self.quantization_bits = quantization_bits
        self.search_threads = search_threads
        self.memory_map = memory_map


def create_index(vectors: np.ndarray, 
                ids: Optional[List[str]] = None,
                config: Optional[VectroConfig] = None) -> SearchIndex:
    """
    Create a search index from vectors.
    
    Args:
        vectors: Array of shape (n_vectors, n_dimensions) 
        ids: Optional list of string IDs for vectors
        config: Optional configuration object
        
    Returns:
        SearchIndex object for performing searches
        
    Example:
        >>> vectors = np.random.randn(1000, 128).astype(np.float32)
        >>> index = create_index(vectors)
        >>> query = np.random.randn(128).astype(np.float32)
        >>> indices, similarities = index.search_vector(query, top_k=5)
    """
    if not _rust_available:
        raise RuntimeError("Rust extension not available. Please install properly.")
        
    config = config or VectroConfig()
    
    # Create dataset
    dataset = EmbeddingDataset()
    
    for i, vector in enumerate(vectors):
        vector_id = ids[i] if ids else f"vec_{i}"
        dataset.add_vector(vector_id, vector.astype(np.float32))
    
    # Create index
    return SearchIndex.from_dataset(dataset)


def create_quantized_index(vectors: np.ndarray,
                          ids: Optional[List[str]] = None,
                          config: Optional[VectroConfig] = None) -> QuantizedIndex:
    """
    Create a quantized search index from vectors for space efficiency.
    
    Args:
        vectors: Array of shape (n_vectors, n_dimensions)
        ids: Optional list of string IDs for vectors  
        config: Optional configuration object
        
    Returns:
        QuantizedIndex object for performing compressed searches
        
    Example:
        >>> vectors = np.random.randn(1000, 128).astype(np.float32) 
        >>> index = create_quantized_index(vectors)
        >>> print(f"Compression ratio: {index.compression_ratio():.2f}x")
        >>> query = np.random.randn(128).astype(np.float32)
        >>> indices, similarities = index.search_vector(query, top_k=5)
    """
    if not _rust_available:
        raise RuntimeError("Rust extension not available. Please install properly.")
        
    config = config or VectroConfig()
    
    # Create dataset
    dataset = EmbeddingDataset()
    
    for i, vector in enumerate(vectors):
        vector_id = ids[i] if ids else f"vec_{i}"
        dataset.add_vector(vector_id, vector.astype(np.float32))
    
    # Create quantized index
    return QuantizedIndex.from_dataset(dataset)


def search_similar(index: Union[SearchIndex, QuantizedIndex],
                  query: np.ndarray,
                  top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for similar vectors using an index.
    
    Args:
        index: SearchIndex or QuantizedIndex to search
        query: Query vector of shape (n_dimensions,)
        top_k: Number of top results to return
        
    Returns:
        Tuple of (indices, similarities) arrays
        
    Example:
        >>> index = create_index(vectors)
        >>> query = np.random.randn(128).astype(np.float32)
        >>> indices, similarities = search_similar(index, query, top_k=5)
        >>> print(f"Found {len(indices)} results")
        >>> print(f"Best similarity: {similarities[0]:.4f}")
    """
    if not _rust_available:
        raise RuntimeError("Rust extension not available. Please install properly.")
        
    return index.search_vector(query.astype(np.float32), top_k)


def batch_search(index: Union[SearchIndex, QuantizedIndex],
                queries: np.ndarray,
                top_k: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Perform batch search for multiple queries.
    
    Args:
        index: SearchIndex or QuantizedIndex to search
        queries: Array of shape (n_queries, n_dimensions)
        top_k: Number of top results to return per query
        
    Returns:
        List of (indices, similarities) tuples, one per query
        
    Example:
        >>> index = create_index(vectors)
        >>> queries = np.random.randn(10, 128).astype(np.float32)
        >>> results = batch_search(index, queries, top_k=5)
        >>> print(f"Processed {len(results)} queries")
    """
    if not _rust_available:
        raise RuntimeError("Rust extension not available. Please install properly.")
        
    return index.batch_search(queries.astype(np.float32), top_k)


def load_embeddings_from_array(vectors: np.ndarray, 
                              ids: Optional[List[str]] = None) -> EmbeddingDataset:
    """
    Load embeddings from a numpy array into an EmbeddingDataset.
    
    Args:
        vectors: Array of shape (n_vectors, n_dimensions)
        ids: Optional list of string IDs
        
    Returns:
        EmbeddingDataset containing the vectors
        
    Example:
        >>> vectors = np.random.randn(100, 64).astype(np.float32)
        >>> dataset = load_embeddings_from_array(vectors)
        >>> print(f"Loaded {len(dataset)} vectors")
    """
    if not _rust_available:
        raise RuntimeError("Rust extension not available. Please install properly.")
        
    dataset = EmbeddingDataset()
    
    for i, vector in enumerate(vectors):
        vector_id = ids[i] if ids else f"vec_{i}"
        dataset.add_vector(vector_id, vector.astype(np.float32))
    
    return dataset


def generate_quality_report(vectors: np.ndarray,
                          quantized_index: QuantizedIndex,
                          num_samples: int = 1000) -> Dict[str, Any]:
    """
    Generate a comprehensive quality report for compressed vectors.
    
    Args:
        vectors: Original vectors array
        quantized_index: QuantizedIndex to analyze
        num_samples: Number of samples to use for analysis
        
    Returns:
        Dictionary containing quality metrics and analysis
        
    Example:
        >>> vectors = np.random.randn(1000, 128).astype(np.float32)
        >>> index = create_quantized_index(vectors)
        >>> report = generate_quality_report(vectors, index)
        >>> print(f"Quality grade: {report['quality_grade']}")
        >>> print(f"Space savings: {report['memory_savings_percent']:.1f}%")
    """
    if not _rust_available:
        raise RuntimeError("Rust extension not available. Please install properly.")
        
    # Get basic quality metrics
    quality_metrics = analyze_compression_quality(vectors, quantized_index, num_samples)
    
    # Add quality grade based on similarity
    avg_sim = quality_metrics["average_similarity"]
    if avg_sim >= 0.99:
        quality_grade = "A+"
    elif avg_sim >= 0.95:
        quality_grade = "A"
    elif avg_sim >= 0.90:
        quality_grade = "B"
    elif avg_sim >= 0.85:
        quality_grade = "C"
    else:
        quality_grade = "D"
    
    # Enhanced report
    report = {
        **quality_metrics,
        "quality_grade": quality_grade,
        "recommendation": _get_quality_recommendation(avg_sim, quality_metrics["compression_ratio"]),
        "memory_usage_mb": quantized_index.memory_usage_bytes() / (1024 * 1024),
        "original_size_estimate_mb": (vectors.nbytes) / (1024 * 1024),
    }
    
    return report


def _get_quality_recommendation(similarity: float, compression_ratio: float) -> str:
    """Get recommendation based on quality metrics."""
    if similarity >= 0.99 and compression_ratio >= 3.0:
        return "Excellent compression with minimal quality loss. Recommended for production."
    elif similarity >= 0.95 and compression_ratio >= 2.0:
        return "Good compression ratio with acceptable quality. Consider for most applications."
    elif similarity >= 0.90:
        return "Moderate compression. May be suitable for applications tolerant to some quality loss."
    else:
        return "Low compression quality. Consider adjusting quantization parameters."


# Convenience functions for backward compatibility
def save_index(index: Union[SearchIndex, QuantizedIndex], filepath: str) -> None:
    """Save an index to disk (placeholder - to be implemented)."""
    raise NotImplementedError("Index serialization not yet implemented")


def load_index(filepath: str) -> Union[SearchIndex, QuantizedIndex]:
    """Load an index from disk (placeholder - to be implemented)."""
    raise NotImplementedError("Index deserialization not yet implemented")


# Package information
def info() -> Dict[str, str]:
    """Get package information."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "rust_available": str(_rust_available),
    }


def version() -> str:
    """Get package version."""
    return __version__