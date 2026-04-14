"""
Type stubs for vectro-plus — high-performance vector compression and search.

These stubs cover both the PyO3 Rust extension (vectro_plus.vectro_py) and
the Python convenience wrappers defined in __init__.py.

Any change to method signatures in vectro_py/src/lib.rs or __init__.py
MUST be reflected here in the same commit (per project rules).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# ── Core types (re-exported from the Rust extension) ─────────────────────────

class Embedding:
    """A single embedding: a string ID paired with an f32 vector."""

    def __init__(self, id: str, vector: NDArray[np.float32]) -> None: ...
    @property
    def id(self) -> str: ...
    @property
    def vector(self) -> NDArray[np.float32]: ...
    def __repr__(self) -> str: ...

class EmbeddingDataset:
    """A mutable collection of Embedding objects."""

    def __init__(self) -> None: ...
    def add_embedding(self, embedding: Embedding) -> None: ...
    def add_vector(self, id: str, vector: NDArray[np.float32]) -> None: ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def get_embedding(self, index: int) -> Optional[Embedding]: ...
    def get_vectors(self) -> NDArray[np.float32]: ...
    def get_ids(self) -> List[str]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class SearchIndex:
    """Exact brute-force cosine-similarity search index."""

    @staticmethod
    def from_dataset(dataset: EmbeddingDataset) -> "SearchIndex": ...
    def search_vector(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.intp], NDArray[np.float32]]:
        """Return (indices, similarities) for the top_k nearest neighbours."""
        ...
    def batch_search(
        self,
        queries: NDArray[np.float32],
        top_k: int,
    ) -> List[Tuple[NDArray[np.intp], NDArray[np.float32]]]:
        """Return a list of (indices, similarities) tuples, one per query row."""
        ...
    def __repr__(self) -> str: ...

class QuantizedIndex:
    """Scalar-quantized (u8) search index for memory-efficient similarity search."""

    @staticmethod
    def from_dataset(dataset: EmbeddingDataset) -> "QuantizedIndex": ...
    def search_vector(
        self,
        query: NDArray[np.float32],
        top_k: int,
    ) -> Tuple[NDArray[np.intp], NDArray[np.float32]]:
        """Return (indices, similarities) for the top_k nearest neighbours."""
        ...
    def compression_ratio(self) -> float:
        """Approximate compression ratio vs raw f32 storage (≥ 1.0)."""
        ...
    def memory_usage_bytes(self) -> int:
        """Estimated memory occupied by the index in bytes."""
        ...
    def __repr__(self) -> str: ...

# ── Rust extension functions ──────────────────────────────────────────────────

def compress_embeddings(
    vectors: NDArray[np.float32],
    ids: Optional[List[str]] = ...,
) -> Tuple[SearchIndex, QuantizedIndex]:
    """
    Build both a SearchIndex and a QuantizedIndex from a 2-D f32 array.

    Args:
        vectors: Shape (n, d), dtype float32.
        ids: Optional per-row string identifiers; defaults to ``vec_0`` … ``vec_n``.

    Returns:
        A (SearchIndex, QuantizedIndex) tuple.
    """
    ...

def analyze_compression_quality(
    original: NDArray[np.float32],
    compressed_index: QuantizedIndex,
    num_samples: Optional[int] = ...,
) -> Dict[str, float]:
    """
    Compute quality metrics comparing original vectors against the quantized index.

    Returned keys: ``average_similarity``, ``max_similarity``, ``min_similarity``,
    ``compression_ratio``, ``memory_savings_percent``, ``samples_analyzed``.
    """
    ...

def benchmark_search_performance(
    index: SearchIndex,
    queries: NDArray[np.float32],
    top_k: int,
    num_runs: Optional[int] = ...,
) -> Dict[str, float]:
    """
    Measure search latency and QPS.

    Returned keys: ``average_latency_ms``, ``queries_per_second``,
    ``successful_queries``, ``total_runs``.
    """
    ...

# ── Python convenience wrappers ───────────────────────────────────────────────

class VectroConfig:
    """Runtime configuration for index creation."""

    compression_method: str
    quantization_bits: int
    search_threads: Optional[int]
    memory_map: bool

    def __init__(
        self,
        compression_method: str = ...,
        quantization_bits: int = ...,
        search_threads: Optional[int] = ...,
        memory_map: bool = ...,
    ) -> None: ...

def create_index(
    vectors: NDArray,
    ids: Optional[List[str]] = ...,
    config: Optional[VectroConfig] = ...,
) -> SearchIndex:
    """Build an exact SearchIndex from a 2-D array.  Accepts any numeric dtype."""
    ...

def create_quantized_index(
    vectors: NDArray,
    ids: Optional[List[str]] = ...,
    config: Optional[VectroConfig] = ...,
) -> QuantizedIndex:
    """Build a QuantizedIndex from a 2-D array.  Accepts any numeric dtype."""
    ...

def search_similar(
    index: Union[SearchIndex, QuantizedIndex],
    query: NDArray,
    top_k: int = ...,
) -> Tuple[NDArray[np.intp], NDArray[np.float32]]:
    """Search an index with a single query vector."""
    ...

def batch_search(
    index: Union[SearchIndex, QuantizedIndex],
    queries: NDArray,
    top_k: int = ...,
) -> List[Tuple[NDArray[np.intp], NDArray[np.float32]]]:
    """Search an index with multiple query vectors (one result tuple per row)."""
    ...

def load_embeddings_from_array(
    vectors: NDArray,
    ids: Optional[List[str]] = ...,
) -> EmbeddingDataset:
    """Wrap a numpy array in an EmbeddingDataset."""
    ...

def generate_quality_report(
    vectors: NDArray,
    quantized_index: QuantizedIndex,
    num_samples: int = ...,
) -> Dict[str, object]:
    """
    Extended quality report including a letter-grade and space-savings estimate.

    Returned keys include all keys from ``analyze_compression_quality`` plus:
    ``quality_grade`` (str), ``recommendation`` (str),
    ``memory_usage_mb`` (float), ``original_size_estimate_mb`` (float).
    """
    ...

def save_index(
    index: Union[SearchIndex, QuantizedIndex],
    filepath: str,
) -> None:
    """Persist an index to disk (not yet implemented — raises NotImplementedError)."""
    ...

def load_index(filepath: str) -> Union[SearchIndex, QuantizedIndex]:
    """Load an index from disk (not yet implemented — raises NotImplementedError)."""
    ...

def info() -> Dict[str, str]:
    """Return a dict with keys: version, author, description, rust_available."""
    ...

def version() -> str:
    """Return the package version string."""
    ...

# ── Package metadata ──────────────────────────────────────────────────────────

__version__: str
__author__: str
__description__: str
