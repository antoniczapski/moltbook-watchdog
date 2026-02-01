"""
Dimensionality reduction for visualization
Supports PCA, t-SNE, UMAP
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try importing UMAP (may need installation)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")


def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    PCA dimensionality reduction
    
    Args:
        embeddings: (n_samples, n_features) array
        n_components: Number of output dimensions
        random_state: For reproducibility
    
    Returns:
        Tuple of (reduced coordinates, metadata dict)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(embeddings)
    
    metadata = {
        "algorithm": "pca",
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_variance_explained": sum(pca.explained_variance_ratio_)
    }
    
    print(f"PCA: {metadata['total_variance_explained']:.1%} variance explained")
    
    return coords, metadata


def reduce_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    t-SNE dimensionality reduction
    
    Args:
        embeddings: (n_samples, n_features) array
        n_components: Number of output dimensions
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: For reproducibility
    
    Returns:
        Tuple of (reduced coordinates, metadata dict)
    """
    # Adjust perplexity if too high for dataset
    perplexity = min(perplexity, (len(embeddings) - 1) / 3)
    
    # Use PCA for initial reduction if embeddings are high-dimensional
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca'
    )
    coords = tsne.fit_transform(embeddings)
    
    metadata = {
        "algorithm": "tsne",
        "n_components": n_components,
        "perplexity": perplexity,
        "n_iter": n_iter,
        "kl_divergence": tsne.kl_divergence_
    }
    
    print(f"t-SNE: KL divergence = {metadata['kl_divergence']:.4f}")
    
    return coords, metadata


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    UMAP dimensionality reduction
    
    Args:
        embeddings: (n_samples, n_features) array
        n_components: Number of output dimensions
        n_neighbors: Number of neighbors for local structure
        min_dist: Minimum distance between points in output
        metric: Distance metric
        random_state: For reproducibility
    
    Returns:
        Tuple of (reduced coordinates, metadata dict)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn not installed. Install with: pip install umap-learn")
    
    # Adjust n_neighbors if too high
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    coords = reducer.fit_transform(embeddings)
    
    metadata = {
        "algorithm": "umap",
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric
    }
    
    print(f"UMAP: Reduced to {n_components}D with {n_neighbors} neighbors")
    
    return coords, metadata


def run_dimensionality_reduction(
    embeddings: np.ndarray,
    algorithm: str = "umap",
    n_components: int = 2,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run dimensionality reduction with specified algorithm
    
    Args:
        embeddings: (n_samples, n_features) array
        algorithm: One of 'pca', 'tsne', 'umap'
        n_components: Number of output dimensions
        **kwargs: Algorithm-specific parameters
    
    Returns:
        Tuple of (reduced coordinates, metadata dict)
    """
    if algorithm == "pca":
        return reduce_pca(embeddings, n_components=n_components, **kwargs)
    elif algorithm == "tsne":
        return reduce_tsne(embeddings, n_components=n_components, **kwargs)
    elif algorithm == "umap":
        return reduce_umap(embeddings, n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown dimensionality reduction algorithm: {algorithm}")


if __name__ == "__main__":
    # Test dimensionality reduction
    np.random.seed(42)
    test_embeddings = np.random.randn(100, 3072)
    
    for algo in ["pca", "tsne"]:  # Skip UMAP if not installed
        coords, meta = run_dimensionality_reduction(test_embeddings, algorithm=algo)
        print(f"{algo}: output shape = {coords.shape}")
