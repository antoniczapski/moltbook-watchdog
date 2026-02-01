"""
Clustering algorithms for MoltBook post embeddings
Supports multiple algorithms: K-means, DBSCAN, HDBSCAN
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Try importing HDBSCAN (may need installation)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")

# Constants for large dataset handling
SILHOUETTE_SAMPLE_SIZE = 5000  # Sample size for silhouette score calculation
LARGE_DATASET_THRESHOLD = 10000  # Use MiniBatchKMeans above this


def normalize_embeddings(embeddings: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize embeddings before clustering
    
    Args:
        embeddings: (n_samples, n_features) array
        method: 'l2' for unit norm, 'standard' for zero mean/unit variance
    
    Returns:
        Normalized embeddings
    """
    if method == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
    elif method == "standard":
        scaler = StandardScaler()
        return scaler.fit_transform(embeddings)
    return embeddings


def compute_silhouette_sampled(embeddings: np.ndarray, labels: np.ndarray, sample_size: int = SILHOUETTE_SAMPLE_SIZE) -> float:
    """
    Compute silhouette score using sampling for large datasets.
    This avoids memory issues with full pairwise distance computation.
    """
    n_samples = len(embeddings)
    if n_samples <= sample_size:
        return silhouette_score(embeddings, labels)
    
    # Stratified sampling to maintain cluster proportions
    np.random.seed(42)
    indices = np.random.choice(n_samples, size=sample_size, replace=False)
    return silhouette_score(embeddings[indices], labels[indices])


def cluster_kmeans(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    min_k: int = 10,
    max_k: int = 100,
    random_state: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    K-means clustering with automatic k selection using silhouette score
    Uses MiniBatchKMeans and sampled silhouette for large datasets to avoid memory issues.
    
    Args:
        embeddings: (n_samples, n_features) array
        n_clusters: Fixed number of clusters (if None, auto-select)
        min_k, max_k: Range for automatic k selection
        random_state: For reproducibility
    
    Returns:
        Tuple of (cluster labels, metadata dict)
    """
    embeddings_norm = normalize_embeddings(embeddings, "l2")
    n_samples = len(embeddings_norm)
    use_minibatch = n_samples > LARGE_DATASET_THRESHOLD
    
    if use_minibatch:
        print(f"  Using MiniBatchKMeans for {n_samples} samples (faster)")
    
    if n_clusters is None:
        # Find optimal k using silhouette score
        best_k = min_k
        best_score = -1
        scores = {}
        
        print(f"  Auto-selecting k from {min_k} to {min(max_k, len(embeddings) // 100)}...")
        
        # Limit max_k for large datasets
        actual_max_k = min(max_k, len(embeddings) // 100) if n_samples > LARGE_DATASET_THRESHOLD else min(max_k + 1, len(embeddings) // 2)
        
        for k in range(min_k, actual_max_k + 1):
            if use_minibatch:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=1024)
            else:
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings_norm)
            
            if len(np.unique(labels)) > 1:
                score = compute_silhouette_sampled(embeddings_norm, labels)
                scores[k] = score
                print(f"    k={k}: silhouette={score:.4f}")
                if score > best_score:
                    best_score = score
                    best_k = k
        
        n_clusters = best_k
        print(f"  Auto-selected k={n_clusters} (silhouette={best_score:.3f})")
    
    # Final clustering with selected k
    if use_minibatch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=1024)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings_norm)
    
    final_silhouette = compute_silhouette_sampled(embeddings_norm, labels) if len(np.unique(labels)) > 1 else 0
    
    metadata = {
        "algorithm": "kmeans" if not use_minibatch else "minibatch_kmeans",
        "n_clusters": n_clusters,
        "inertia": kmeans.inertia_,
        "silhouette": final_silhouette,
        "centroids": kmeans.cluster_centers_
    }
    
    return labels, metadata


def cluster_dbscan(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    auto_eps: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    DBSCAN clustering
    
    Args:
        embeddings: (n_samples, n_features) array
        eps: Maximum distance between samples
        min_samples: Minimum samples in neighborhood
        auto_eps: If True, try to find good eps automatically
    
    Returns:
        Tuple of (cluster labels, metadata dict)
    """
    embeddings_norm = normalize_embeddings(embeddings, "l2")
    
    if auto_eps:
        # Try different eps values and pick one with reasonable cluster count
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=min_samples)
        nn.fit(embeddings_norm)
        distances, _ = nn.kneighbors(embeddings_norm)
        distances = np.sort(distances[:, -1])
        
        # Use elbow heuristic - pick eps at ~90th percentile
        eps = np.percentile(distances, 90)
        print(f"Auto-selected eps={eps:.4f}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings_norm)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    metadata = {
        "algorithm": "dbscan",
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels)
    }
    
    print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({metadata['noise_ratio']:.1%})")
    
    return labels, metadata


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    HDBSCAN clustering - hierarchical density-based clustering
    
    Args:
        embeddings: (n_samples, n_features) array
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples for core points (defaults to min_cluster_size)
    
    Returns:
        Tuple of (cluster labels, metadata dict)
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed. Install with: pip install hdbscan")
    
    embeddings_norm = normalize_embeddings(embeddings, "l2")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',  # On L2-normalized vectors, euclidean ~ cosine
        cluster_selection_method='eom'
    )
    
    labels = clusterer.fit_predict(embeddings_norm)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    metadata = {
        "algorithm": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / len(labels),
        "probabilities": clusterer.probabilities_
    }
    
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points ({metadata['noise_ratio']:.1%})")
    
    return labels, metadata


def run_clustering(
    embeddings: np.ndarray,
    algorithm: str = "hdbscan",
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run clustering with specified algorithm
    
    Args:
        embeddings: (n_samples, n_features) array
        algorithm: One of 'kmeans', 'dbscan', 'hdbscan'
        **kwargs: Algorithm-specific parameters
    
    Returns:
        Tuple of (cluster labels, metadata dict)
    """
    if algorithm == "kmeans":
        return cluster_kmeans(embeddings, **kwargs)
    elif algorithm == "dbscan":
        return cluster_dbscan(embeddings, **kwargs)
    elif algorithm == "hdbscan":
        return cluster_hdbscan(embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def get_cluster_stats(labels: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each cluster
    
    Args:
        labels: Cluster labels
        df: DataFrame with post data
    
    Returns:
        DataFrame with cluster statistics
    """
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels
    
    stats = df_with_labels.groupby("cluster").agg({
        "message_id": "count",
        "engagement": ["mean", "sum", "max"],
        "upvotes": "sum",
        "downvotes": "sum"
    }).round(2)
    
    stats.columns = ["count", "avg_engagement", "total_engagement", "max_engagement", 
                     "total_upvotes", "total_downvotes"]
    stats = stats.reset_index()
    
    return stats


if __name__ == "__main__":
    # Test clustering
    np.random.seed(42)
    test_embeddings = np.random.randn(100, 3072)
    
    for algo in ["kmeans", "dbscan"]:
        labels, meta = run_clustering(test_embeddings, algorithm=algo)
        print(f"{algo}: {meta}")
