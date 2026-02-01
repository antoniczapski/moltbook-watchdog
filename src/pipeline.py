"""
Main pipeline orchestrator for MoltBook Watchdog
Runs the full analysis pipeline: load → embed → cluster → reduce → label → visualize
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, PROCESSED_DIR
from .data_loader import load_all_posts, get_top_posts
from .embeddings import generate_embeddings, get_embedding_matrix
from .clustering import run_clustering, get_cluster_stats
from .dimensionality_reduction import run_dimensionality_reduction
from .cluster_labeling import label_all_clusters, get_risk_summary
from .visualization import create_cluster_visualization, create_dashboard_html, save_figure


def run_pipeline(
    n_posts: int = 1000,
    clustering_algo: str = "hdbscan",
    dr_algo: str = "umap",
    force_recompute_embeddings: bool = False,
    use_label_cache: bool = True,
    output_filename: str = "dashboard.html",
    clustering_kwargs: Optional[Dict] = None,
    dr_kwargs: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Run the full MoltBook Watchdog analysis pipeline
    
    Args:
        n_posts: Number of top posts to analyze
        clustering_algo: Clustering algorithm ('kmeans', 'dbscan', 'hdbscan')
        dr_algo: Dimensionality reduction algorithm ('pca', 'tsne', 'umap')
        force_recompute_embeddings: If True, recompute all embeddings
        use_label_cache: If True, use cached cluster labels
        output_filename: Name of output HTML file
        clustering_kwargs: Additional arguments for clustering
        dr_kwargs: Additional arguments for dimensionality reduction
    
    Returns:
        Dict with results and metadata
    """
    print("=" * 60)
    print("🔍 MoltBook Watchdog - Analysis Pipeline")
    print("=" * 60)
    
    clustering_kwargs = clustering_kwargs or {}
    dr_kwargs = dr_kwargs or {}
    
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_posts": n_posts,
        "clustering_algo": clustering_algo,
        "dr_algo": dr_algo
    }
    
    # Step 1: Load data
    print("\n📥 Step 1: Loading posts...")
    posts_df = get_top_posts(n_posts)
    results["total_posts"] = len(posts_df)
    
    # Step 2: Generate embeddings
    print("\n🧠 Step 2: Generating embeddings...")
    posts_with_embeddings = generate_embeddings(
        posts_df,
        force_recompute=force_recompute_embeddings
    )
    
    # Step 3: Extract embedding matrix
    print("\n📊 Step 3: Preparing embedding matrix...")
    embeddings = get_embedding_matrix(posts_with_embeddings)
    print(f"  Embedding matrix shape: {embeddings.shape}")
    
    # Step 4: Clustering
    print(f"\n🎯 Step 4: Clustering with {clustering_algo}...")
    cluster_labels, cluster_meta = run_clustering(
        embeddings,
        algorithm=clustering_algo,
        **clustering_kwargs
    )
    posts_with_embeddings["cluster"] = cluster_labels
    results["n_clusters"] = cluster_meta.get("n_clusters", len(set(cluster_labels)))
    results["clustering_meta"] = {k: v for k, v in cluster_meta.items() if not isinstance(v, np.ndarray)}
    
    # Step 5: Dimensionality reduction
    print(f"\n📉 Step 5: Dimensionality reduction with {dr_algo}...")
    coords, dr_meta = run_dimensionality_reduction(
        embeddings,
        algorithm=dr_algo,
        **dr_kwargs
    )
    posts_with_embeddings["umap_x"] = coords[:, 0]
    posts_with_embeddings["umap_y"] = coords[:, 1]
    results["dr_meta"] = dr_meta
    
    # Step 6: Generate cluster labels
    print("\n🏷️ Step 6: Generating cluster labels with LLM...")
    cluster_label_data = label_all_clusters(
        posts_with_embeddings,
        use_cache=use_label_cache
    )
    results["risk_summary"] = get_risk_summary(cluster_label_data)
    print(f"  Risk summary: {results['risk_summary']}")
    
    # Step 7: Create visualization
    print("\n📊 Step 7: Creating visualization...")
    fig = create_cluster_visualization(
        posts_with_embeddings,
        coords,
        cluster_label_data,
        dr_algorithm=dr_algo.upper()
    )
    
    # Step 8: Create dashboard HTML
    print("\n🖥️ Step 8: Generating dashboard...")
    metadata = {
        "total_posts": len(posts_with_embeddings),
        "n_clusters": results["n_clusters"],
        "risk_summary": results["risk_summary"],
        "clustering_algo": clustering_algo,
        "dr_algo": dr_algo,
        "timestamp": results["timestamp"]
    }
    
    output_path = OUTPUT_DIR / output_filename
    create_dashboard_html(fig, cluster_label_data, metadata, output_path)
    results["output_path"] = str(output_path)
    
    # Save cluster statistics
    cluster_stats = get_cluster_stats(cluster_labels, posts_with_embeddings)
    stats_path = OUTPUT_DIR / "cluster_stats.csv"
    cluster_stats.to_csv(stats_path, index=False)
    print(f"  Cluster stats saved to {stats_path}")
    
    # Save results metadata
    results_path = OUTPUT_DIR / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print(f"📁 Dashboard: {output_path}")
    print("=" * 60)
    
    return results


def run_multi_algorithm_comparison(
    n_posts: int = 1000,
    clustering_algos: list = None,
    dr_algos: list = None
) -> Dict[str, Any]:
    """
    Run pipeline with multiple algorithms for comparison
    
    Args:
        n_posts: Number of posts to analyze
        clustering_algos: List of clustering algorithms to compare
        dr_algos: List of dimensionality reduction algorithms to compare
    
    Returns:
        Dict with comparison results
    """
    clustering_algos = clustering_algos or ["kmeans", "hdbscan"]
    dr_algos = dr_algos or ["pca", "tsne", "umap"]
    
    # Load data once
    print("📥 Loading posts...")
    posts_df = get_top_posts(n_posts)
    
    # Generate embeddings once
    print("🧠 Generating embeddings...")
    posts_with_embeddings = generate_embeddings(posts_df)
    embeddings = get_embedding_matrix(posts_with_embeddings)
    
    comparison_results = {}
    
    for cluster_algo in clustering_algos:
        print(f"\n🎯 Clustering with {cluster_algo}...")
        try:
            cluster_labels, cluster_meta = run_clustering(
                embeddings,
                algorithm=cluster_algo
            )
            
            for dr_algo in dr_algos:
                print(f"  📉 Reducing with {dr_algo}...")
                try:
                    coords, dr_meta = run_dimensionality_reduction(
                        embeddings,
                        algorithm=dr_algo
                    )
                    
                    key = f"{cluster_algo}_{dr_algo}"
                    comparison_results[key] = {
                        "n_clusters": cluster_meta.get("n_clusters"),
                        "silhouette": cluster_meta.get("silhouette"),
                        "noise_ratio": cluster_meta.get("noise_ratio"),
                        "dr_variance": dr_meta.get("total_variance_explained")
                    }
                    
                except Exception as e:
                    print(f"    ⚠️ {dr_algo} failed: {e}")
                    
        except Exception as e:
            print(f"  ⚠️ {cluster_algo} failed: {e}")
    
    return comparison_results


if __name__ == "__main__":
    # Run default pipeline
    results = run_pipeline(n_posts=1000)
    print(f"\nResults: {json.dumps(results, indent=2, default=str)}")
