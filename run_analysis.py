"""
MoltBook Watchdog - Main entry point
Run this script to execute the full analysis pipeline
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline, run_multi_algorithm_comparison
from src.config import GEMINI_API_KEY


def main():
    parser = argparse.ArgumentParser(
        description="MoltBook Watchdog - AI Agent Social Network Monitor"
    )
    parser.add_argument(
        "-n", "--n-posts",
        type=int,
        default=1000,
        help="Number of top posts to analyze (default: 1000)"
    )
    parser.add_argument(
        "-c", "--clustering",
        choices=["kmeans", "dbscan", "hdbscan"],
        default="hdbscan",
        help="Clustering algorithm (default: hdbscan)"
    )
    parser.add_argument(
        "-d", "--dim-reduction",
        choices=["pca", "tsne", "umap"],
        default="umap",
        help="Dimensionality reduction algorithm (default: umap)"
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Force recomputation of all embeddings"
    )
    parser.add_argument(
        "--no-label-cache",
        action="store_true",
        help="Disable cluster label caching"
    )
    parser.add_argument(
        "-o", "--output",
        default="dashboard.html",
        help="Output filename (default: dashboard.html)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison of multiple algorithms"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set.")
        print("Please set it in .env file or as environment variable:")
        print("  $env:GEMINI_API_KEY='your_api_key'")
        sys.exit(1)
    
    if args.compare:
        print("Running multi-algorithm comparison...")
        results = run_multi_algorithm_comparison(
            n_posts=args.n_posts,
            clustering_algos=["kmeans", "hdbscan"],
            dr_algos=["pca", "tsne", "umap"]
        )
        print("\n📊 Comparison Results:")
        for key, val in results.items():
            print(f"  {key}: {val}")
    else:
        results = run_pipeline(
            n_posts=args.n_posts,
            clustering_algo=args.clustering,
            dr_algo=args.dim_reduction,
            force_recompute_embeddings=args.force_embeddings,
            use_label_cache=not args.no_label_cache,
            output_filename=args.output
        )
    
    return results


if __name__ == "__main__":
    main()
