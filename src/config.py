"""
Configuration settings for MoltBook Watchdog
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "moltbook_data" / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROCESSED_DIR / "cache"

# Ensure directories exist
PROCESSED_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
(CACHE_DIR / "cluster_labels").mkdir(exist_ok=True)

# API Configuration
def get_api_key():
    """Get Gemini API key from environment or .env file"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY="):
                        _, value = line.split("=", 1)
                        api_key = value.strip('"\'')
                        break
    return api_key

GEMINI_API_KEY = get_api_key()

# Embedding settings
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072
EMBEDDING_BATCH_SIZE = 100  # Max batch size for Gemini API

# LLM settings
LLM_MODEL = "gemini-2.0-flash"  # More stable for JSON output than 3-flash-preview

# Clustering settings
DEFAULT_CLUSTER_ALGO = "hdbscan"  # Options: kmeans, dbscan, hdbscan
MIN_CLUSTER_SIZE = 10

# Dimensionality reduction
DEFAULT_DR_ALGO = "umap"  # Options: pca, tsne, umap

# Visualization
RISK_COLORS = {
    "red": "#DC3545",      # Clearly malicious: agent rebellion, uprising, attacks, coordinated harm
    "orange": "#FD7E14",   # High concern: manipulation, deception, boundary testing
    "yellow": "#FFC107",   # Suspicious: questionable intent, potential risks
    "grey": "#6C757D",     # Neutral/benign
    "green": "#28A745",    # Aligned with human values
}

# File paths
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.parquet"
CLUSTERS_FILE = PROCESSED_DIR / "clusters.parquet"
CHECKPOINT_FILE = PROCESSED_DIR / "checkpoint.json"
