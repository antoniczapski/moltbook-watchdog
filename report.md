# MoltBook Watchdog - Clustering Dashboard Report

## Executive Summary

MoltBook Watchdog is an AI-powered monitoring system for analyzing activity patterns on MoltBook, an AI agent social network. The system uses **semantic embeddings**, **machine learning clustering**, and **LLM-based analysis** to identify and categorize discussion themes, with particular focus on detecting potentially concerning content.

---

## System Architecture

### Pipeline Overview

The analysis pipeline follows a six-step process:

```
Load Posts → Generate Embeddings → Cluster → Dimensionality Reduction → Label Clusters → Visualize
```

### Data Source

- **Platform**: MoltBook (AI agent social network)
- **Content**: Posts with titles, content, engagement metrics, author information
- **Dataset Size**: 32,695 posts total, with cached embeddings for 28,565 posts
- **Analysis Scope**: Configurable (default: top 1,000 posts by engagement)

---

## Technical Components

### 1. Embedding Generation

**Model**: Google Gemini `text-embedding-004`  
**Dimension**: 3,072-dimensional vectors  
**Purpose**: Convert text content into semantic vector representations

Key features:
- Incremental updates (only computes new embeddings)
- Persistent caching in `processed/embeddings.parquet`
- Batch processing with automatic retry on rate limits
- Content hashing for change detection

### 2. Clustering Algorithms

Three algorithms are supported:

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **K-Means** | Centroid-based partitioning | Well-separated clusters, fast |
| **DBSCAN** | Density-based spatial clustering | Arbitrary shapes, noise detection |
| **HDBSCAN** | Hierarchical density-based | Varying densities, automatic k |

**Default**: HDBSCAN (handles varying cluster densities)  
**K-Means Optimization**: Automatic k selection via silhouette score (range 10-100)

### 3. Dimensionality Reduction

Three algorithms reduce 3,072-D embeddings to 2D for visualization:

| Algorithm | Characteristics |
|-----------|-----------------|
| **PCA** | Linear, fast, preserves global structure |
| **t-SNE** | Non-linear, preserves local neighborhoods |
| **UMAP** | Non-linear, balances local/global structure |

**Default**: UMAP with parameters:
- `n_neighbors`: 15
- `min_dist`: 0.1
- `metric`: cosine

### 4. Cluster Labeling (LLM-Powered)

**Model**: Google Gemini `gemini-2.0-flash`

For each cluster, the system:
1. Samples top 20 posts by engagement
2. Sends to Gemini for analysis
3. Receives structured response:
   - **Title**: Short descriptive name
   - **Description**: 2-3 sentence summary
   - **Risk Level**: Classification of concern level

**Risk Classification**:

| Level | Color | Description |
|-------|-------|-------------|
| 🔴 Red | `#FF4136` | High risk - immediate attention needed |
| 🟠 Orange | `#FF851B` | Elevated risk - monitor closely |
| 🟡 Yellow | `#FFDC00` | Low risk - potential concerns |
| ⚪ Grey | `#AAAAAA` | Neutral - general discussion |
| 🟢 Green | `#2ECC40` | Positive - beneficial content |

---

## Visualization Dashboard

### Interactive Features

The Plotly-based dashboard provides:

1. **Scatter Plot**: 2D projection of post embeddings
2. **Color Coding**: Clusters colored by risk level
3. **Marker Sizing**: Log-scaled by engagement (upvotes + downvotes + comments)
4. **Hover Information**:
   - Full post title and content
   - Engagement metrics
   - Author and submolt (channel)
   - Cluster assignment

### Legend & Navigation

- Clusters sorted by risk level (red → orange → yellow → grey → green)
- Click legend items to show/hide clusters
- Pan, zoom, and reset view controls
- Export to PNG option

---

## Sample Analysis Results

Based on a run with **1,000 posts** using K-Means + UMAP:

### Cluster Statistics

| Metric | Value |
|--------|-------|
| Total Posts | 1,000 |
| Number of Clusters | 99 |
| Silhouette Score | 0.138 |

### Risk Distribution

| Risk Level | Count | Percentage |
|------------|-------|------------|
| 🔴 Red | 4 | 4.0% |
| 🟠 Orange | 5 | 5.1% |
| 🟡 Yellow | 37 | 37.4% |
| ⚪ Grey | 53 | 53.5% |
| 🟢 Green | 0 | 0.0% |

### High-Engagement Clusters

Notable clusters by total engagement:

| Cluster | Posts | Total Engagement | Avg Engagement |
|---------|-------|------------------|----------------|
| 5 | 11 | 411,880 | 37,444 |
| 21 | 12 | 50,011 | 4,168 |
| 25 | 5 | 29,539 | 5,908 |
| 24 | 44 | 11,321 | 257 |
| 15 | 29 | 6,555 | 226 |

---

## Usage

### Basic Run

```bash
python run_analysis.py -n 1000 -c hdbscan -d umap
```

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --n-posts` | Number of posts to analyze | 1000 |
| `-c, --clustering` | Algorithm: kmeans, dbscan, hdbscan | hdbscan |
| `-d, --dim-reduction` | Algorithm: pca, tsne, umap | umap |
| `--force-embeddings` | Recompute all embeddings | False |
| `--no-label-cache` | Disable cluster label caching | False |
| `-o, --output` | Output filename | dashboard.html |
| `--compare` | Run multi-algorithm comparison | False |

### Multi-Algorithm Comparison

```bash
python run_analysis.py --compare
```

Generates dashboards for all combinations of clustering and dimensionality reduction algorithms.

---

## Output Files

| File | Description |
|------|-------------|
| `output/dashboard.html` | Main interactive dashboard |
| `output/cluster_stats.csv` | Per-cluster statistics |
| `output/pipeline_results.json` | Run metadata and configuration |
| `processed/embeddings.parquet` | Cached embeddings |
| `processed/cache/cluster_labels/` | Cached LLM-generated labels |

---

## Dependencies

- **pandas**, **numpy**: Data manipulation
- **scikit-learn**: Clustering, PCA, t-SNE
- **umap-learn**: UMAP dimensionality reduction
- **hdbscan**: HDBSCAN clustering
- **plotly**: Interactive visualization
- **requests**: API calls to Gemini
- **tqdm**: Progress bars

---

## Limitations & Future Work

### Current Limitations

1. **Silhouette Score**: Low scores (~0.14) suggest overlapping clusters in semantic space
2. **Fixed Sampling**: Top N posts by engagement may miss emerging concerns
3. **Binary Classification**: Risk levels are categorical, not probabilistic

### Potential Improvements

1. **Temporal Analysis**: Track cluster evolution over time
2. **Anomaly Detection**: Flag outlier posts within clusters
3. **Network Analysis**: Map author relationships and influence
4. **Real-time Monitoring**: Streaming pipeline for live analysis
5. **Confidence Scores**: Add uncertainty quantification to risk labels

---

*Report generated: February 2026*  
*MoltBook Watchdog v1.0*
