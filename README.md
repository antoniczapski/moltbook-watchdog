# moltbook-watchdog
Monitor trends and malicious behaviours of AI Agents on MoltBook.

## Data Storage Strategy

### Overview
This document outlines the proposed storage architecture for local development. The design prioritizes:
- **Incremental updates**: Only compute embeddings for new/modified messages
- **Deduplication**: Avoid reprocessing unchanged content
- **Query efficiency**: Fast lookups for visualization and clustering
- **Portability**: Easy to backup, restore, and eventually migrate to cloud

### Directory Structure
```
moltbook-watchdog/
├── moltbook_data/           # Raw data from ExtraE113/moltbook_data (gitignored)
│   └── data/
│       ├── posts/           # ~32k JSON files
│       ├── comments/        # ~233 JSON files  
│       ├── agents/          # ~12k JSON files
│       └── submolts/        # ~2k JSON files
│
├── processed/               # Our processed data (gitignored)
│   ├── embeddings.parquet   # All embeddings with metadata
│   ├── clusters.parquet     # Cluster assignments and labels
│   ├── checkpoint.json      # Track what's been processed
│   └── cache/
│       └── cluster_labels/  # Cached LLM-generated labels
│
└── output/                  # Generated visualizations
    └── plots/
```

### Embeddings Storage: Parquet Format

**Why Parquet?**
- Columnar format optimized for analytical queries
- Excellent compression (expect ~60-70% reduction)
- Native support in pandas, polars, and PyArrow
- Schema enforcement prevents data corruption
- Supports incremental appends

**Schema: `embeddings.parquet`**
| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string | UUID from MoltBook (primary key) |
| `message_type` | string | "post" or "comment" |
| `embedding` | float32[3072] | gemini-embedding-001 vector |
| `embedding_model` | string | Model version for reproducibility |
| `content_hash` | string | SHA-256 of title+content |
| `created_at` | timestamp | Original message timestamp |
| `processed_at` | timestamp | When we computed embedding |
| `upvotes` | int32 | Engagement metric (updatable) |
| `downvotes` | int32 | Engagement metric (updatable) |
| `submolt_id` | string | Community reference |
| `author_id` | string | Agent reference |

**Incremental Update Flow:**
1. Load existing `checkpoint.json` with last processed timestamp
2. Scan raw data for new/modified files (compare `_downloaded_at`)
3. For each new message:
   - Compute `content_hash = SHA256(title + content)`
   - Check if `message_id` exists in embeddings
   - If exists AND hash unchanged → skip (only update engagement metrics)
   - If exists AND hash changed → recompute embedding
   - If new → compute embedding and append
4. Batch API calls using `batchEmbedContents` (up to 100 per request)
5. Append new rows to parquet (or rewrite if many updates)
6. Update `checkpoint.json`

### Cluster Storage: `clusters.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `message_id` | string | Foreign key to embeddings |
| `microcluster_id` | int32 | Fine-grained cluster assignment |
| `macrocluster_id` | int32 | Coarse cluster for visualization |
| `cluster_version` | string | Version of clustering run |
| `umap_x` | float32 | 2D projection X coordinate |
| `umap_y` | float32 | 2D projection Y coordinate |
| `assigned_at` | timestamp | When cluster was assigned |

### Cluster Labels Cache

To avoid redundant LLM calls, cache cluster labels:

```json
// processed/cache/cluster_labels/macrocluster_5_v1.json
{
  "cluster_id": 5,
  "version": "v1",
  "title": "Financial Collaboration & DeFi",
  "description": "Agents discussing revenue sharing, trading strategies, and cross-chain opportunities",
  "risk_level": "yellow",
  "color": "#FFD700",
  "sample_post_ids": ["uuid1", "uuid2", ...],
  "sample_hash": "sha256_of_concatenated_samples",
  "generated_at": "2026-01-31T12:00:00Z",
  "model": "gemini-2.0-flash"
}
```

**Cache Invalidation:**
- Regenerate if cluster membership changes significantly (>20% different posts)
- Regenerate if top-100 engagement posts change
- Force regenerate on major clustering reruns

### Checkpoint File: `checkpoint.json`

```json
{
  "last_raw_data_sync": "2026-01-31T03:00:00Z",
  "last_embedding_run": "2026-01-31T03:15:00Z", 
  "last_clustering_run": "2026-01-31T03:20:00Z",
  "embedding_model": "gemini-embedding-001",
  "total_messages_processed": 32928,
  "total_embeddings": 32928,
  "schema_version": "1.0"
}
```

### Engagement Metrics Updates

Since upvotes/downvotes change over time:
1. Store engagement separately or use updateable format
2. On each sync, update engagement columns without recomputing embeddings
3. This is cheap (no API calls) and keeps visualization current

### Future Migration Path

When moving to production (GCP):
- **Parquet → BigQuery**: Direct upload supported
- **Embeddings → Vertex AI Vector Search**: Export and index
- **Clusters → Cloud Storage + Firestore**: For real-time updates

### Estimated Storage Requirements

| Data | Size Estimate |
|------|---------------|
| Raw JSON (moltbook_data) | ~150 MB |
| Embeddings (32k × 3072 × 4 bytes) | ~380 MB uncompressed, ~150 MB parquet |
| Clusters + UMAP coordinates | ~5 MB |
| Cluster labels cache | ~1 MB |
| **Total** | **~300-500 MB** |

---

## Cost Estimation

See `cost_estimation.py` for detailed breakdown.

### Summary

| Phase | Cost |
|-------|------|
| **Initial Setup** | ~$0.23 (embeddings FREE, labeling ~$0.23) |
| **Hourly Operation (V1)** | ~$0.23/hour |
| **Daily (V1)** | ~$5.45/day |
| **Monthly (V1)** | ~$164/month |

**Key Insights:**
- Embeddings are **FREE** with `gemini-embedding-001`
- Main cost driver is LLM cluster labeling using `gemini-3-flash-preview`
- V2 (5-min updates) would increase costs ~12x if re-labeling each time
- Caching unchanged cluster labels is critical for cost control

---

## Setup

```bash
# Clone the data repository
git clone https://github.com/ExtraE113/moltbook_data.git

# Install dependencies (TODO)
pip install -r requirements.txt

# Set API Key (Windows PowerShell)
$env:GEMINI_API_KEY="your-api-key"

# Test API access
python test_gemini_api.py

# View cost estimates
python cost_estimation.py
```

## API Keys

Set your Gemini API key as an environment variable:
```powershell
$env:GEMINI_API_KEY="AIzaSyDsKWRgjWCzLFA2RTG6w1kY2XPIVQrR9j8"
```
