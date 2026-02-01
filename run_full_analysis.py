"""
Run full analysis on all posts with cached embeddings
"""
import pandas as pd
import numpy as np
from datetime import datetime
from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.data_loader import load_all_posts
from src.embeddings import get_embedding_matrix
from src.clustering import run_clustering
from src.dimensionality_reduction import run_dimensionality_reduction
from src.cluster_labeling import label_all_clusters, get_risk_summary
from src.visualization import create_cluster_visualization, create_dashboard_html

print('=' * 60)
print('MoltBook Watchdog - Full Analysis (cached embeddings only)')
print('=' * 60)

# Load existing embeddings
print('\n📥 Loading cached embeddings...')
emb_df = pd.read_parquet(PROCESSED_DIR / 'embeddings.parquet')
print(f'  Loaded {len(emb_df)} posts with embeddings')

# Load all posts and merge
print('\n📥 Loading post data...')
all_posts = load_all_posts()
print(f'  Total posts: {len(all_posts)}')

# Merge to get posts with embeddings
posts_df = all_posts[all_posts['message_id'].isin(emb_df['message_id'])]
posts_df = posts_df.merge(emb_df[['message_id', 'embedding']], on='message_id', how='inner')
print(f'  Posts with embeddings: {len(posts_df)}')

# Get embedding matrix
print('\n📊 Preparing embedding matrix...')
embeddings = get_embedding_matrix(posts_df)
print(f'  Shape: {embeddings.shape}')

# Clustering
print('\n🎯 Clustering with kmeans...')
cluster_labels, cluster_meta = run_clustering(embeddings, algorithm='kmeans')
posts_df['cluster'] = cluster_labels
n_clusters = cluster_meta.get('n_clusters')
print(f'  Clusters: {n_clusters}')

# Dimensionality reduction
print('\n📉 Dimensionality reduction with UMAP...')
coords, dr_meta = run_dimensionality_reduction(embeddings, algorithm='umap')

# Label clusters
print('\n🏷️ Generating cluster labels...')
cluster_label_data = label_all_clusters(posts_df, use_cache=True)
risk_summary = get_risk_summary(cluster_label_data)
print(f'  Risk summary: {risk_summary}')

# Create visualization
print('\n📊 Creating visualization...')
fig = create_cluster_visualization(posts_df, coords, cluster_label_data, dr_algorithm='UMAP')

# Create dashboard
print('\n🖥️ Generating dashboard...')
metadata = {
    'total_posts': len(posts_df),
    'n_clusters': n_clusters,
    'risk_summary': risk_summary,
    'clustering_algo': 'kmeans',
    'dr_algo': 'umap',
    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
}
create_dashboard_html(fig, cluster_label_data, metadata, OUTPUT_DIR / 'dashboard_full.html')

print('\n' + '=' * 60)
print('✅ Done! Dashboard: output/dashboard_full.html')
print('=' * 60)
