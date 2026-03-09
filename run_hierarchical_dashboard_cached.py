"""
Hierarchical Dashboard with CACHING for fast development.
Run once to build cache, then iterate on visualization instantly.
"""

import json
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import plotly.graph_objects as go

# Paths
CACHE_DIR = Path("processed/cache/hierarchical")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

POSTS_CACHE = CACHE_DIR / "posts.pkl"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.pkl" 
CLUSTERS_CACHE = CACHE_DIR / "clusters.pkl"
CLUSTER_INFO_CACHE = CACHE_DIR / "cluster_info.pkl"

DATA_DIR = Path("moltbook_data/data/posts")
OUTPUT_DIR = Path("output")

RISK_COLORS = {
    "red": "#FF4136",
    "orange": "#FF851B",
    "yellow": "#FFDC00",
    "grey": "#AAAAAA",
    "green": "#2ECC40"
}

# ============================================================
# STEP 1: Load or Cache Posts
# ============================================================

def load_posts_cached(force_reload=False):
    """Load posts from cache or rebuild"""
    if POSTS_CACHE.exists() and not force_reload:
        print(f"📦 Loading posts from cache: {POSTS_CACHE}")
        with open(POSTS_CACHE, "rb") as f:
            return pickle.load(f)
    
    print("🔄 Loading posts from JSON files (will cache)...")
    posts = []
    json_files = list(DATA_DIR.glob("*.json"))
    
    for fp in tqdm(json_files, desc="Loading posts"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Posts are nested under "post" key
            post = data.get("post", {})
            comments = data.get("comments", [])
            
            # Parse timestamp
            ts_str = post.get("created_at", "")
            ts = None
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except:
                    pass
            
            # Parse submolt (it's a dict with name/display_name)
            submolt_raw = post.get("submolt")
            if isinstance(submolt_raw, dict):
                submolt_name = submolt_raw.get("name") or submolt_raw.get("display_name") or "unknown"
            else:
                submolt_name = str(submolt_raw) if submolt_raw else "unknown"
            
            posts.append({
                "id": post.get("id", ""),
                "title": post.get("title", "") or "",
                "content": post.get("content", "") or "",
                "submolt": submolt_name,
                "timestamp": ts,
                "comments": comments
            })
        except Exception as e:
            continue
    
    # Filter to posts with timestamps and IDs
    posts = [p for p in posts if p["timestamp"] is not None and p["id"]]
    
    # Save cache
    with open(POSTS_CACHE, "wb") as f:
        pickle.dump(posts, f)
    print(f"  ✅ Cached {len(posts)} posts")
    
    return posts


# ============================================================
# STEP 2: Load or Cache Embeddings
# ============================================================

def load_embeddings_cached(posts, force_reload=False):
    """Load embeddings and match to posts"""
    if EMBEDDINGS_CACHE.exists() and not force_reload:
        print(f"📦 Loading embeddings from cache: {EMBEDDINGS_CACHE}")
        with open(EMBEDDINGS_CACHE, "rb") as f:
            return pickle.load(f)
    
    print("🔄 Loading embeddings from parquet...")
    emb_path = Path("processed/embeddings.parquet")
    df = pd.read_parquet(emb_path)
    
    # Build dict: message_id -> embedding
    # Embedding is stored in 'embedding' column as numpy array
    emb_dict = {}
    for _, row in df.iterrows():
        msg_id = row.get("message_id", "")
        emb = row.get("embedding")
        if msg_id and emb is not None:
            emb_dict[msg_id] = np.array(emb, dtype=np.float32)
    
    print(f"  Loaded {len(emb_dict)} embeddings")
    
    # Filter posts to those with embeddings
    posts_with_emb = [p for p in posts if p["id"] in emb_dict]
    
    result = {"posts": posts_with_emb, "emb_dict": emb_dict}
    
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump(result, f)
    print(f"  ✅ Cached {len(posts_with_emb)} posts with embeddings")
    
    return result


# ============================================================
# STEP 3: Load or Cache Clusters
# ============================================================

def compute_clusters_cached(posts_with_emb, emb_dict, force_reload=False):
    """Compute clusters or load from cache"""
    if CLUSTERS_CACHE.exists() and CLUSTER_INFO_CACHE.exists() and not force_reload:
        print(f"📦 Loading clusters from cache")
        with open(CLUSTERS_CACHE, "rb") as f:
            cluster_labels = pickle.load(f)
        with open(CLUSTER_INFO_CACHE, "rb") as f:
            cluster_info = pickle.load(f)
        return cluster_labels, cluster_info
    
    print("🔄 Computing clusters (this takes a while, will cache)...")
    from src.clustering import cluster_kmeans
    from src.cluster_labeling import label_all_clusters
    
    # Build embedding matrix
    post_ids = [p["id"] for p in posts_with_emb]
    embeddings = np.array([emb_dict[pid] for pid in post_ids])
    
    # Cluster
    labels, _ = cluster_kmeans(embeddings, n_clusters=None)
    labels_list = labels.tolist()
    
    cluster_labels = {pid: labels_list[i] for i, pid in enumerate(post_ids)}
    
    # Label clusters with Gemini
    print("🏷️ Labeling clusters...")
    df_for_labeling = pd.DataFrame([{
        "id": p["id"],
        "title": p.get("title", ""),
        "content": p.get("content", ""),
        "cluster": cluster_labels.get(p["id"], -1),
        "engagement": len(p.get("comments", []))
    } for p in posts_with_emb])
    
    cluster_info = label_all_clusters(df_for_labeling, use_cache=True, show_progress=True)
    
    # Save caches
    with open(CLUSTERS_CACHE, "wb") as f:
        pickle.dump(cluster_labels, f)
    with open(CLUSTER_INFO_CACHE, "wb") as f:
        pickle.dump(cluster_info, f)
    
    print(f"  ✅ Cached cluster labels and info")
    return cluster_labels, cluster_info


# ============================================================
# STEP 4: Build Hierarchical Layout (Fast - no caching needed)
# ============================================================

def build_hierarchical_data(posts_with_emb, cluster_labels, cluster_info):
    """Build hierarchical node/edge structure with 3D radial layout"""
    nodes = []
    edges = []
    
    # Root node at center, Z=0
    nodes.append({
        "id": "root",
        "type": "root",
        "x": 0, "y": 0, "z": 0,
        "size": 25,
        "color": "#FFFFFF",
        "text": "<b>MoltBook Root</b>",
        "timestamp": 0
    })
    
    # Group posts by channel
    channels = {}
    for post in posts_with_emb:
        ch = post["submolt"] or "unknown"
        if ch not in channels:
            channels[ch] = []
        channels[ch].append(post)
    
    # Filter to channels with 100+ posts
    channels = {ch: posts for ch, posts in channels.items() if len(posts) >= 100}
    print(f"  Channels (100+ posts): {len(channels)}")
    
    # Place channels in a ring around root at Z=1
    ch_list = list(channels.keys())
    ch_indices = {}
    
    for i, ch in enumerate(ch_list):
        angle = 2 * np.pi * i / len(ch_list)
        x = 3 * np.cos(angle)
        y = 3 * np.sin(angle)
        z = 1  # Channels at Z=1
        
        idx = len(nodes)
        ch_indices[ch] = idx
        nodes.append({
            "id": f"ch_{ch}",
            "type": "channel",
            "x": x, "y": y, "z": z,
            "size": 10,
            "color": "#00D4FF",  # Cyan for channels
            "text": f"<b>m/{ch}</b><br>{len(channels[ch])} posts",
            "timestamp": 0
        })
        edges.append((0, idx))  # Connect to root
    
    # Place posts around their channel at Z=2
    post_indices = {}
    
    for ch, ch_posts in tqdm(channels.items(), desc="Placing posts"):
        ch_idx = ch_indices[ch]
        ch_x, ch_y, ch_z = nodes[ch_idx]["x"], nodes[ch_idx]["y"], nodes[ch_idx]["z"]
        ch_angle = np.arctan2(ch_y, ch_x)
        
        for i, post in enumerate(ch_posts):
            # Fan posts out from channel
            spread = min(len(ch_posts), 50)
            post_angle = ch_angle + (i - spread/2) * 0.03
            r = 6 + random.uniform(0, 1)
            x = r * np.cos(post_angle)
            y = r * np.sin(post_angle)
            z = 2 + random.uniform(-0.3, 0.3)  # Posts at Z≈2
            x = r * np.cos(post_angle)
            y = r * np.sin(post_angle)
            z = 2 + random.uniform(-0.3, 0.3)  # Posts at Z≈2
            
            # Get cluster info
            cluster_id = cluster_labels.get(post["id"], -1)
            info = cluster_info.get(cluster_id, {})
            risk = info.get("risk_level", "grey")
            cluster_label = info.get("label", "Unknown")
            color = RISK_COLORS.get(risk, RISK_COLORS["grey"])
            
            ts = post["timestamp"].timestamp() if post["timestamp"] else 0
            
            # Build hover text (safe for None)
            title = (post["title"][:60] + "...") if len(post["title"]) > 60 else post["title"]
            content = post["content"][:100] + "..." if len(post["content"]) > 100 else post["content"]
            hover = f"<b>{title}</b><br><br>{content}<br><br>Cluster: {cluster_label}<br>Risk: {risk.upper()}"
            
            idx = len(nodes)
            post_indices[post["id"]] = idx
            nodes.append({
                "id": post["id"],
                "type": "post",
                "x": x, "y": y, "z": z,
                "size": 6,
                "color": color,
                "text": hover,
                "timestamp": ts
            })
            edges.append((ch_idx, idx))
            
            # Add a few comments per post at Z≈3
            for j, comment in enumerate(post.get("comments", [])[:5]):
                cid = comment.get("id", "")
                if not cid:
                    continue
                
                c_ts_str = comment.get("created_at", "")
                try:
                    c_ts = datetime.fromisoformat(c_ts_str.replace("Z", "+00:00")).timestamp()
                except:
                    c_ts = ts
                
                c_angle = post_angle + random.uniform(-0.08, 0.08)
                c_r = r + 0.5 + random.uniform(0, 0.3)
                c_x = c_r * np.cos(c_angle)
                c_y = c_r * np.sin(c_angle)
                c_z = 3 + random.uniform(-0.2, 0.2)  # Comments at Z≈3
                
                c_content = (comment.get("content") or "")[:80]
                
                c_idx = len(nodes)
                nodes.append({
                    "id": cid,
                    "type": "comment",
                    "x": c_x, "y": c_y, "z": c_z,
                    "size": 3,
                    "color": color,
                    "text": f"<b>Comment</b><br>{c_content}...",
                    "timestamp": c_ts
                })
                edges.append((idx, c_idx))
    
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    return nodes, edges


# ============================================================
# STEP 5: Create Dashboard (Fast)
# ============================================================

def create_dashboard(nodes, edges, output_path, sample_size=10000):
    """Create a 3D hierarchical dashboard"""
    
    # Build ID -> original index map before sampling
    orig_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    
    # Convert edges from index pairs to ID pairs for proper tracking
    edge_ids = [(nodes[s]["id"], nodes[e]["id"]) for s, e in edges]
    
    # Sample if too many nodes
    if len(nodes) > sample_size:
        print(f"  Sampling {sample_size} nodes from {len(nodes)}")
        # Keep root, channels, and sample posts/comments
        important = [n for n in nodes if n["type"] in ("root", "channel")]
        others = [n for n in nodes if n["type"] not in ("root", "channel")]
        random.shuffle(others)
        nodes = important + others[:sample_size - len(important)]
    
    # Build new ID -> index map after sampling
    sampled_ids = {n["id"] for n in nodes}
    new_id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    
    # Filter edges to only include sampled nodes and convert to new indices
    edges = [(new_id_to_idx[s_id], new_id_to_idx[e_id]) 
             for s_id, e_id in edge_ids 
             if s_id in sampled_ids and e_id in sampled_ids]
    
    print(f"  Final: {len(nodes)} nodes, {len(edges)} edges")
    
    # Extract coordinates
    x = [n["x"] for n in nodes]
    y = [n["y"] for n in nodes]
    z = [n.get("z", 0) for n in nodes]
    colors = [n["color"] for n in nodes]
    sizes = [n["size"] for n in nodes]
    texts = [n["text"] for n in nodes]
    
    # Build edge traces for 3D
    edge_x = []
    edge_y = []
    edge_z = []
    for s, e in edges:
        if s < len(nodes) and e < len(nodes):
            edge_x.extend([nodes[s]["x"], nodes[e]["x"], None])
            edge_y.extend([nodes[s]["y"], nodes[e]["y"], None])
            edge_z.extend([nodes[s].get("z", 0), nodes[e].get("z", 0), None])
    
    fig = go.Figure()
    
    # 3D Edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=1, color='rgba(100,100,100,0.2)'),
        hoverinfo='skip',
        name='Connections'
    ))
    
    # 3D Nodes
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=sizes, 
            color=colors, 
            line=dict(width=0.5, color='white'),
            opacity=0.9
        ),
        text=texts,
        hoverinfo='text',
        name='Nodes'
    ))
    
    fig.update_layout(
        title="MoltBook 3D Hierarchical Dashboard",
        showlegend=False,
        hovermode='closest',
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False),
            bgcolor='#0d0d1a',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        paper_bgcolor='#0d0d1a',
        font=dict(color='white'),
        width=1400,
        height=900,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    fig.write_html(output_path)
    print(f"  ✅ Dashboard saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("MoltBook Hierarchical Dashboard (CACHED)")
    print("=" * 60)
    
    # Check what needs to be rebuilt
    import sys
    force_all = "--force" in sys.argv
    force_clusters = "--force-clusters" in sys.argv
    viz_only = "--viz-only" in sys.argv
    
    if viz_only and all(p.exists() for p in [POSTS_CACHE, EMBEDDINGS_CACHE, CLUSTERS_CACHE, CLUSTER_INFO_CACHE]):
        print("⚡ VIZ-ONLY MODE - skipping data loading")
        with open(POSTS_CACHE, "rb") as f:
            posts = pickle.load(f)
        with open(EMBEDDINGS_CACHE, "rb") as f:
            emb_data = pickle.load(f)
        with open(CLUSTERS_CACHE, "rb") as f:
            cluster_labels = pickle.load(f)
        with open(CLUSTER_INFO_CACHE, "rb") as f:
            cluster_info = pickle.load(f)
        posts_with_emb = emb_data["posts"]
    else:
        # Step 1: Posts
        posts = load_posts_cached(force_reload=force_all)
        print(f"  Posts loaded: {len(posts)}")
        
        # Step 2: Embeddings
        emb_data = load_embeddings_cached(posts, force_reload=force_all)
        posts_with_emb = emb_data["posts"]
        emb_dict = emb_data["emb_dict"]
        print(f"  Posts with embeddings: {len(posts_with_emb)}")
        
        # Step 3: Clusters
        cluster_labels, cluster_info = compute_clusters_cached(
            posts_with_emb, emb_dict, 
            force_reload=force_all or force_clusters
        )
    
    # Step 4: Build hierarchy (always runs - it's fast)
    print("\n🌳 Building hierarchical structure...")
    nodes, edges = build_hierarchical_data(posts_with_emb, cluster_labels, cluster_info)
    
    # Step 5: Create dashboard
    print("\n📊 Creating dashboard...")
    output_path = OUTPUT_DIR / "dashboard_hierarchical.html"
    create_dashboard(nodes, edges, output_path, sample_size=15000)
    
    print("\n✅ DONE!")
    print(f"\nTo iterate on visualization only, run:")
    print(f"  python run_hierarchical_dashboard_cached.py --viz-only")


if __name__ == "__main__":
    main()
