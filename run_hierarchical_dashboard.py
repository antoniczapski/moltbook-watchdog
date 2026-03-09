"""
MoltBook Watchdog - Hierarchical Tree Dashboard
==============================================
Creates a tree visualization: Root -> Channels -> Posts -> Comments

Strategy: 
1. Static dashboard with ALL nodes (no animation) - for full view
2. Time-slider implemented via JavaScript visibility toggle (no frames duplication)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import plotly.graph_objects as go
import random

from src.config import DATA_DIR, OUTPUT_DIR, PROCESSED_DIR, RISK_COLORS
from src.cluster_labeling import label_all_clusters
from src.clustering import cluster_kmeans


def load_all_data():
    """Load all posts with their comments and timestamps"""
    posts_dir = DATA_DIR / "posts"
    posts = []
    
    print("Loading posts...")
    post_files = list(posts_dir.glob("*.json"))
    
    for pf in tqdm(post_files, desc="Loading posts"):
        try:
            with open(pf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            post = data.get("post", data)
            if not post.get("id"):
                continue
            
            submolt = post.get("submolt", {})
            submolt_name = submolt.get("name", "unknown")
            submolt_display = submolt.get("display_name", submolt_name)
            
            created_at = post.get("created_at", "")
            try:
                ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except:
                ts = None
            
            comments = data.get("comments", [])
            
            posts.append({
                "id": post["id"],
                "title": post.get("title", ""),
                "content": post.get("content", ""),
                "submolt": submolt_name,
                "submolt_display": submolt_display,
                "timestamp": ts,
                "comments": comments
            })
        except Exception as e:
            continue
    
    print(f"Loaded {len(posts)} posts")
    posts_with_ts = [p for p in posts if p["timestamp"]]
    print(f"Posts with timestamps: {len(posts_with_ts)}")
    return posts


def load_cached_embeddings():
    """Load embeddings from parquet cache"""
    emb_path = PROCESSED_DIR / "embeddings.parquet"
    if emb_path.exists():
        print(f"\n📥 Loading cached embeddings...")
        df = pd.read_parquet(emb_path)
        emb_dict = {}
        for _, row in df.iterrows():
            emb_dict[row['message_id']] = np.array(row['embedding'])
        print(f"  Cached embeddings: {len(emb_dict)}")
        return emb_dict
    return {}


def build_hierarchical_data(posts, emb_dict, cluster_labels, cluster_info):
    """Build hierarchical structure with positions"""
    
    # Group posts by channel
    channels = defaultdict(list)
    for p in posts:
        if p["id"] in emb_dict:  # Only include posts with embeddings
            channels[p["submolt"]].append(p)
    
    # Build node arrays
    nodes = []  # Each: (id, type, x, y, size, color, text, timestamp)
    edges = []  # Each: (src_idx, tgt_idx)
    
    # Root node at center
    root_idx = 0
    nodes.append({
        "id": "ROOT",
        "type": "root",
        "x": 0.0,
        "y": 0.0,
        "size": 25,
        "color": RISK_COLORS["grey"],
        "text": "MoltBook Network",
        "timestamp": 0  # Always visible
    })
    
    # Place channels in a circle around root
    channel_names = sorted(channels.keys())
    n_channels = len(channel_names)
    channel_radius = 2.0
    
    channel_indices = {}
    for i, ch in enumerate(channel_names):
        angle = 2 * np.pi * i / n_channels
        x = channel_radius * np.cos(angle)
        y = channel_radius * np.sin(angle)
        
        idx = len(nodes)
        channel_indices[ch] = idx
        nodes.append({
            "id": f"channel:{ch}",
            "type": "channel", 
            "x": x,
            "y": y,
            "size": 18,
            "color": RISK_COLORS["grey"],
            "text": f"m/{ch}<br>Posts: {len(channels[ch])}",
            "timestamp": 0  # Always visible
        })
        edges.append((root_idx, idx))
    
    print(f"  Channels: {n_channels}")
    
    # Place posts around their channels
    post_radius_base = 4.0
    post_indices = {}
    
    for ch_name in tqdm(channel_names, desc="Placing posts"):
        ch_posts = channels[ch_name]
        ch_idx = channel_indices[ch_name]
        ch_x, ch_y = nodes[ch_idx]["x"], nodes[ch_idx]["y"]
        
        # Angle of channel from center
        ch_angle = np.arctan2(ch_y, ch_x)
        
        # Place posts in a fan around the channel
        n_posts = len(ch_posts)
        if n_posts == 0:
            continue
            
        fan_angle = min(np.pi / 2, np.pi * n_posts / 100)  # Limit spread
        
        for i, post in enumerate(ch_posts):
            # Spread posts in a fan
            if n_posts > 1:
                offset_angle = -fan_angle/2 + fan_angle * i / (n_posts - 1)
            else:
                offset_angle = 0
            
            # Add some randomness to avoid overlaps
            r = post_radius_base + random.uniform(-0.3, 0.3)
            angle = ch_angle + offset_angle + random.uniform(-0.05, 0.05)
            
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            
            # Get cluster info for this post
            cluster_id = cluster_labels.get(post["id"], -1)
            if cluster_id >= 0 and cluster_id in cluster_info:
                info = cluster_info[cluster_id]
                risk = info.get("risk_level", "grey")
                cluster_label = info.get("label", f"Cluster {cluster_id}")
            else:
                risk = "grey"
                cluster_label = "Unclustered"
            
            color = RISK_COLORS.get(risk, RISK_COLORS["grey"])
            
            # Timestamp
            ts = post["timestamp"].timestamp() if post["timestamp"] else 0
            
            # Build hover text (handle None values)
            post_title = post.get("title") or ""
            post_content = post.get("content") or ""
            title = (post_title[:60] + "...") if len(post_title) > 60 else post_title
            content_preview = (post_content[:100] + "...") if len(post_content) > 100 else post_content
            hover_text = f"<b>{title}</b><br><br>{content_preview}<br><br>Cluster: {cluster_label}<br>Risk: {risk.upper()}"
            
            idx = len(nodes)
            post_indices[post["id"]] = idx
            nodes.append({
                "id": post["id"],
                "type": "post",
                "x": x,
                "y": y,
                "size": 8,
                "color": color,
                "text": hover_text,
                "timestamp": ts
            })
            edges.append((ch_idx, idx))
            
            # Add comments around the post
            for j, comment in enumerate(post.get("comments", [])[:10]):  # Limit comments
                comment_id = comment.get("id", "")
                if not comment_id:
                    continue
                
                # Inherit cluster from parent post
                c_ts_str = comment.get("created_at", "")
                try:
                    c_ts = datetime.fromisoformat(c_ts_str.replace("Z", "+00:00")).timestamp()
                except:
                    c_ts = ts
                
                # Place comment near its post
                c_angle = angle + random.uniform(-0.2, 0.2)
                c_r = r + 0.3 + random.uniform(0, 0.2)
                c_x = c_r * np.cos(c_angle)
                c_y = c_r * np.sin(c_angle)
                
                c_content = comment.get("content", "")[:80]
                c_hover = f"<b>Comment</b><br>{c_content}..."
                
                c_idx = len(nodes)
                nodes.append({
                    "id": comment_id,
                    "type": "comment",
                    "x": c_x,
                    "y": c_y,
                    "size": 4,
                    "color": color,  # Inherit from post
                    "text": c_hover,
                    "timestamp": c_ts
                })
                edges.append((idx, c_idx))
    
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")
    
    return nodes, edges


def create_static_dashboard(nodes, edges, output_path):
    """Create static (non-animated) dashboard - works with large datasets"""
    print("\n📊 Creating static dashboard...")
    
    # Prepare arrays
    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_sizes = [n["size"] for n in nodes]
    node_colors = [n["color"] for n in nodes]
    node_texts = [n["text"] for n in nodes]
    
    # Build edge coordinates
    edge_x = []
    edge_y = []
    for src, tgt in edges:
        edge_x.extend([nodes[src]["x"], nodes[tgt]["x"], None])
        edge_y.extend([nodes[src]["y"], nodes[tgt]["y"], None])
    
    # Count by type
    n_posts = sum(1 for n in nodes if n["type"] == "post")
    n_comments = sum(1 for n in nodes if n["type"] == "comment")
    n_channels = sum(1 for n in nodes if n["type"] == "channel")
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.3, color='rgba(150,150,150,0.15)'),
        hoverinfo='none',
        name='Connections'
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=0.5, color='rgba(255,255,255,0.3)')
        ),
        text=node_texts,
        hoverinfo='text',
        hoverlabel=dict(bgcolor="white", font_size=11, namelength=-1),
        name='Nodes'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"MoltBook Hierarchical Network<br>"
                 f"<sub>Channels: {n_channels} | Posts: {n_posts} | Comments: {n_comments}</sub>",
            x=0.5, font=dict(size=18, color='white')
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='white'),
        margin=dict(l=20, r=150, t=80, b=20)
    )
    
    # Add risk legend
    for i, (label, color) in enumerate([
        ("🔴 CRITICAL", RISK_COLORS["red"]),
        ("🟠 HIGH CONCERN", RISK_COLORS["orange"]),
        ("🟡 SUSPICIOUS", RISK_COLORS["yellow"]),
        ("⚪ NEUTRAL", RISK_COLORS["grey"]),
        ("🟢 ALIGNED", RISK_COLORS["green"]),
    ]):
        fig.add_annotation(
            x=1.02, y=0.95 - i * 0.05,
            xref="paper", yref="paper",
            text=f"<span style='color:{color}'>●</span> {label}",
            showarrow=False, font=dict(size=11), align="left"
        )
    
    # Node type legend
    fig.add_annotation(
        x=1.02, y=0.65,
        xref="paper", yref="paper",
        text="<b>Node Types:</b>",
        showarrow=False, font=dict(size=10), align="left"
    )
    for i, (label, size) in enumerate([
        ("○ Channel", 18),
        ("• Post", 8),
        ("· Comment", 4),
    ]):
        fig.add_annotation(
            x=1.02, y=0.60 - i * 0.05,
            xref="paper", yref="paper",
            text=label,
            showarrow=False, font=dict(size=10), align="left"
        )
    
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ Dashboard saved to {output_path}")


def create_sampled_animated_dashboard(nodes, edges, output_path, max_nodes=8000, n_frames=30):
    """Create animated dashboard with sampled data to fit in memory"""
    print(f"\n🎬 Creating animated dashboard (sampling {max_nodes} nodes, {n_frames} frames)...")
    
    # Sample nodes (keep all channels and root, sample posts/comments)
    important_nodes = [n for n in nodes if n["type"] in ("root", "channel")]
    other_nodes = [n for n in nodes if n["type"] not in ("root", "channel")]
    
    # Sample other nodes
    n_sample = max_nodes - len(important_nodes)
    if len(other_nodes) > n_sample:
        sampled = random.sample(other_nodes, n_sample)
    else:
        sampled = other_nodes
    
    sample_nodes = important_nodes + sampled
    sample_ids = {n["id"] for n in sample_nodes}
    
    # Rebuild index mapping
    id_to_new_idx = {n["id"]: i for i, n in enumerate(sample_nodes)}
    
    # Filter edges
    sample_edges = []
    for src, tgt in edges:
        src_id = nodes[src]["id"]
        tgt_id = nodes[tgt]["id"]
        if src_id in sample_ids and tgt_id in sample_ids:
            sample_edges.append((id_to_new_idx[src_id], id_to_new_idx[tgt_id]))
    
    print(f"  Sampled nodes: {len(sample_nodes)}")
    print(f"  Sampled edges: {len(sample_edges)}")
    
    # Get timestamp range
    timestamps = [n["timestamp"] for n in sample_nodes if n["timestamp"] > 0]
    if not timestamps:
        print("  No timestamps found, using static dashboard")
        return create_static_dashboard(sample_nodes, sample_edges, output_path)
    
    t_min, t_max = min(timestamps), max(timestamps)
    time_bins = np.linspace(t_min, t_max, n_frames)
    
    print(f"  Time range: {datetime.fromtimestamp(t_min)} to {datetime.fromtimestamp(t_max)}")
    
    # Prepare static arrays
    node_x = np.array([n["x"] for n in sample_nodes])
    node_y = np.array([n["y"] for n in sample_nodes])
    node_sizes = np.array([n["size"] for n in sample_nodes])
    node_colors = [n["color"] for n in sample_nodes]
    node_texts = [n["text"] for n in sample_nodes]
    node_timestamps = np.array([n["timestamp"] for n in sample_nodes])
    
    edge_sources = np.array([e[0] for e in sample_edges])
    edge_targets = np.array([e[1] for e in sample_edges])
    
    # Build frames
    frames = []
    invisible_color = 'rgba(0,0,0,0)'
    
    for frame_idx, t_bin in enumerate(tqdm(time_bins, desc="Creating frames")):
        # Visibility: visible if timestamp <= t_bin or timestamp == 0 (always visible)
        node_visible = (node_timestamps <= t_bin) | (node_timestamps == 0)
        
        # Create frame colors/sizes
        frame_colors = [node_colors[i] if node_visible[i] else invisible_color for i in range(len(sample_nodes))]
        frame_sizes = [node_sizes[i] if node_visible[i] else 0 for i in range(len(sample_nodes))]
        
        # Edge visibility
        edge_visible = node_visible[edge_sources] & node_visible[edge_targets]
        
        frame_edge_x = []
        frame_edge_y = []
        for i in np.where(edge_visible)[0]:
            s, t = edge_sources[i], edge_targets[i]
            frame_edge_x.extend([node_x[s], node_x[t], None])
            frame_edge_y.extend([node_y[s], node_y[t], None])
        
        if not frame_edge_x:
            frame_edge_x = [None]
            frame_edge_y = [None]
        
        n_visible = np.sum(node_visible)
        dt = datetime.fromtimestamp(t_bin)
        
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=frame_edge_x, y=frame_edge_y,
                    mode='lines',
                    line=dict(width=0.3, color='rgba(150,150,150,0.15)'),
                    hoverinfo='none'
                ),
                go.Scatter(
                    x=node_x.tolist(), y=node_y.tolist(),
                    mode='markers',
                    marker=dict(size=frame_sizes, color=frame_colors),
                    text=node_texts, hoverinfo='text'
                )
            ],
            name=str(frame_idx),
            layout=go.Layout(title=dict(text=f"MoltBook Network - {dt.strftime('%Y-%m-%d %H:%M')}<br><sub>Nodes: {n_visible}</sub>"))
        )
        frames.append(frame)
    
    # Build full edge coordinates for final view
    edge_x_full, edge_y_full = [], []
    for s, t in sample_edges:
        edge_x_full.extend([node_x[s], node_x[t], None])
        edge_y_full.extend([node_y[s], node_y[t], None])
    
    # Create figure
    fig = go.Figure(
        data=[
            go.Scatter(x=edge_x_full, y=edge_y_full, mode='lines',
                      line=dict(width=0.3, color='rgba(150,150,150,0.15)'), hoverinfo='none'),
            go.Scatter(x=node_x.tolist(), y=node_y.tolist(), mode='markers',
                      marker=dict(size=node_sizes.tolist(), color=node_colors),
                      text=node_texts, hoverinfo='text')
        ],
        frames=frames
    )
    
    # Slider
    slider_steps = []
    for i, t in enumerate(time_bins[::max(1, len(time_bins)//20)]):
        dt = datetime.fromtimestamp(t)
        slider_steps.append(dict(
            args=[[str(i * max(1, len(time_bins)//20))], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=dt.strftime('%m/%d %H:%M'),
            method="animate"
        ))
    
    fig.update_layout(
        title=dict(text=f"MoltBook Network Animation<br><sub>{len(sample_nodes)} nodes (sampled)</sub>",
                  x=0.5, font=dict(size=18, color='white')),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='white'),
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.1,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )],
        sliders=[dict(active=len(frames)-1, pad=dict(b=10, t=50), len=0.9, x=0.1, y=0, steps=slider_steps)]
    )
    
    # Legend
    for i, (label, color) in enumerate([
        ("🔴 CRITICAL", RISK_COLORS["red"]),
        ("🟠 HIGH", RISK_COLORS["orange"]),
        ("🟡 SUSPICIOUS", RISK_COLORS["yellow"]),
        ("⚪ NEUTRAL", RISK_COLORS["grey"]),
    ]):
        fig.add_annotation(x=1.02, y=0.95 - i * 0.05, xref="paper", yref="paper",
                          text=f"<span style='color:{color}'>●</span> {label}",
                          showarrow=False, font=dict(size=10), align="left")
    
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"✅ Animated dashboard saved to {output_path}")


def main():
    print("=" * 60)
    print("MoltBook Watchdog - Hierarchical Dashboard")
    print("=" * 60)
    
    # Load data
    posts = load_all_data()
    emb_dict = load_cached_embeddings()
    
    # Filter to posts with embeddings
    posts_with_emb = [p for p in posts if p["id"] in emb_dict]
    print(f"Posts with embeddings: {len(posts_with_emb)}")
    
    # Cluster embeddings
    print("\n🎯 Clustering...")
    post_ids = [p["id"] for p in posts_with_emb]
    embeddings = np.array([emb_dict[pid] for pid in post_ids])
    
    labels, _ = cluster_kmeans(embeddings, n_clusters=None)
    # Ensure labels are Python ints
    labels_list = labels.tolist()
    cluster_labels = dict(zip(post_ids, labels_list))
    
    # Create DataFrame for labeling
    print("\n🏷️ Labeling clusters...")
    df_for_labeling = pd.DataFrame([{
        "id": p["id"],
        "title": p.get("title", ""),
        "content": p.get("content", ""),
        "cluster": cluster_labels.get(p["id"], -1),
        "engagement": len(p.get("comments", []))  # Use comment count as engagement
    } for p in posts_with_emb])
    
    cluster_info = label_all_clusters(df_for_labeling, use_cache=True)
    
    # Risk summary
    risk_counts = defaultdict(int)
    for cid, info in cluster_info.items():
        risk_counts[info.get("risk_level", "grey")] += 1
    print(f"  Risk summary: {dict(risk_counts)}")
    
    # Build hierarchical structure
    print("\n🌳 Building hierarchical structure...")
    nodes, edges = build_hierarchical_data(posts_with_emb, emb_dict, cluster_labels, cluster_info)
    
    # Create STATIC dashboard (main output - works with all data)
    static_path = OUTPUT_DIR / "dashboard_hierarchical.html"
    create_static_dashboard(nodes, edges, static_path)
    
    # Create ANIMATED dashboard (sampled for memory)
    animated_path = OUTPUT_DIR / "dashboard_hierarchical_animated.html"
    create_sampled_animated_dashboard(nodes, edges, animated_path, max_nodes=8000, n_frames=30)
    
    print("\n" + "=" * 60)
    print("✅ Complete!")
    print(f"  Static dashboard: {static_path}")
    print(f"  Animated dashboard: {animated_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
