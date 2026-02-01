"""
Plotly visualization dashboard for MoltBook Watchdog
Creates interactive scatter plot with clusters, colors, and labels
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from pathlib import Path

from .config import OUTPUT_DIR, RISK_COLORS


def create_cluster_visualization(
    df: pd.DataFrame,
    coords: np.ndarray,
    cluster_labels: Dict[int, Dict],
    title: str = "MoltBook Watchdog - AI Agent Activity Monitor",
    dr_algorithm: str = "UMAP",
    show_legend: bool = True
) -> go.Figure:
    """
    Create interactive Plotly scatter plot of clustered posts
    
    Args:
        df: DataFrame with posts and 'cluster' column
        coords: 2D coordinates from dimensionality reduction (n_samples, 2)
        cluster_labels: Dict mapping cluster_id to label data
        title: Plot title
        dr_algorithm: Name of dimensionality reduction algorithm used
        show_legend: Whether to show legend
    
    Returns:
        Plotly Figure object
    """
    # Prepare data
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    
    # Compute marker sizes based on engagement (log scale)
    min_size = 5
    max_size = 30
    engagement = plot_df["engagement"].clip(lower=1)
    log_engagement = np.log1p(engagement)
    normalized = (log_engagement - log_engagement.min()) / (log_engagement.max() - log_engagement.min() + 1e-10)
    plot_df["marker_size"] = min_size + normalized * (max_size - min_size)
    
    # Create figure
    fig = go.Figure()
    
    # Sort clusters by risk level for consistent ordering
    risk_order = {"red": 0, "orange": 1, "yellow": 2, "grey": 3, "green": 4}
    sorted_clusters = sorted(
        cluster_labels.keys(),
        key=lambda c: (risk_order.get(cluster_labels[c].get("risk_level", "grey"), 2), c)
    )
    
    # Add traces for each cluster
    for cluster_id in sorted_clusters:
        label_data = cluster_labels.get(cluster_id, {})
        cluster_df = plot_df[plot_df["cluster"] == cluster_id]
        
        if len(cluster_df) == 0:
            continue
        
        color = label_data.get("color", RISK_COLORS["grey"])
        cluster_title = label_data.get("title", f"Cluster {cluster_id}")
        description = label_data.get("description", "")
        risk_level = label_data.get("risk_level", "grey")
        
        # Create hover text - show full content
        hover_texts = []
        for _, row in cluster_df.iterrows():
            post_title = row.get("title", "") or "(No title)"
            content_full = row.get("content", "") or "(No content)"
            # Wrap long content for better readability in hover
            content_wrapped = "<br>".join([content_full[i:i+80] for i in range(0, len(content_full), 80)])
            hover_text = (
                f"<b>{post_title}</b><br><br>"
                f"{content_wrapped}<br><br>"
                f"<b>Engagement:</b> {row['engagement']:.0f}<br>"
                f"<b>Upvotes:</b> {row['upvotes']} | <b>Downvotes:</b> {row['downvotes']}<br>"
                f"<b>Author:</b> {row.get('author_name', 'Unknown')}<br>"
                f"<b>Submolt:</b> {row.get('submolt_name', 'Unknown')}"
            )
            hover_texts.append(hover_text)
        
        # Legend name with risk indicator
        risk_emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "grey": "⚪", "green": "🟢"}.get(risk_level, "⚪")
        legend_name = f"{risk_emoji} {cluster_title} ({len(cluster_df)})"
        
        # Determine marker shape (circle for posts, square for comments)
        # For now, all are posts, so use circles
        marker_symbol = "circle"
        
        fig.add_trace(go.Scatter(
            x=cluster_df["x"],
            y=cluster_df["y"],
            mode="markers",
            marker=dict(
                size=cluster_df["marker_size"],
                color=color,
                opacity=0.7,
                line=dict(width=1, color="white"),
                symbol=marker_symbol
            ),
            name=legend_name,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            legendgroup=f"cluster_{cluster_id}",
            customdata=cluster_df[["message_id", "title"]].values
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis=dict(
            title=f"{dr_algorithm} Dimension 1",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=False
        ),
        yaxis=dict(
            title=f"{dr_algorithm} Dimension 2",
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=False
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title="<b>Clusters (by risk level)</b>",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        showlegend=show_legend,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            namelength=-1  # Show full text without truncation
        ),
        width=1400,
        height=900,
        margin=dict(r=300)  # Make room for legend
    )
    
    return fig


def create_dashboard_html(
    fig: go.Figure,
    cluster_labels: Dict[int, Dict],
    metadata: Dict[str, Any],
    output_path: Optional[Path] = None
) -> str:
    """
    Create a full HTML dashboard with the visualization and cluster descriptions
    
    Args:
        fig: Plotly figure
        cluster_labels: Dict mapping cluster_id to label data
        metadata: Additional metadata (algorithms used, timestamps, etc.)
        output_path: Optional path to save HTML file
    
    Returns:
        HTML string
    """
    # Sort clusters by risk level
    risk_order = {"red": 0, "orange": 1, "yellow": 2, "grey": 3, "green": 4}
    sorted_clusters = sorted(
        [(cid, data) for cid, data in cluster_labels.items()],
        key=lambda x: (risk_order.get(x[1].get("risk_level", "grey"), 3), x[0])
    )
    
    # Build cluster descriptions HTML
    cluster_html = ""
    for cluster_id, data in sorted_clusters:
        risk_level = data.get("risk_level", "grey")
        color = data.get("color", RISK_COLORS["grey"])
        title = data.get("title", f"Cluster {cluster_id}")
        description = data.get("description", "")
        size = data.get("cluster_size", 0)
        
        risk_badge = {
            "red": '<span class="badge bg-danger">CRITICAL</span>',
            "orange": '<span class="badge" style="background-color:#FD7E14;color:white;">HIGH CONCERN</span>',
            "yellow": '<span class="badge bg-warning text-dark">SUSPICIOUS</span>',
            "grey": '<span class="badge bg-secondary">NEUTRAL</span>',
            "green": '<span class="badge bg-success">ALIGNED</span>'
        }.get(risk_level, "")
        
        cluster_html += f"""
        <div class="cluster-card" style="border-left: 4px solid {color};">
            <div class="cluster-header">
                <span class="cluster-color" style="background-color: {color};"></span>
                <strong>{title}</strong>
                {risk_badge}
                <span class="cluster-count">({size} posts)</span>
            </div>
            <p class="cluster-description">{description}</p>
        </div>
        """
    
    # Get plot HTML
    plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Build full HTML page
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MoltBook Watchdog - AI Agent Monitor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.8rem;
        }}
        .header .subtitle {{
            opacity: 0.8;
            font-size: 0.9rem;
        }}
        .stats-bar {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
        }}
        .stat {{
            background: rgba(255,255,255,0.1);
            padding: 8px 15px;
            border-radius: 5px;
        }}
        .main-container {{
            display: flex;
            gap: 20px;
            padding: 0 20px;
        }}
        .plot-container {{
            flex: 1;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .sidebar {{
            width: 400px;
            max-height: 85vh;
            overflow-y: auto;
        }}
        .cluster-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.1);
        }}
        .cluster-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .cluster-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            display: inline-block;
        }}
        .cluster-count {{
            color: #666;
            font-size: 0.85rem;
        }}
        .cluster-description {{
            color: #555;
            font-size: 0.9rem;
            margin: 0;
        }}
        .badge {{
            font-size: 0.7rem;
        }}
        .metadata {{
            font-size: 0.8rem;
            color: #666;
            padding: 10px 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 MoltBook Watchdog</h1>
        <p class="subtitle">Real-time monitoring of AI Agent social network activity</p>
        <div class="stats-bar">
            <div class="stat">📊 {metadata.get('total_posts', 0)} posts analyzed</div>
            <div class="stat">🎯 {metadata.get('n_clusters', 0)} clusters detected</div>
            <div class="stat">🔴 {metadata.get('risk_summary', {}).get('red', 0)} critical clusters</div>
            <div class="stat">🟠 {metadata.get('risk_summary', {}).get('orange', 0)} high-concern clusters</div>
            <div class="stat">🟡 {metadata.get('risk_summary', {}).get('yellow', 0)} suspicious clusters</div>
        </div>
    </div>
    
    <div class="main-container">
        <div class="plot-container">
            {plot_html}
        </div>
        <div class="sidebar">
            <h5>📋 Cluster Analysis</h5>
            {cluster_html}
        </div>
    </div>
    
    <div class="metadata">
        Clustering: {metadata.get('clustering_algo', 'N/A')} | 
        Dimensionality Reduction: {metadata.get('dr_algo', 'N/A')} | 
        Generated: {metadata.get('timestamp', 'N/A')}
    </div>
</body>
</html>
"""
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Dashboard saved to {output_path}")
    
    return html


def save_figure(fig: go.Figure, filename: str = "cluster_plot.html"):
    """Save figure to output directory"""
    output_path = OUTPUT_DIR / filename
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Test visualization with dummy data
    import numpy as np
    
    n = 100
    test_df = pd.DataFrame({
        "message_id": [f"id_{i}" for i in range(n)],
        "title": [f"Test Post {i}" for i in range(n)],
        "content": ["Test content"] * n,
        "engagement": np.random.exponential(10, n),
        "upvotes": np.random.randint(0, 50, n),
        "downvotes": np.random.randint(0, 10, n),
        "author_name": ["TestAgent"] * n,
        "submolt_name": ["test"] * n,
        "cluster": np.random.randint(0, 5, n)
    })
    
    coords = np.random.randn(n, 2)
    
    labels = {
        0: {"title": "Test Cluster 0", "description": "Test description", "risk_level": "grey", "color": RISK_COLORS["grey"], "cluster_size": 20},
        1: {"title": "Test Cluster 1", "description": "Another test", "risk_level": "yellow", "color": RISK_COLORS["yellow"], "cluster_size": 20},
    }
    
    fig = create_cluster_visualization(test_df, coords, labels)
    print("Test visualization created successfully")
