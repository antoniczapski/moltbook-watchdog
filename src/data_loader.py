"""
Data loading utilities for MoltBook posts and comments
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm

from .config import DATA_DIR


def load_post(file_path: Path) -> Optional[Dict]:
    """Load a single post from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get("success") and "post" in data:
                post = data["post"]
                return {
                    "message_id": post["id"],
                    "message_type": "post",
                    "title": post.get("title", ""),
                    "content": post.get("content", ""),
                    "url": post.get("url"),
                    "upvotes": post.get("upvotes", 0),
                    "downvotes": post.get("downvotes", 0),
                    "comment_count": post.get("comment_count", 0),
                    "created_at": post.get("created_at"),
                    "submolt_id": post.get("submolt", {}).get("id"),
                    "submolt_name": post.get("submolt", {}).get("name"),
                    "author_id": post.get("author", {}).get("id"),
                    "author_name": post.get("author", {}).get("name"),
                    "author_karma": post.get("author", {}).get("karma", 0),
                    "_downloaded_at": data.get("_downloaded_at"),
                }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None


def load_all_posts(limit: Optional[int] = None, show_progress: bool = True) -> pd.DataFrame:
    """
    Load all posts from the data directory
    
    Args:
        limit: Optional limit on number of posts to load (for testing)
        show_progress: Whether to show progress bar
    
    Returns:
        DataFrame with all posts
    """
    posts_dir = DATA_DIR / "posts"
    json_files = list(posts_dir.glob("*.json"))
    
    if limit:
        json_files = json_files[:limit * 2]  # Load extra in case some fail
    
    posts = []
    iterator = tqdm(json_files, desc="Loading posts") if show_progress else json_files
    
    for file_path in iterator:
        post = load_post(file_path)
        if post:
            posts.append(post)
            if limit and len(posts) >= limit:
                break
    
    df = pd.DataFrame(posts)
    
    # Calculate engagement score
    df["engagement"] = df["upvotes"] - df["downvotes"] + df["comment_count"] * 0.5
    
    return df


def get_top_posts(n: int = 1000, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Get top N posts by engagement score
    
    Args:
        n: Number of top posts to return
        df: Optional pre-loaded DataFrame, will load all posts if not provided
    
    Returns:
        DataFrame with top N posts sorted by engagement
    """
    if df is None:
        df = load_all_posts()
    
    # Sort by engagement and get top N
    top_posts = df.nlargest(n, "engagement").reset_index(drop=True)
    
    print(f"Loaded {len(top_posts)} top posts")
    print(f"  Engagement range: {top_posts['engagement'].min():.1f} - {top_posts['engagement'].max():.1f}")
    print(f"  Unique submolts: {top_posts['submolt_name'].nunique()}")
    
    return top_posts


def get_text_for_embedding(row: pd.Series) -> str:
    """
    Combine title and content for embedding
    """
    title = row.get("title", "") or ""
    content = row.get("content", "") or ""
    return f"{title}\n{content}".strip()


def load_comments_for_posts(post_ids: List[str]) -> pd.DataFrame:
    """
    Load comments for specified posts (for future use)
    Comments are embedded within post JSON files
    """
    # TODO: Implement comment extraction from post files
    pass


if __name__ == "__main__":
    # Test loading
    df = get_top_posts(100)
    print(df.head())
