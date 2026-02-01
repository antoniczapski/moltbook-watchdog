"""
Embedding generation and caching for MoltBook posts
Supports incremental updates - only computes embeddings for new posts
"""
import hashlib
import json
import requests
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

from .config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE, EMBEDDINGS_FILE, CHECKPOINT_FILE
)
from .data_loader import get_text_for_embedding

# Rate limit settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 5.0  # seconds


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of content for change detection"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def embed_texts_batch(texts: List[str], api_key: str = None) -> List[List[float]]:
    """
    Embed multiple texts using Gemini batch API with retry logic
    
    Args:
        texts: List of texts to embed (max 100)
        api_key: Optional API key override
    
    Returns:
        List of embedding vectors
    """
    api_key = api_key or GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:batchEmbedContents?key={api_key}"
    
    payload = {
        "requests": [
            {"model": f"models/{EMBEDDING_MODEL}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return [emb.get("values", []) for emb in data.get("embeddings", [])]
        elif response.status_code == 429:
            # Rate limited - exponential backoff
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            print(f"\n⚠️ Rate limited. Waiting {backoff:.0f}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(backoff)
            last_error = f"Rate limit (429)"
        else:
            last_error = f"API error: {response.status_code}"
            break
    
    raise Exception(f"Embedding API error after {MAX_RETRIES} retries: {last_error}")


def load_existing_embeddings() -> Tuple[pd.DataFrame, dict]:
    """
    Load existing embeddings from parquet file
    
    Returns:
        Tuple of (embeddings DataFrame, checkpoint dict)
    """
    embeddings_df = None
    checkpoint = {
        "embedding_model": EMBEDDING_MODEL,
        "total_embeddings": 0,
        "schema_version": "1.0"
    }
    
    if EMBEDDINGS_FILE.exists():
        embeddings_df = pd.read_parquet(EMBEDDINGS_FILE)
        print(f"Loaded {len(embeddings_df)} existing embeddings")
    
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
    
    return embeddings_df, checkpoint


def save_embeddings(df: pd.DataFrame, checkpoint: dict):
    """Save embeddings to parquet and update checkpoint"""
    df.to_parquet(EMBEDDINGS_FILE, index=False)
    
    checkpoint["total_embeddings"] = len(df)
    checkpoint["last_embedding_run"] = datetime.utcnow().isoformat() + "Z"
    
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Saved {len(df)} embeddings to {EMBEDDINGS_FILE}")


def generate_embeddings(
    posts_df: pd.DataFrame,
    force_recompute: bool = False,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Generate embeddings for posts with incremental update support
    
    Args:
        posts_df: DataFrame with posts (must have message_id, title, content)
        force_recompute: If True, recompute all embeddings
        show_progress: Whether to show progress bar
    
    Returns:
        DataFrame with embeddings added
    """
    # Load existing embeddings
    existing_df, checkpoint = load_existing_embeddings()
    
    # Compute content hashes for all posts
    posts_df = posts_df.copy()
    posts_df["text"] = posts_df.apply(get_text_for_embedding, axis=1)
    posts_df["content_hash"] = posts_df["text"].apply(compute_content_hash)
    
    # Determine which posts need embedding
    if existing_df is not None and not force_recompute:
        existing_hashes = dict(zip(existing_df["message_id"], existing_df["content_hash"]))
        
        needs_embedding = []
        for _, row in posts_df.iterrows():
            msg_id = row["message_id"]
            if msg_id not in existing_hashes:
                needs_embedding.append(msg_id)
            elif existing_hashes[msg_id] != row["content_hash"]:
                needs_embedding.append(msg_id)  # Content changed
        
        posts_to_embed = posts_df[posts_df["message_id"].isin(needs_embedding)]
        print(f"Found {len(posts_to_embed)} posts needing embeddings ({len(posts_df) - len(posts_to_embed)} cached)")
    else:
        posts_to_embed = posts_df
        print(f"Computing embeddings for {len(posts_to_embed)} posts")
    
    # Generate embeddings in batches with incremental saving
    if len(posts_to_embed) > 0:
        all_embeddings = []
        texts = posts_to_embed["text"].tolist()
        
        batch_iterator = range(0, len(texts), EMBEDDING_BATCH_SIZE)
        total_batches = len(texts) // EMBEDDING_BATCH_SIZE + 1
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc="Generating embeddings", 
                                  total=total_batches)
        
        save_interval = 25  # Save every 25 batches (2500 embeddings)
        batch_count = 0
        
        for i in batch_iterator:
            batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = embed_texts_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            batch_count += 1
            
            # Incremental save to avoid losing progress on errors
            if batch_count % save_interval == 0 or i + EMBEDDING_BATCH_SIZE >= len(texts):
                # Create embeddings DataFrame for processed posts so far
                processed_count = len(all_embeddings)
                partial_posts = posts_to_embed.iloc[:processed_count]
                
                partial_embeddings_df = partial_posts[["message_id", "message_type", "content_hash",
                                                        "created_at", "upvotes", "downvotes",
                                                        "submolt_id", "author_id"]].copy()
                partial_embeddings_df["embedding"] = all_embeddings
                partial_embeddings_df["embedding_model"] = EMBEDDING_MODEL
                partial_embeddings_df["processed_at"] = datetime.utcnow().isoformat() + "Z"
                
                # Merge with existing embeddings
                if existing_df is not None and not force_recompute:
                    existing_keep = existing_df[~existing_df["message_id"].isin(partial_posts["message_id"])]
                    embeddings_df = pd.concat([existing_keep, partial_embeddings_df], ignore_index=True)
                else:
                    embeddings_df = partial_embeddings_df
                
                # Save checkpoint
                save_embeddings(embeddings_df, checkpoint)
        
        # Final save already done in loop
    else:
        embeddings_df = existing_df
    
    # Return merged DataFrame with posts data and embeddings
    result = posts_df.merge(
        embeddings_df[["message_id", "embedding"]],
        on="message_id",
        how="left"
    )
    
    return result


def get_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extract embedding matrix from DataFrame
    
    Args:
        df: DataFrame with 'embedding' column
    
    Returns:
        numpy array of shape (n_samples, embedding_dim)
    """
    embeddings = np.array(df["embedding"].tolist())
    return embeddings


if __name__ == "__main__":
    # Test embedding generation
    from .data_loader import get_top_posts
    
    posts = get_top_posts(10)
    result = generate_embeddings(posts)
    print(f"Generated embeddings shape: {get_embedding_matrix(result).shape}")
