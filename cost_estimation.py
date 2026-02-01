"""
MoltBook Watchdog - Cost Estimation
Based on system design document requirements
"""

# Data counts (from actual moltbook_data)
NUM_POSTS = 32695
NUM_COMMENTS = 233  # Note: Comments are embedded in posts, not separate
TOTAL_MESSAGES = NUM_POSTS + NUM_COMMENTS

# Average text length estimation
AVG_CHARS_PER_MESSAGE = 800  # ~790 from sampling
CHARS_PER_TOKEN = 4  # Rule of thumb for English text
AVG_TOKENS_PER_MESSAGE = AVG_CHARS_PER_MESSAGE / CHARS_PER_TOKEN  # ~200 tokens

# Gemini API Pricing (as of Jan 2026)
# https://ai.google.dev/pricing
EMBEDDING_PRICE_PER_1M_TOKENS = 0.00  # Free tier: up to 1500 requests/min

# Gemini 3 Flash Preview Pricing
GEMINI_3_FLASH_INPUT_PER_1M = 0.50     # $0.50 per 1M input tokens
GEMINI_3_FLASH_OUTPUT_PER_1M = 0.50    # Assumed symmetric pricing (usually output is same or higher)

# Legacy / Comparison
GEMINI_FLASH_INPUT_PER_1M = 0.075     # $0.075 per 1M input tokens
GEMINI_FLASH_OUTPUT_PER_1M = 0.30     # $0.30 per 1M output tokens

def estimate_costs():
    print("=" * 70)
    print("MoltBook Watchdog - Cost Estimation")
    print("=" * 70)
    
    # === EMBEDDING COSTS ===
    print("\n📊 EMBEDDING COSTS (gemini-embedding-001)")
    print("-" * 50)
    
    total_embedding_tokens = TOTAL_MESSAGES * AVG_TOKENS_PER_MESSAGE
    print(f"  Total messages: {TOTAL_MESSAGES:,}")
    print(f"  Avg tokens/message: {AVG_TOKENS_PER_MESSAGE:.0f}")
    print(f"  Total tokens: {total_embedding_tokens:,.0f}")
    print(f"  Cost: FREE (within free tier limits)")
    print(f"         Rate limit: 1500 req/min, 1M tokens/min")
    
    # === LLM COSTS FOR CLUSTER LABELING ===
    print("\n📊 LLM COSTS FOR CLUSTER LABELING")
    print("-" * 50)
    
    # Per system design: labels generated from 100 top posts per cluster
    # Estimate 20-50 clusters dynamically
    est_clusters = 30
    posts_per_cluster_summary = 100
    tokens_per_post_summary = 150  # Shorter summary version
    
    # Input tokens for cluster labeling
    input_tokens_per_cluster = posts_per_cluster_summary * tokens_per_post_summary
    total_label_input_tokens = est_clusters * input_tokens_per_cluster
    
    # Output tokens (title + description per cluster)
    output_tokens_per_cluster = 150  # ~50 word description + title
    total_label_output_tokens = est_clusters * output_tokens_per_cluster
    
    print(f"  Estimated clusters: {est_clusters}")
    print(f"  Posts sampled per cluster: {posts_per_cluster_summary}")
    print(f"  Input tokens/cluster: {input_tokens_per_cluster:,}")
    print(f"  Total input tokens: {total_label_input_tokens:,}")
    print(f"  Total output tokens: {total_label_output_tokens:,}")
    
    # Cost calculation
    flash_label_cost = (total_label_input_tokens / 1_000_000 * GEMINI_3_FLASH_INPUT_PER_1M + 
                        total_label_output_tokens / 1_000_000 * GEMINI_3_FLASH_OUTPUT_PER_1M)
    
    print(f"\n  Cost with Gemini 3 Flash: ${flash_label_cost:.4f}")
    
    # === HOURLY OPERATION COSTS ===
    print("\n📊 HOURLY OPERATION COSTS (V1 - hourly updates)")
    print("-" * 50)
    
    # Estimate 500 new messages per hour (rough estimate for growing platform)
    new_messages_per_hour = 500
    hours_per_day = 24
    
    # Embedding new messages (still free)
    hourly_embedding_tokens = new_messages_per_hour * AVG_TOKENS_PER_MESSAGE
    
    # Re-cluster and re-label every hour
    hourly_llm_input = total_label_input_tokens  # Full re-labeling
    hourly_llm_output = total_label_output_tokens
    
    hourly_flash_cost = (hourly_llm_input / 1_000_000 * GEMINI_3_FLASH_INPUT_PER_1M + 
                         hourly_llm_output / 1_000_000 * GEMINI_3_FLASH_OUTPUT_PER_1M)
    daily_flash_cost = hourly_flash_cost * hours_per_day
    monthly_flash_cost = daily_flash_cost * 30
    
    print(f"  New messages/hour: {new_messages_per_hour}")
    print(f"  Embedding tokens/hour: {hourly_embedding_tokens:,} (FREE)")
    print(f"  LLM cost/hour (Gemini 3 Flash): ${hourly_flash_cost:.4f}")
    print(f"  Daily cost: ${daily_flash_cost:.2f}")
    print(f"  Monthly cost: ${monthly_flash_cost:.2f}")
    
    # === INITIAL FULL PROCESSING ===
    print("\n📊 INITIAL FULL PROCESSING (one-time)")
    print("-" * 50)
    
    initial_embedding_cost = 0  # Free
    initial_label_cost = flash_label_cost
    total_initial = initial_embedding_cost + initial_label_cost
    
    print(f"  Embed all {TOTAL_MESSAGES:,} messages: FREE")
    print(f"  Generate cluster labels: ${initial_label_cost:.4f}")
    print(f"  TOTAL INITIAL COST: ${total_initial:.4f}")
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("💰 COST SUMMARY")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │ INITIAL SETUP (one-time)                                    │
  │   • Embed 32,928 messages: FREE                             │
  │   • Label ~30 clusters: ~${initial_label_cost:.2f}                            │
  │   • TOTAL: ~${total_initial:.2f}                                        │
  ├─────────────────────────────────────────────────────────────┤
  │ HOURLY OPERATION (V1)                                       │
  │   • Embeddings: FREE                                        │
  │   • Cluster labeling: ~${hourly_flash_cost:.4f}/hour                       │
  │   • Daily: ~${daily_flash_cost:.2f}                                         │
  │   • Monthly: ~${monthly_flash_cost:.2f}                                       │
  ├─────────────────────────────────────────────────────────────┤
  │ NOTES                                                       │
  │   • Embeddings are FREE with gemini-embedding-001           │
  │   • Main cost driver is LLM cluster labeling                │
  │   • Consider caching unchanged cluster labels               │
  │   • 5-min updates (V2) would 12x the LLM costs              │
  └─────────────────────────────────────────────────────────────┘
    """)
    
    return {
        "initial_cost": total_initial,
        "hourly_cost": hourly_flash_cost,
        "daily_cost": daily_flash_cost,
        "monthly_cost": monthly_flash_cost
    }


if __name__ == "__main__":
    estimate_costs()
