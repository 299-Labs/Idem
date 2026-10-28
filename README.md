# Idem

A customizable recommendation system where users own their profile.

## Concept

Idem implements a two-tower recommendation engine with MMR diversification that surfaces technically dense content without redundancy. Users maintain a portable preference profile (JSON) that defines:

- **Positive interests** (`semantic_interests`): Weighted topic seeds that attract content
- **Negative interests** (`negative_interests`): Repulsion seeds that penalize irrelevant content
- **Objectives**: Recency decay, semantic weight, negative penalty, novelty bonus

The recommendation pipeline runs in three stages:

1. **HNSW ANN Retrieval** (`db.retrieve_candidates`): 1,000 candidate items from Postgres via vector similarity (<10ms)
2. **Coarse Scoring** (`rank_candidates`): Blends positive relevance, negative penalty, recency decay, and domain depth bonus (1,000 → 50 items)
3. **MMR Reranking** (`rerank.mmr_rerank`): Maximal Marginal Relevance diversification selecting 20 diverse items from the top 50 (50 → 20)

Users can edit their portable profile via the sidebar, export/import it, and switch between recommendation engines.

## Project Structure

```
Idem/
├── app.py                # Streamlit UI + 3-stage recommendation pipeline
├── embeddings.py         # SentenceTransformer with local_files_only fallback
├── sources.py            # RSS & ATProto feed ingestion
├── db.py                 # Postgres HNSW vector operations
├── rerank.py             # MMR diversity reranker module
├── worker.py             # Ingestion worker (populates DB)
├── profiles/             # User profile JSON files
├── data/                 # Embedding cache and index files
├── overview.md           # Development documentation
└── readme.md             # User-facing documentation
```

## Key Features

- **Dynamic model loading**: `local_files_only=True` for instant offline load; automatic Hugging Face Hub download on cache miss
- **Two-tower architecture**: User vector (u_pos, u_neg) ANN retrieval + coarse scoring + MMR diversification
- **Portable profiles**: User-controlled preference JSON that persists across sessions
- **Diverse feeds**: MMR ensures technical density without redundancy when multiple sources cover same topic
- **Flexible ingestion**: RSS feeds + ATProtocol (Bluesky) feeds with proper User-Agent headers
- **Session caching**: Streamlit `@cache_resource` and `@cache_data` for model and embedding persistence
