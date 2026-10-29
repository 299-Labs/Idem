"""
rerank.py — Maximal Marginal Relevance (MMR) diversification engine.
"""
from typing import List, Dict, Any
import numpy as np


def compute_pairwise_cosine_sims(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes an N x N cosine similarity matrix for a batch of unit vectors.
    """
    # Normalize rows to unit vectors to make dot-product equal cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # Guard against division by zero
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def mmr_rerank(
    candidates: List[Dict[str, Any]],
    target_k: int = 20,
    lambda_param: float = 0.70
) -> List[Dict[str, Any]]:
    """
    Greedily selects target_k items using Maximal Marginal Relevance:
    MMR = argmax [ lambda * Sim(item, User) - (1 - lambda) * max(Sim(item, Selected)) ]
    
    :param candidates: Pre-filtered list of candidate dicts (must contain 'item' with 'embedding', and 'score').
    :param target_k: Number of final diverse items to return (e.g., 20).
    :param lambda_param: 1.0 = pure relevance, 0.0 = maximum diversity. 0.70 is the sweet spot.
    """
    if not candidates or target_k <= 0:
        return []

    if len(candidates) <= target_k:
        return candidates

    # Candidates from the pipeline have structure {"item": {..., "embedding": ...}, "score": float, "rationale": str}
    # Extract embeddings and scores from the item wrapper
    embeddings = np.array([c["item"]["embedding"] for c in candidates], dtype=np.float32)
    relevance_scores = np.array([c["score"] for c in candidates], dtype=np.float32)

    # 1. Normalize relevance scores to [0, 1] so they are scale-compatible with cosine similarity
    min_score = relevance_scores.min()
    max_score = relevance_scores.max()
    if max_score > min_score:
        norm_rel = (relevance_scores - min_score) / (max_score - min_score)
    else:
        norm_rel = np.ones_like(relevance_scores)

    # 2. Precompute pairwise similarities between all candidate items
    sim_matrix = compute_pairwise_cosine_sims(embeddings)

    selected_indices: List[int] = []
    unselected_indices = list(range(len(candidates)))

    # 3. Seed the slate with the single highest-scoring item
    best_initial_idx = int(np.argmax(norm_rel))
    selected_indices.append(best_initial_idx)
    unselected_indices.remove(best_initial_idx)

    # 4. Greedily select remaining items
    while len(selected_indices) < target_k and unselected_indices:
        best_mmr = -float("inf")
        best_candidate_idx = None

        for idx in unselected_indices:
            relevance = norm_rel[idx]
            
            # Maximum similarity between this candidate and any item already selected
            max_redundancy = max(sim_matrix[idx, s] for s in selected_indices)

            # MMR equation
            mmr_score = (lambda_param * relevance) - ((1.0 - lambda_param) * max_redundancy)

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_candidate_idx = idx

        if best_candidate_idx is not None:
            selected_indices.append(best_candidate_idx)
            unselected_indices.remove(best_candidate_idx)
        else:
            break

    # Build the final ordered slate with diversity metadata
    diversified_slate = []
    for rank, idx in enumerate(selected_indices):
        # Preserve the {"item": item_dict, "score": ..., "rationale": ...} wrapper
        # that the Streamlit rendering code expects at app.py line 355: item = res["item"]
        original = candidates[idx]
        item_dict = {"item": original["item"], "score": original["score"], "rationale": original.get("rationale", "")}
        item_dict["diversity_rank"] = rank + 1
        diversified_slate.append(item_dict)

    return diversified_slate