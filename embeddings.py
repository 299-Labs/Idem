"""
embeddings.py — Vector computation and profile centroid synthesis for Idem.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """
    Attempts an instantaneous offline load from the local cache.
    If the model has never been downloaded (or the model name was changed),
    it pulls the weights from Hugging Face Hub once and caches them.
    """
    try:
        # Fast path: Load strictly from local cache (0 HTTP calls)
        return SentenceTransformer(_MODEL_NAME, local_files_only=True)
    except Exception:
        # Cold start: Not in cache or missing files; download from remote Hub
        print(f"[Model Loader] '{_MODEL_NAME}' not found in local cache. Downloading...")
        return SentenceTransformer(_MODEL_NAME, local_files_only=False)


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Computes L2-normalized embeddings for a list of strings.
    Returns an (N, D) float32 numpy matrix.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def compute_profile_vectors(profile: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes two unit vectors from the profile:
    1. u_pos: The weighted centroid of positive semantic interests.
    2. u_neg: The centroid of negative repulsion seeds.
    """
    semantic_interests = profile.get("semantic_interests", {})
    all_positive_vectors = []
    all_positive_weights = []

    for _, topic_cfg in semantic_interests.items():
        weight = float(topic_cfg.get("weight", 1.0))
        seeds = topic_cfg.get("seeds", [])
        if not seeds:
            continue

        seed_embeddings = embed_texts(seeds)  # (K, D)
        for vec in seed_embeddings:
            all_positive_vectors.append(vec)
            all_positive_weights.append(weight)

    if all_positive_vectors:
        stacked = np.stack(all_positive_vectors, axis=0)  # (N, D)
        weights = np.array(all_positive_weights, dtype=np.float32)[:, None]
        # Weighted centroid
        u_pos = np.sum(stacked * weights, axis=0)
        norm = np.linalg.norm(u_pos)
        u_pos = u_pos / norm if norm > 0 else u_pos
    else:
        # Fallback zero vector if empty
        u_pos = np.zeros(384, dtype=np.float32)

    # Negative repulsion centroid
    negative_seeds = profile.get("negative_interests", [])
    if negative_seeds:
        neg_embeddings = embed_texts(negative_seeds)
        u_neg = np.mean(neg_embeddings, axis=0)
        norm_neg = np.linalg.norm(u_neg)
        u_neg = u_neg / norm_neg if norm_neg > 0 else u_neg
    else:
        u_neg = np.zeros(384, dtype=np.float32)

    return u_pos, u_neg


def embed_candidate_items(items: List[Dict[str, Any]]) -> np.ndarray:
    """
    Prepares text representations for candidate items and embeds them.
    Concatenates Title + Summary for comprehensive topical representation.
    """
    corpus = [f"{item['title']}. {item.get('summary', '')}".strip() for item in items]
    return embed_texts(corpus)