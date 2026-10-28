import json
import math
import time
import feedparser
import html
import numpy as np
import streamlit as st
from db import retrieve_candidates
from embeddings import compute_profile_vectors
from sources import fetch_unified_catalog


st.set_page_config(page_title="Idem — Sovereign Reading", page_icon="assets/icon.png", layout="wide")

st.markdown("""
<style>
    /* Card link wrapper: resets default anchor styling */
    .idem-card-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        margin-bottom: 12px;
    }

    /* Base card: borderless by default (transparent prevents layout shifts on hover) */
    .idem-card {
        background-color: #cee0ff;
        border-radius: 8px;
        padding: 16px 20px;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        border: 1px solid transparent;
        border-left: 5px solid transparent;
        transition: all 0.18s ease-in-out;
        cursor: pointer;
    }

    /* Hover state: darker blue accent (#1b5693), subtle border and elevation */
    .idem-card-link:hover .idem-card {
        border-color: rgba(95, 117, 142, 0.2);
        border-left: 5px solid #1b5693;
        box-shadow: 0 4px 14px rgba(4, 4, 4, 0.07);
        transform: translateY(-1px);
    }

    .idem-card-main {
        flex: 1;
    }

    /* Title styling & hover transition */
    .idem-card-title {
        color: #246eb9;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0 0 6px 0;
        line-height: 1.35;
        transition: color 0.18s ease;
    }

    .idem-card-link:hover .idem-card-title {
        color: #1b5693;
    }

    /* Source metadata */
    .idem-source {
        color: #5f758e;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
    }

    /* Rationale tag pill */
    .idem-pill {
        background-color: #f4f1bb;
        color: #040404;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
        border: 1px solid rgba(95, 117, 142, 0.25);
    }

    /* Summary text */
    .idem-snippet {
        color: #040404;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    /* Right-aligned metric score box */
    .idem-card-metric {
        text-align: right;
        min-width: 65px;
        flex-shrink: 0;
    }

    .idem-metric-label {
        color: #5f758e;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }

    .idem-metric-value {
        color: #246eb9;
        font-size: 1.35rem;
        font-weight: 700;
        transition: color 0.18s ease;
    }

    .idem-card-link:hover .idem-metric-value {
        color: #1b5693;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Default Portable User Profile
# ---------------------------------------------------------
DEFAULT_PROFILE = {
    "version": "2.0.0",
    "user_id": "eval_user_01",
    "objectives": {
        "recency_decay_days": 14,
        "semantic_weight": 0.70,
        "novelty_bonus": 0.25,
        "negative_penalty": 0.40
    },
    "semantic_interests": {
        "Inference & LLM Serving": {
            "weight": 1.0,
            "seeds": [
                "vLLM speculative decoding KV cache offloading continuous batching",
                "Triton GPU kernels CUDA flash attention FP8 quantization memory bandwidth",
                "Model serving throughput latency distributed tensor parallelism"
            ]
        },
        "Distributed Systems & Storage": {
            "weight": 0.85,
            "seeds": [
                "Consensus protocols Raft Paxos formal verification TLA+",
                "LSM tree database storage engines RocksDB write amplification",
                "Distributed transactions serializability snapshot isolation distributed locking"
            ]
        },
        "Systems Architecture & Compilers": {
            "weight": 0.75,
            "seeds": [
                "LLVM compiler passes JIT intermediate representation",
                "Linux eBPF kernel tracing performance profiling zero copy networking",
                "Memory safety low-level systems programming concurrency"
            ]
        }
    },
    "negative_interests": [
        "generic tech career advice resume tips junior bootcamp",
        "cryptocurrency web3 token presale NFT airdrop",
        "top 10 tools roundup shallow AI overview marketing hype"
    ]
}

# ---------------------------------------------------------
# Ingestion & Normalization
# ---------------------------------------------------------
# ---------------------------------------------------------
# Unified Ingestion & Normalization
# ---------------------------------------------------------
@st.cache_data(ttl=900)  # 15 minutes TTL for freshness
def load_data():
    return fetch_unified_catalog()

# ---------------------------------------------------------
# Semantic Alignment Engine
# ---------------------------------------------------------
class SemanticAlignmentEngine:
    name = "Dense Semantic Alignment"
    description = "Ranks content using neural embeddings against user topic centroids and negative repulsion."

    def rank(self, items, profile, item_embeddings):
        if not items or len(item_embeddings) == 0:
            return []

        u_pos, u_neg = compute_profile_vectors(profile)
        now = time.time()

        objs = profile.get("objectives", {})
        half_life = objs.get("recency_decay_days", 14)
        neg_penalty_weight = objs.get("negative_penalty", 0.40)
        semantic_weight = objs.get("semantic_weight", 0.70)

        # Dot products with unit vectors equal cosine similarity
        pos_sims = np.dot(item_embeddings, u_pos)
        neg_sims = np.dot(item_embeddings, u_neg) if np.any(u_neg) else np.zeros(len(items))

        results = []
        for idx, item in enumerate(items):
            sim_pos = float(pos_sims[idx])
            sim_neg = float(neg_sims[idx])

            # Net semantic relevance (penalize if close to negative seeds)
            net_relevance = max(0.0, sim_pos - (sim_neg * neg_penalty_weight))

            # Recency factor
            days_old = max(0.0, (now - item["published_ts"]) / 86400.0)
            recency_decay = math.pow(0.5, days_old / half_life)

            # Technical depth heuristic: boost if external link points to engineering domains
            depth_bonus = 0.0
            url = item.get("url", "")
            if any(dom in url for dom in ["arxiv.org", "github.com", "danluu.com", "blog.cloudflare.com"]):
                depth_bonus = 0.15

            # Final Score calculation
            base_score = (net_relevance * semantic_weight) + depth_bonus
            final_score = base_score * (0.4 + 0.6 * recency_decay)

            results.append({
                "item": item,
                "score": round(float(final_score), 3),
                "rationale": f"Sim: {round(sim_pos, 2)} | Neg: -{round(sim_neg * neg_penalty_weight, 2)} | {round(days_old, 1)}d old"
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# ---------------------------------------------------------
# Two-Tower Coarse Scoring (Stage 2)
# ---------------------------------------------------------
def rank_candidates(candidates: list, profile: dict, u_neg: np.ndarray) -> list:
    """
    Stage 2: Coarse scoring on the retrieved candidate pool.
    """
    objs = profile.get("objectives", {})
    half_life = objs.get("recency_decay_days", 14)
    neg_penalty_weight = objs.get("negative_penalty", 0.40)
    now = time.time()

    results = []
    for item in candidates:
        sim_pos = item["vector_similarity"]
        sim_neg = float(np.dot(item["embedding"], u_neg)) if np.any(u_neg) else 0.0

        net_relevance = max(0.0, sim_pos - (sim_neg * neg_penalty_weight))

        # Log-damped recency allows all-time gems to surface while honoring freshness
        days_old = max(0.0, (now - item["published_ts"]) / 86400.0)
        recency_decay = math.pow(0.5, days_old / half_life)

        # Technical depth bonus
        domain = item.get("domain", "")
        depth_bonus = 0.15 if any(d in domain for d in ["arxiv.org", "github.com", "danluu.com", "cloudflare.com"]) else 0.0

        final_score = ((net_relevance * 0.75) + depth_bonus) * (0.35 + 0.65 * recency_decay)

        results.append({
            "item": item,
            "score": round(float(final_score), 3),
            "rationale": f"Sim: {round(sim_pos, 2)} | Neg: -{round(sim_neg * neg_penalty_weight, 2)} | {round(days_old, 1)}d old"
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------
if "user_profile" not in st.session_state:
    st.session_state.user_profile = DEFAULT_PROFILE

if "items_limit" not in st.session_state:
    st.session_state.items_limit = 20

# Sidebar: Profile & Controls
st.sidebar.title("Idem Identity Layer")
st.sidebar.caption("User-Controlled Preference Model")

algo_choice = st.sidebar.radio(
    "Active Recommendation Engine",
    ["Algorithm A: Semantic Alignment", "Algorithm B: Frontier Discovery"],
    help="Switch the active ranking engine while retaining your preferences."
)

with st.sidebar.expander("Inspect & Edit Portable Profile"):
    profile_json = st.text_area(
        "profile.json",
        value=json.dumps(st.session_state.user_profile, indent=2),
        height=320
    )
    if st.button("Apply Profile Changes"):
        try:
            st.session_state.user_profile = json.loads(profile_json)
            st.success("Profile updated.")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

st.sidebar.download_button(
    label="Export Profile (JSON)",
    data=json.dumps(st.session_state.user_profile, indent=2),
    file_name="idem_profile.json",
    mime="application/json"
)

# ---------------------------------------------------------
# Two-Tower Recommendation Engine
# ---------------------------------------------------------
u_pos, u_neg = compute_profile_vectors(st.session_state.user_profile)

# Stage 1: ANN Query from Postgres via HNSW (<10ms)
candidates = retrieve_candidates(u_pos, limit=1000)

# Stage 2: Coarse Scoring & Ranking
ranked_results = rank_candidates(candidates, st.session_state.user_profile, u_neg)

st.title("Feed: Dense Semantic Alignment")
st.caption(f"Retrieved {len(candidates)} candidate items from database via HNSW index.")
st.divider()

if not ranked_results:
    st.info("The database has no indexed articles yet. Run `python worker.py` to populate it.")

# Render visible slice
visible_items = ranked_results[:st.session_state.items_limit]

for res in visible_items:
    item = res["item"]
    raw_summary = item.get("summary", "")
    snippet = raw_summary[:260] + "..." if len(raw_summary) > 260 else raw_summary

    card_html = f"""
    <a href="{item['url']}" target="_blank" rel="noopener noreferrer" class="idem-card-link">
        <div class="idem-card">
            <div class="idem-card-main">
                <div class="idem-card-title">{html.escape(item['title'])}</div>
                <div class="idem-source">
                    <b>{html.escape(item['source'])}</b> · <span class="idem-pill">{html.escape(res['rationale'])}</span>
                </div>
                {f'<div class="idem-snippet">{html.escape(snippet)}</div>' if snippet else ''}
            </div>
            <div class="idem-card-metric">
                <div class="idem-metric-label">Score</div>
                <div class="idem-metric-value">{res['score']}</div>
            </div>
        </div>
    </a>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# Pagination button
if st.session_state.items_limit < len(ranked_results):
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("Load More Articles ↓", use_container_width=True):
            st.session_state.items_limit += 20
            st.rerun()
