import json
import math
import time
import feedparser
import html
import streamlit as st
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
    "version": "1.0.0",
    "user_id": "eval_user_01",
    "objectives": {
        "recency_decay_days": 7,
        "novelty_bonus": 0.35,
        "prefer_longform": 0.80
    },
    "interests": {
        "distributed_systems": 0.95,
        "inference_optimization": 0.90,
        "llm_architecture": 0.85,
        "databases": 0.75,
        "security": 0.60,
        "frontend": 0.15
    },
    "blocked_terms": [
        "roundup",
        "top 10",
        "career advice",
        "crypto",
        "nft"
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
# Competing Recommendation Engines
# ---------------------------------------------------------
class DirectAlignmentEngine:
    """Algorithm A: Rewards high topical overlap and recent posts."""
    name = "Direct Alignment"
    description = "Optimizes strictly for declared core competencies and publication recency."

    def rank(self, items, profile):
        results = []
        now = time.time()
        interests = profile.get("interests", {})
        half_life = profile.get("objectives", {}).get("recency_decay_days", 7)
        blocked = [b.lower() for b in profile.get("blocked_terms", [])]

        for item in items:
            text = f"{item['title']} {item['summary']}".lower()
            if any(b in text for b in blocked):
                continue

            # Compute topic overlap
            topic_score = 0.0
            matched_topics = []
            for topic, weight in interests.items():
                clean_topic = topic.replace("_", " ")
                if clean_topic in text:
                    topic_score += weight
                    matched_topics.append(clean_topic)

            # Compute recency decay
            days_old = max(0, (now - item["published_ts"]) / 86400)
            decay = math.pow(0.5, days_old / half_life)

            final_score = (0.2 + topic_score) * decay

            results.append({
                "item": item,
                "score": round(final_score, 3),
                "rationale": f"Matched: {', '.join(matched_topics) if matched_topics else 'General relevance'} | {round(days_old, 1)}d old"
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

class FrontierDiscoveryEngine:
    """Algorithm B: Penalizes source repetition and rewards peripheral exploration."""
    name = "Frontier Discovery"
    description = "Introduces novelty and enforces domain diversity across the slate."

    def rank(self, items, profile):
        results = []
        source_counts = {}
        blocked = [b.lower() for b in profile.get("blocked_terms", [])]
        novelty_bonus = profile.get("objectives", {}).get("novelty_bonus", 0.35)

        for item in items:
            text = f"{item['title']} {item['summary']}".lower()
            if any(b in text for b in blocked):
                continue

            # Diversity damper: penalize repetitive sources
            source = item["source"]
            repetition_penalty = 0.75 ** source_counts.get(source, 0)

            base_score = 0.5 + novelty_bonus
            final_score = base_score * repetition_penalty
            source_counts[source] = source_counts.get(source, 0) + 1

            results.append({
                "item": item,
                "score": round(final_score, 3),
                "rationale": f"Diversity-weighted discovery | Source count penalty applied ({source})"
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

# ---------------------------------------------------------
# User Interface
# ---------------------------------------------------------
if "user_profile" not in st.session_state:
    st.session_state.user_profile = DEFAULT_PROFILE

# Sidebar: Profile & Controls
st.sidebar.title("Idem Identity Layer")
st.sidebar.caption("User-Controlled Preference Model")

algo_choice = st.sidebar.radio(
    "Active Recommendation Engine",
    ["Algorithm A: Direct Alignment", "Algorithm B: Frontier Discovery"],
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

# Main Feed Rendering
raw_items = load_data()

engine = DirectAlignmentEngine() if "Direct" in algo_choice else FrontierDiscoveryEngine()
ranked_results = engine.rank(raw_items, st.session_state.user_profile)

unique_sources_count = len({res["item"]["source"] for res in ranked_results})

st.title(f"Feed: {engine.name}")
st.write(f"*{engine.description}*")
st.caption(
    f"Evaluated {len(ranked_results)} candidate articles across {unique_sources_count} distinct sources."
)
st.divider()

for res in ranked_results[:20]:
    item = res["item"]

    # Sanitize and truncate summary
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
