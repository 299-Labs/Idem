import json
import math
import time
import feedparser
import streamlit as st

st.set_page_config(page_title="Idem — Sovereign Reading", page_icon="🧭", layout="wide")

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
# Content Seed Catalog
# ---------------------------------------------------------
FEEDS = {
    "Anthropic Research": "https://www.anthropic.com/feed.xml",
    "OpenAI News": "https://openai.com/news/rss.xml",
    "Simon Willison": "https://simonwillison.net/atom/everything/",
    "Cloudflare Blog": "https://blog.cloudflare.com/rss/",
    "Dan Luu": "https://danluu.com/atom.xml",
    "Hacker News": "https://news.ycombinator.com/rss",
    "Fly.io Blog": "https://fly.io/blog/feed.atom"
}

# ---------------------------------------------------------
# Ingestion & Normalization
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_catalog():
    items = []
    for source_name, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                link = entry.get("link", "")
                if not link:
                    continue
                
                title = entry.get("title", "Untitled")
                summary = entry.get("summary", entry.get("description", ""))
                
                # Parse timestamp
                ts = time.time()
                if entry.get("published_parsed"):
                    ts = time.mktime(entry.published_parsed)
                elif entry.get("updated_parsed"):
                    ts = time.mktime(entry.updated_parsed)
                    
                items.append({
                    "id": f"{source_name}_{abs(hash(link))}",
                    "source": source_name,
                    "title": title,
                    "url": link,
                    "summary": summary,
                    "published_ts": ts
                })
        except Exception:
            continue
    return items

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
raw_items = fetch_catalog()

engine = DirectAlignmentEngine() if "Direct" in algo_choice else FrontierDiscoveryEngine()
ranked_results = engine.rank(raw_items, st.session_state.user_profile)

st.title(f"Feed: {engine.name}")
st.write(f"*{engine.description}*")
st.caption(f"Evaluated {len(ranked_results)} candidate articles across {len(FEEDS)} independent sources.")
st.divider()

for res in ranked_results[:20]:
    item = res["item"]
    col_content, col_metric = st.columns([0.85, 0.15])
    
    with col_content:
        st.markdown(f"### [{item['title']}]({item['url']})")
        st.markdown(f"**{item['source']}** · `{res['rationale']}`")
        if item["summary"]:
            snippet = item["summary"][:260].replace("<p>", "").replace("</p>", "")
            st.write(snippet + "..." if len(item["summary"]) > 260 else snippet)
            
    with col_metric:
        st.metric("Score", res["score"])
        
    st.divider()