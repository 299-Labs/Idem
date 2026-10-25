"""
sources.py — Data ingestion layer for RSS feeds and AT Protocol sources.
"""
from datetime import datetime, timezone
import re
import time
from typing import Any, Dict, List, Optional
import feedparser
import requests
import html

PUBLIC_APPVIEW = "https://public.api.bsky.app/xrpc"

# ---------------------------------------------------------
# Curated Technical Seed Sources
# ---------------------------------------------------------
RSS_FEEDS = {
    "Anthropic Research": "https://www.anthropic.com/feed.xml",
    "OpenAI News": "https://openai.com/news/rss.xml",
    "Simon Willison": "https://simonwillison.net/atom/everything/",
    "Cloudflare Blog": "https://blog.cloudflare.com/rss/",
    "Dan Luu": "https://danluu.com/atom.xml",
    "Hacker News": "https://news.ycombinator.com/rss",
    "Fly.io Blog": "https://fly.io/blog/feed.atom"
}

# Example AT Protocol technical feeds and lists
# You can paste either bsky.app web links or raw at:// URIs
ATPROTO_FEEDS = {
    # Custom Feeds (Feed Generators)
    "ArXiv ML Feed": "at://did:plc:bpuq5cgmyvssgi3iwsyvd4gn/app.bsky.feed.generator/arxiv-feed",
    "Terry's Starter Pack": "https://bsky.app/starter-pack/terrytangyuan.xyz/3lafppimwk42u",
}


# ---------------------------------------------------------
# AT Protocol Resolvers and Fetchers
# ---------------------------------------------------------

def resolve_handle_to_did(handle: str) -> str:
    """Resolves an AT Protocol handle to its DID."""
    url = f"{PUBLIC_APPVIEW}/com.atproto.identity.resolveHandle"
    res = requests.get(url, params={"handle": handle}, timeout=5)
    res.raise_for_status()
    return res.json()["did"]


def extract_list_uri_from_starterpack(starter_pack_uri: str) -> str:
    """
    Given an app.bsky.graph.starterpack AT-URI, calls getStarterPack
    and extracts the underlying app.bsky.graph.list AT-URI.
    """
    url = f"{PUBLIC_APPVIEW}/app.bsky.graph.getStarterPack"
    res = requests.get(url, params={"starterPack": starter_pack_uri}, timeout=8)
    res.raise_for_status()
    data = res.json().get("starterPack", {})
    
    # Extract underlying list URI
    list_uri = data.get("record", {}).get("list") or data.get("list", {}).get("uri")
    if not list_uri:
        raise ValueError(f"Could not find an underlying list in starter pack: {starter_pack_uri}")
    return list_uri


def resolve_identifier_to_target_feed(identifier: str) -> tuple[str, str]:
    """
    Takes any Bluesky URL or at:// URI and returns:
    (resolved_at_uri, feed_type) where feed_type is 'list' or 'feed'.
    """
    # 1. Handle Starter Pack web URL: https://bsky.app/starter-pack/<actor>/<rkey>
    sp_match = re.search(r"starter-pack/([^/]+)/([^/]+)", identifier)
    if sp_match:
        actor, rkey = sp_match.groups()
        did = resolve_handle_to_did(actor) if not actor.startswith("did:") else actor
        sp_at_uri = f"at://{did}/app.bsky.graph.starterpack/{rkey}"
        return extract_list_uri_from_starterpack(sp_at_uri), "list"

    # 2. Handle Starter Pack AT URI: at://<did>/app.bsky.graph.starterpack/<rkey>
    if "app.bsky.graph.starterpack" in identifier:
        return extract_list_uri_from_starterpack(identifier), "list"

    # 3. Handle Profile List or Feed web URLs
    profile_match = re.search(r"profile/([^/]+)/(lists|feed)/([^/]+)", identifier)
    if profile_match:
        actor, kind, rkey = profile_match.groups()
        did = resolve_handle_to_did(actor) if not actor.startswith("did:") else actor
        collection = "app.bsky.graph.list" if kind == "lists" else "app.bsky.feed.generator"
        feed_type = "list" if kind == "lists" else "feed"
        return f"at://{did}/{collection}/{rkey}", feed_type

    # 4. Handle raw at:// URIs
    if identifier.startswith("at://"):
        feed_type = "list" if "app.bsky.graph.list" in identifier else "feed"
        return identifier, feed_type

    raise ValueError(f"Unsupported AT Protocol source identifier: {identifier}")


def fetch_atproto_items(source_name: str, identifier: str, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Fetches and normalizes posts from a List, Feed, or Starter Pack.
    """
    items = []
    try:
        at_uri, feed_type = resolve_identifier_to_target_feed(identifier)

        endpoint = "app.bsky.feed.getListFeed" if feed_type == "list" else "app.bsky.feed.getFeed"
        param_name = "list" if feed_type == "list" else "feed"

        url = f"{PUBLIC_APPVIEW}/{endpoint}"
        res = requests.get(url, params={param_name: at_uri, "limit": limit}, timeout=10)
        res.raise_for_status()
        raw_feed = res.json().get("feed", [])

        for entry in raw_feed:
            post = entry.get("post", {})
            record = post.get("record", {})
            embed = post.get("embed", {})
            author = post.get("author", {})

            # Timestamp parsing
            created_at = record.get("createdAt", post.get("indexedAt"))
            ts = time.time()
            if created_at:
                try:
                    clean_iso = created_at.replace("Z", "+00:00")
                    ts = datetime.fromisoformat(clean_iso).timestamp()
                except Exception:
                    ts = time.time()

            # External link card detection
            external = None
            if embed.get("$type") == "app.bsky.embed.external#view":
                external = embed.get("external", {})
            elif embed.get("$type") == "app.bsky.embed.recordWithMedia#view":
                media = embed.get("media", {})
                if media.get("$type") == "app.bsky.embed.external#view":
                    external = media.get("external", {})

            handle = author.get("handle", "unknown")
            post_rkey = post.get("uri", "").split("/")[-1]
            post_permalink = f"https://bsky.app/profile/{handle}/post/{post_rkey}"

            if external and external.get("uri"):
                title = external.get("title") or f"Post by @{handle}"
                url = external.get("uri")
                summary = external.get("description") or record.get("text", "")
                is_external = True
            else:
                title = record.get("text", "").split("\n")[0][:100] or f"Post by @{handle}"
                url = post_permalink
                summary = record.get("text", "")
                is_external = False

            items.append({
                "id": post.get("cid", f"atproto_{abs(hash(url))}"),
                "source": f"{source_name} (@{handle})",
                "source_type": "atproto",
                "title": title.strip(),
                "url": url.strip(),
                "summary": summary.strip(),
                "published_ts": ts,
                "is_external": is_external,
                "metrics": {
                    "likes": post.get("likeCount", 0),
                    "reposts": post.get("repostCount", 0),
                    "replies": post.get("replyCount", 0)
                }
            })

    except Exception as e:
        print(f"Error fetching '{source_name}' ({identifier}): {e}")

    return items

# ---------------------------------------------------------
# RSS Fetcher (Normalized)
# ---------------------------------------------------------
def fetch_rss_items(source_name: str, feed_url: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Fetches and normalizes an RSS/Atom feed to match the unified schema."""
    items = []
    try:
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:limit]:
            link = entry.get("link", "")
            if not link:
                continue

            title = entry.get("title", "Untitled")
            raw_summary = entry.get("summary", entry.get("description", ""))
            summary = clean_html(raw_summary)

            ts = time.time()
            if entry.get("published_parsed"):
                ts = time.mktime(entry.published_parsed)
            elif entry.get("updated_parsed"):
                ts = time.mktime(entry.updated_parsed)

            items.append({
                "id": f"rss_{abs(hash(link))}",
                "source": source_name,
                "source_type": "rss",
                "title": title.strip(),
                "url": link.strip(),
                "summary": summary.strip(),
                "published_ts": ts,
                "is_external": True,
                "metrics": {
                    "likes": 0,
                    "reposts": 0,
                    "replies": 0
                }
            })
    except Exception as e:
        print(f"Error parsing RSS '{source_name}': {e}")

    return items

def fetch_unified_catalog():
    all_items = []

    # 1. Ingest RSS feeds
    for source_name, url in RSS_FEEDS.items():
        all_items.extend(fetch_rss_items(source_name, url, limit=10))

    # 2. Ingest AT Protocol feeds & lists
    for source_name, identifier in ATPROTO_FEEDS.items():
        all_items.extend(fetch_atproto_items(source_name, identifier, limit=25))

    return all_items

def clean_html(raw_html: str) -> str:
    """Strips all HTML tags, decodes HTML entities, and normalizes whitespace."""
    if not raw_html:
        return ""
    # Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
    text = html.unescape(raw_html)
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse multiple spaces and newlines into a single space
    return " ".join(text.split()).strip()