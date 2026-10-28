"""
sources.py — Fast, parallel ingestion for RSS and AT Protocol feeds.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
import html
import re
import time
from typing import Any, Dict, List, Tuple
import feedparser
import requests

PUBLIC_APPVIEW = "https://public.api.bsky.app/xrpc"

RSS_FEEDS = {
    "Anthropic Research": "https://www.anthropic.com/feed.xml",
    "OpenAI News": "https://openai.com/news/rss.xml",
    "Simon Willison": "https://simonwillison.net/atom/everything/",
    "Cloudflare Blog": "https://blog.cloudflare.com/rss/",
    "Dan Luu": "https://danluu.com/atom.xml",
    "Hacker News": "https://news.ycombinator.com/rss",
    "Fly.io Blog": "https://fly.io/blog/feed.atom",
}

ATPROTO_FEEDS = {
    "ArXiv ML Feed": "at://did:plc:bpuq5cgmyvssgi3iwsyvd4gn/app.bsky.feed.generator/arxiv-feed",
    "Terry's Starter Pack": "https://bsky.app/starter-pack/terrytangyuan.xyz/3lafppimwk42u",
}

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = html.unescape(raw_html)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split()).strip()

@lru_cache(maxsize=128)
def resolve_handle_to_did(handle: str) -> str:
    url = f"{PUBLIC_APPVIEW}/com.atproto.identity.resolveHandle"
    res = requests.get(url, params={"handle": handle}, timeout=4)
    res.raise_for_status()
    return res.json()["did"]

@lru_cache(maxsize=64)
def extract_list_uri_from_starterpack(starter_pack_uri: str) -> str:
    url = f"{PUBLIC_APPVIEW}/app.bsky.graph.getStarterPack"
    res = requests.get(url, params={"starterPack": starter_pack_uri}, timeout=5)
    res.raise_for_status()
    data = res.json().get("starterPack", {})
    list_uri = data.get("record", {}).get("list") or data.get("list", {}).get("uri")
    if not list_uri:
        raise ValueError(f"No list found in starter pack: {starter_pack_uri}")
    return list_uri

def resolve_identifier_to_target_feed(identifier: str) -> Tuple[str, str]:
    sp_match = re.search(r"starter-pack/([^/]+)/([^/]+)", identifier)
    if sp_match:
        actor, rkey = sp_match.groups()
        did = resolve_handle_to_did(actor) if not actor.startswith("did:") else actor
        return extract_list_uri_from_starterpack(f"at://{did}/app.bsky.graph.starterpack/{rkey}"), "list"
    if "app.bsky.graph.starterpack" in identifier:
        return extract_list_uri_from_starterpack(identifier), "list"
    profile_match = re.search(r"profile/([^/]+)/(lists|feed)/([^/]+)", identifier)
    if profile_match:
        actor, kind, rkey = profile_match.groups()
        did = resolve_handle_to_did(actor) if not actor.startswith("did:") else actor
        collection = "app.bsky.graph.list" if kind == "lists" else "app.bsky.feed.generator"
        feed_type = "list" if kind == "lists" else "feed"
        return f"at://{did}/{collection}/{rkey}", feed_type
    if identifier.startswith("at://"):
        feed_type = "list" if "app.bsky.graph.list" in identifier else "feed"
        return identifier, feed_type
    raise ValueError(f"Unsupported identifier: {identifier}")

def fetch_rss_items(source_name: str, feed_url: str, limit: int = 15) -> List[Dict[str, Any]]:
    items = []
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(feed_url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return items
        parsed = feedparser.parse(resp.content)
        for entry in parsed.entries[:limit]:
            link = entry.get("link", "").strip()
            if not link:
                continue
            raw_summary = entry.get("summary", entry.get("description", ""))
            ts = time.time()
            if entry.get("published_parsed"):
                ts = time.mktime(entry.published_parsed)
            elif entry.get("updated_parsed"):
                ts = time.mktime(entry.updated_parsed)
            items.append({
                "id": f"rss_{abs(hash(link))}",
                "source": source_name,
                "source_type": "rss",
                "title": clean_html(entry.get("title", "Untitled")),
                "url": link,
                "summary": clean_html(raw_summary),
                "published_ts": ts,
                "is_external": True,
                "metrics": {"likes": 0, "reposts": 0, "replies": 0}
            })
    except Exception as e:
        print(f"[RSS Warning] Failed to fetch {source_name}: {e}")
    return items

def fetch_atproto_items(source_name: str, identifier: str, limit: int = 25) -> List[Dict[str, Any]]:
    items = []
    try:
        at_uri, feed_type = resolve_identifier_to_target_feed(identifier)
        endpoint = "app.bsky.feed.getListFeed" if feed_type == "list" else "app.bsky.feed.getFeed"
        param_name = "list" if feed_type == "list" else "feed"
        url = f"{PUBLIC_APPVIEW}/{endpoint}"
        res = requests.get(url, params={param_name: at_uri, "limit": limit}, timeout=6)
        if res.status_code != 200:
            return items
        for entry in res.json().get("feed", []):
            post = entry.get("post", {})
            record = post.get("record", {})
            embed = post.get("embed", {})
            author = post.get("author", {})
            created_at = record.get("createdAt", post.get("indexedAt"))
            ts = time.time()
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
                except Exception:
                    pass
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
                "title": clean_html(title),
                "url": url.strip(),
                "summary": clean_html(summary),
                "published_ts": ts,
                "is_external": is_external,
                "metrics": {
                    "likes": post.get("likeCount", 0),
                    "reposts": post.get("repostCount", 0),
                    "replies": post.get("replyCount", 0)
                }
            })
    except Exception as e:
        print(f"[ATProto Warning] Failed to fetch {source_name}: {e}")
    return items

def fetch_unified_catalog() -> List[Dict[str, Any]]:
    all_items = []
    tasks = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for name, url in RSS_FEEDS.items():
            tasks.append(executor.submit(fetch_rss_items, name, url, 10))
        for name, ident in ATPROTO_FEEDS.items():
            tasks.append(executor.submit(fetch_atproto_items, name, ident, 25))
        for future in as_completed(tasks):
            try:
                res = future.result()
                if res:
                    all_items.extend(res)
            except Exception as e:
                print(f"[Parallel Ingest Error] {e}")
    return all_items
