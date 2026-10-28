"""
worker.py — Background process that feeds PostgreSQL + pgvector independently of the UI.
"""
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
import time
from db import init_db, get_existing_urls, upsert_items
from sources import fetch_unified_catalog
from embeddings import embed_texts

# Common tracking / marketing parameters to strip
STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "feed", "rss", "fbclid", "gclid"
}

def canonicalize_url(url: str) -> str:
    """
    Strips tracking parameters and fragments without modifying schemes,
    subdomains, or path structures that could break links.
    """
    if not url:
        return ""
    
    parsed = urlparse(url.strip())
    
    # 1. Lowercase hostname only (RFC compliant, completely safe)
    netloc = parsed.netloc.lower()
    
    # 2. Filter query parameters to strip tracking keys
    clean_params = []
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            if k.lower() not in STRIP_PARAMS:
                clean_params.append((k, v))
                
    clean_query = urlencode(clean_params) if clean_params else ""
    
    # Reassemble: preserve scheme, path, params, query; drop #fragment
    return urlunparse((
        parsed.scheme,
        netloc,
        parsed.path,
        parsed.params,
        clean_query,
        ""
    ))

def run_ingestion_cycle():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting background crawl...")
    init_db()

    # 1. Fetch live feeds
    incoming_items = fetch_unified_catalog()
    if not incoming_items:
        print("No items fetched.")
        return

    # 2. Canonicalize and deduplicate within the incoming batch
    deduped_incoming = []
    seen_batch_urls = set()

    for it in incoming_items:
        clean_url = canonicalize_url(it.get("url", ""))
        if clean_url and clean_url not in seen_batch_urls:
            it["url"] = clean_url
            it["domain"] = urlparse(clean_url).netloc.lower()
            seen_batch_urls.add(clean_url)
            deduped_incoming.append(it)

    # 3. Filter out items already stored in Postgres
    existing_urls = get_existing_urls()
    new_items = [it for it in deduped_incoming if it["url"] not in existing_urls]
    print(f"Catalog fetched: {len(incoming_items)} items | Distinct: {len(deduped_incoming)} | Unindexed: {len(new_items)}")

    if not new_items:
        print("Database is up to date. Zero embeddings needed.")
        return

    # 4. Compute embeddings strictly for unindexed items
    corpus = [f"{it['title']}. {it.get('summary', '')}".strip() for it in new_items]
    print(f"Computing embeddings for {len(corpus)} items...")
    vectors = embed_texts(corpus)

    for idx, vec in enumerate(vectors):
        new_items[idx]["embedding"] = vec

    # 5. Upsert to Postgres
    upsert_items(new_items)
    print(f"Successfully indexed {len(new_items)} items into PostgreSQL.")

if __name__ == "__main__":
    run_ingestion_cycle()