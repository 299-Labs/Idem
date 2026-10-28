"""
worker.py — Background process that feeds PostgreSQL + pgvector independently of the UI.
"""
from urllib.parse import urlparse
import time
from db import init_db, get_existing_urls, upsert_items
from sources import fetch_unified_catalog
from embeddings import embed_texts

def run_ingestion_cycle():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting background crawl...")
    init_db()

    # 1. Fetch live feeds
    incoming_items = fetch_unified_catalog()
    if not incoming_items:
        print("No items fetched.")
        return

    # 2. Filter out already-indexed items
    existing_urls = get_existing_urls()
    new_items = [it for it in incoming_items if it["url"] not in existing_urls]
    print(f"Found {len(incoming_items)} total items ({len(new_items)} new).")

    if not new_items:
        print("Database is up to date.")
        return

    # 3. Add domain metadata
    for it in new_items:
        it["domain"] = urlparse(it["url"]).netloc.lower()

    # 4. Compute embeddings strictly for new items
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