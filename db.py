"""
db.py — PostgreSQL + pgvector persistence and ANN retrieval layer.
"""
import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Explicitly load .env from the directory containing db.py
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing or empty.")

def get_connection():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    """Initializes the pgvector extension, table, and HNSW index."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 2. Candidate items table (384 dimensions for all-MiniLM-L6-v2)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    summary TEXT,
                    published_ts DOUBLE PRECISION NOT NULL,
                    domain TEXT,
                    likes INT DEFAULT 0,
                    reposts INT DEFAULT 0,
                    replies INT DEFAULT 0,
                    embedding vector(384) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Create HNSW index for sub-10ms Approximate Nearest Neighbor (ANN) search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS items_embedding_hnsw 
                ON items USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
        conn.commit()


def get_existing_urls() -> set:
    """Returns all URLs already stored in the database to prevent duplicate embedding compute."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM items;")
            return {row[0] for row in cur.fetchall()}


def upsert_items(items_with_embeddings: List[Dict[str, Any]]):
    """Batch inserts new items and embeddings, skipping duplicates."""
    if not items_with_embeddings:
        return

    records = []
    for it in items_with_embeddings:
        records.append((
            it["id"],
            it["source"],
            it.get("source_type", "rss"),
            it["title"],
            it["url"],
            it.get("summary", ""),
            float(it["published_ts"]),
            it.get("domain", ""),
            it.get("metrics", {}).get("likes", 0),
            it.get("metrics", {}).get("reposts", 0),
            it.get("metrics", {}).get("replies", 0),
            it["embedding"]  # numpy float32 array
        ))

    sql = """
        INSERT INTO items (
            id, source, source_type, title, url, summary, 
            published_ts, domain, likes, reposts, replies, embedding
        ) VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            likes = EXCLUDED.likes,
            reposts = EXCLUDED.reposts,
            replies = EXCLUDED.replies;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, records)
        conn.commit()


def retrieve_candidates(u_pos: np.ndarray, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Two-Tower Stage 1: Retrieves the top candidate articles using HNSW vector distance.
    Cosine distance operator in pgvector is <=> (1 - cosine_similarity).
    """
    sql = """
        SELECT 
            id, source, source_type, title, url, summary, 
            published_ts, domain, likes, reposts, replies,
            1 - (embedding <=> %s) AS vector_similarity,
            embedding
        FROM items
        ORDER BY embedding <=> %s ASC
        LIMIT %s;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (u_pos, u_pos, limit))
            rows = cur.fetchall()

    candidates = []
    for r in rows:
        candidates.append({
            "id": r[0],
            "source": r[1],
            "source_type": r[2],
            "title": r[3],
            "url": r[4],
            "summary": r[5],
            "published_ts": r[6],
            "domain": r[7],
            "metrics": {"likes": r[8], "reposts": r[9], "replies": r[10]},
            "vector_similarity": float(r[11]),
            "embedding": np.array(r[12], dtype=np.float32)
        })
    return candidates