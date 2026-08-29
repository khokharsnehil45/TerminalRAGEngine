"""
TerminalRAGEngine (TRAG) - Local Vector Store & Document Chunk Database.
Uses SQLite to store documents, text chunks, vector embeddings, and collections.
"""

import json
import math
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

DB_PATH = Path.home() / ".trag_database.db"

def init_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Collections / Knowledge Bases
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL
    )
    """)
    
    # 2. Documents (PDFs, TXT, MD, Code)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        file_path TEXT,
        file_type TEXT,
        char_count INTEGER,
        chunk_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
    )
    """)
    
    # 3. Document Chunks & Embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        collection_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        char_length INTEGER NOT NULL,
        embedding TEXT, -- JSON array of floats
        created_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    )
    """)
    
    # 4. Query & Chat History
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        collection_id INTEGER,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        model_used TEXT,
        sources TEXT, -- JSON array of chunk references
        latency_seconds REAL,
        created_at TEXT NOT NULL
    )
    """)
    
    # Seed default collection if none exists
    cursor.execute("SELECT COUNT(*) FROM collections")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO collections (name, description, created_at) VALUES (?, ?, ?)",
            ("default", "Primary Local Knowledge Base", datetime.now().isoformat())
        )
        
    conn.commit()
    conn.close()

# ==========================================
# COLLECTIONS
# ==========================================

def get_collections(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.*, 
               (SELECT COUNT(*) FROM documents WHERE collection_id = c.id) as doc_count,
               (SELECT COUNT(*) FROM chunks WHERE collection_id = c.id) as chunk_count
        FROM collections c ORDER BY c.id ASC
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def create_collection(name: str, description: Optional[str] = None, db_path: Path = DB_PATH) -> Tuple[bool, str, Optional[int]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO collections (name, description, created_at) VALUES (?, ?, ?)",
            (name.strip().lower(), description or f"Knowledge base for {name}", datetime.now().isoformat())
        )
        cid = cursor.lastrowid
        conn.commit()
        conn.close()
        return True, "Collection created", cid
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Collection name already exists", None

def delete_collection(collection_id: int, db_path: Path = DB_PATH) -> bool:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE collection_id = ?", (collection_id,))
    cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))
    cursor.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

# ==========================================
# DOCUMENTS & CHUNKING
# ==========================================

def add_document(collection_id: int, filename: str, file_path: str, file_type: str, char_count: int, db_path: Path = DB_PATH) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (collection_id, filename, file_path, file_type, char_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (collection_id, filename, file_path, file_type, char_count, datetime.now().isoformat())
    )
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def insert_chunks(doc_id: int, collection_id: int, chunk_tuples: List[Tuple[int, str, List[float]]], db_path: Path = DB_PATH):
    """
    chunk_tuples: list of (chunk_index, content, embedding_vector)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    
    rows = []
    for idx, content, emb in chunk_tuples:
        emb_json = json.dumps(emb) if emb else None
        rows.append((doc_id, collection_id, idx, content, len(content), emb_json, now))
        
    cursor.executemany(
        "INSERT INTO chunks (document_id, collection_id, chunk_index, content, char_length, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    
    cursor.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (len(chunk_tuples), doc_id))
    conn.commit()
    conn.close()

def get_documents(collection_id: Optional[int] = None, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if collection_id:
        cursor.execute("SELECT * FROM documents WHERE collection_id = ? ORDER BY id DESC", (collection_id,))
    else:
        cursor.execute("SELECT * FROM documents ORDER BY id DESC")
        
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def delete_document(doc_id: int, db_path: Path = DB_PATH) -> bool:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

# ==========================================
# HYBRID RETRIEVAL (BM25 + Cosine Vector Fusion via RRF)
# ==========================================

import re
from rank_bm25 import BM25Okapi

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric keywords."""
    return re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())

def search_hybrid_chunks(query_str: str, query_vector: List[float], collection_id: Optional[int] = None, top_k: int = 4, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """
    State-of-the-Art Hybrid Search:
    Combines BM25 exact keyword matching with Dense Vector Cosine Similarity via Reciprocal Rank Fusion (RRF).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if collection_id:
        cursor.execute("""
            SELECT c.*, d.filename 
            FROM chunks c 
            JOIN documents d ON c.document_id = d.id 
            WHERE c.collection_id = ?
        """, (collection_id,))
    else:
        cursor.execute("""
            SELECT c.*, d.filename 
            FROM chunks c 
            JOIN documents d ON c.document_id = d.id 
        """)
        
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []

    # 1. Vector Cosine Similarity Ranking
    q_vec = np.array(query_vector, dtype=np.float32)
    vector_scores = []
    
    for r in rows:
        score = 0.0
        if r["embedding"]:
            emb_data = json.loads(r["embedding"])
            c_vec = np.array(emb_data, dtype=np.float32)
            score = cosine_similarity(q_vec, c_vec)
        vector_scores.append((r["id"], score, r))

    # Rank by vector score
    vector_scores.sort(key=lambda x: x[1], reverse=True)
    vector_rank = {item[0]: rank + 1 for rank, item in enumerate(vector_scores)}

    # 2. BM25 Keyword Search Ranking
    tokenized_corpus = [tokenize_text(r["content"]) for r in rows]
    query_tokens = tokenize_text(query_str)
    
    bm25_rank = {}
    if query_tokens and any(tokenized_corpus):
        try:
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(query_tokens)
            bm25_scores = [(rows[i]["id"], doc_scores[i], rows[i]) for i in range(len(rows))]
            bm25_scores.sort(key=lambda x: x[1], reverse=True)
            bm25_rank = {item[0]: rank + 1 for rank, item in enumerate(bm25_scores)}
        except Exception:
            bm25_rank = {r["id"]: 999 for r in rows}
    else:
        bm25_rank = {r["id"]: 999 for r in rows}

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (60 + VectorRank) + 1 / (60 + BM25Rank)
    k_rrf = 60.0
    combined = []
    
    row_map = {r["id"]: r for r in rows}
    for chunk_id, r in row_map.items():
        v_r = vector_rank.get(chunk_id, 999)
        b_r = bm25_rank.get(chunk_id, 999)
        
        # Calculate RRF score
        rrf_score = (1.0 / (k_rrf + v_r)) + (1.0 / (k_rrf + b_r))
        
        # Retrieve raw vector similarity for display
        raw_vec_score = next((x[1] for x in vector_scores if x[0] == chunk_id), 0.0)
        
        combined.append({
            "chunk_id": r["id"],
            "document_id": r["document_id"],
            "filename": r["filename"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "similarity_score": raw_vec_score,
            "rrf_score": rrf_score,
            "vector_rank": v_r,
            "bm25_rank": b_r
        })

    # Sort by highest combined RRF fusion score
    combined.sort(key=lambda x: x["rrf_score"], reverse=True)
    return combined[:top_k]

def search_similar_chunks(query_vector: List[float], collection_id: Optional[int] = None, top_k: int = 4, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Legacy vector-only search fallback."""
    return search_hybrid_chunks("", query_vector, collection_id=collection_id, top_k=top_k, db_path=db_path)

# ==========================================
# QUERY HISTORY
# ==========================================

def log_query(query: str, response: str, model_used: str, sources: List[Dict[str, Any]], latency: float, collection_id: Optional[int] = None, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO queries (collection_id, query, response, model_used, sources, latency_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (collection_id, query, response, model_used, json.dumps(sources), latency, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_recent_queries(limit: int = 10, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
