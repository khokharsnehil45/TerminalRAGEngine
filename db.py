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
# VECTOR SIMILARITY SEARCH (Cosine Similarity)
# ==========================================

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def search_similar_chunks(query_vector: List[float], collection_id: Optional[int] = None, top_k: int = 4, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if collection_id:
        cursor.execute("""
            SELECT c.*, d.filename 
            FROM chunks c 
            JOIN documents d ON c.document_id = d.id 
            WHERE c.collection_id = ? AND c.embedding IS NOT NULL
        """, (collection_id,))
    else:
        cursor.execute("""
            SELECT c.*, d.filename 
            FROM chunks c 
            JOIN documents d ON c.document_id = d.id 
            WHERE c.embedding IS NOT NULL
        """)
        
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    q_vec = np.array(query_vector, dtype=np.float32)
    scored = []
    
    for r in rows:
        emb_data = json.loads(r["embedding"])
        c_vec = np.array(emb_data, dtype=np.float32)
        score = cosine_similarity(q_vec, c_vec)
        scored.append({
            "chunk_id": r["id"],
            "document_id": r["document_id"],
            "filename": r["filename"],
            "chunk_index": r["chunk_index"],
            "content": r["content"],
            "similarity_score": score
        })
        
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_k]

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
