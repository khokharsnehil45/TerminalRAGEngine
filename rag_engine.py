"""
TerminalRAGEngine (TRAG) - Document ingestion, recursive chunking, and semantic vector engine.
Supports PDF, Markdown, Plain Text, Code files (Python, JS, Go, Rust, C++), and HTML.
"""

import os
import re
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pypdf
import db

CONFIG_FILE = Path.home() / ".trag_config.json"

DEFAULT_CONFIG = {
    "embedding_provider": "ollama", # "ollama" or "gemini"
    "ollama_host": "http://localhost:11434",
    "embedding_model": "nomic-embed-text",
    "llm_provider": "ollama",
    "ollama_llm_model": "llama3.2:3b",
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash",
    "chunk_size": 800,
    "chunk_overlap": 150
}

def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return {**DEFAULT_CONFIG, **cfg}
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# ==========================================
# DOCUMENT & TABULAR EXTRACTION
# ==========================================

import csv
import pandas as pd

def extract_text_from_file(file_path: Path) -> Tuple[str, str]:
    """Returns (extracted_text, file_type) with dedicated tabular format handling."""
    suffix = file_path.suffix.lower()
    
    # 1. Tabular: CSV / TSV
    if suffix in [".csv", ".tsv"]:
        try:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(str(file_path), sep=sep, low_memory=False)
            
            # Format row-by-row semantic representation
            text = f"Dataset: {file_path.name}\nTotal Rows: {len(df)} | Columns: {', '.join(df.columns.astype(str))}\n\n"
            
            # Markdown table summary overview
            text += "### Summary Sample Table:\n"
            text += df.head(10).to_markdown(index=False) + "\n\n"
            
            # Semantic record-by-record representation
            text += "### Itemized Records:\n"
            for idx, row in df.iterrows():
                row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                text += f"[Row {idx + 1}] {row_str}\n"
                
            return text.strip(), "csv"
        except Exception as e:
            raise RuntimeError(f"Failed to read tabular file {file_path.name}: {e}")

    # 2. Tabular: Excel (XLSX / XLS)
    elif suffix in [".xlsx", ".xls"]:
        try:
            excel = pd.ExcelFile(str(file_path))
            text = f"Excel Workbook: {file_path.name}\nSheets: {', '.join(excel.sheet_names)}\n\n"
            
            for sheet_name in excel.sheet_names:
                df = excel.parse(sheet_name)
                text += f"## Sheet: {sheet_name} ({len(df)} rows)\n"
                text += f"Columns: {', '.join(df.columns.astype(str))}\n\n"
                
                # Sample table
                if not df.empty:
                    text += df.head(8).to_markdown(index=False) + "\n\n"
                    text += "### Rows:\n"
                    for idx, row in df.iterrows():
                        row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                        text += f"[{sheet_name} - Row {idx + 1}] {row_str}\n"
                text += "\n---\n\n"
                
            return text.strip(), "excel"
        except Exception as e:
            raise RuntimeError(f"Failed to read Excel spreadsheet {file_path.name}: {e}")

    # 3. Tabular: JSON Array or Object
    elif suffix == ".json":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                text = f"JSON Dataset: {file_path.name} ({len(df)} records)\n\n"
                text += df.head(10).to_markdown(index=False) + "\n\n"
                text += "### Structured Entries:\n"
                for idx, row in df.iterrows():
                    row_str = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    text += f"[Entry {idx + 1}] {row_str}\n"
                return text.strip(), "json_table"
            else:
                return json.dumps(data, indent=2), "json"
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON file {file_path.name}: {e}")

    # 4. PDF Documents
    elif suffix == ".pdf":
        text = ""
        try:
            reader = pypdf.PdfReader(str(file_path))
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text += f"\n--- Page {i+1} ---\n" + extracted
            return text.strip(), "pdf"
        except Exception as e:
            raise RuntimeError(f"Failed to read PDF {file_path.name}: {e}")
            
    # 5. Plain Text, Markdown & Code
    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            ftype = suffix.replace(".", "") if suffix else "txt"
            return content.strip(), ftype
        except Exception as e:
            raise RuntimeError(f"Failed to read file {file_path.name}: {e}")

# ==========================================
# RECURSIVE CHUNKING
# ==========================================

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    """Splits text cleanly using paragraph boundaries, newlines, and sentence breaks."""
    if not text:
        return []
        
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        if len(current_chunk) + len(p) <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If paragraph itself is too large, split by sentences
            if len(p) > chunk_size:
                sentences = re.split(r'(?<=[.?!])\s+', p)
                sub_chunk = ""
                for s in sentences:
                    if len(sub_chunk) + len(s) <= chunk_size:
                        sub_chunk += (" " if sub_chunk else "") + s
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = s
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = p
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

# ==========================================
# EMBEDDINGS (Local Ollama / Gemini Cloud)
# ==========================================

def get_embedding(text: str, config: Optional[Dict[str, Any]] = None) -> List[float]:
    cfg = config or load_config()
    provider = cfg.get("embedding_provider", "ollama")
    
    if provider == "ollama":
        host = cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
        model = cfg.get("embedding_model", "nomic-embed-text")
        url = f"{host}/api/embeddings"
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("embedding", [])
        except Exception as e:
            # Fallback simple hash embedding if server is offline
            raise RuntimeError(f"Ollama embedding request failed ({model} at {host}): {e}")
            
    elif provider == "gemini":
        api_key = cfg.get("gemini_api_key", "").strip()
        if not api_key:
            raise ValueError("Gemini API Key is missing in settings")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        payload = json.dumps({
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]}
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]["values"]
            
    raise ValueError(f"Unknown embedding provider: {provider}")

# ==========================================
# INGESTION PIPELINE
# ==========================================

def ingest_file(file_path: Path, collection_id: int, config: Optional[Dict[str, Any]] = None, progress_callback=None) -> Dict[str, Any]:
    cfg = config or load_config()
    raw_text, ftype = extract_text_from_file(file_path)
    
    if not raw_text:
        raise ValueError(f"File {file_path.name} contains no readable text.")
        
    chunks = chunk_text(raw_text, chunk_size=cfg.get("chunk_size", 800), chunk_overlap=cfg.get("chunk_overlap", 150))
    doc_id = db.add_document(collection_id, file_path.name, str(file_path.resolve()), ftype, len(raw_text))
    
    chunk_tuples = []
    total = len(chunks)
    
    for i, ch in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, total, file_path.name)
        try:
            emb = get_embedding(ch, cfg)
        except Exception as e:
            emb = None
        chunk_tuples.append((i, ch, emb))
        
    db.insert_chunks(doc_id, collection_id, chunk_tuples)
    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "char_count": len(raw_text),
        "chunk_count": len(chunks)
    }

# ==========================================
# RAG RETRIEVAL & SYNTHESIS
# ==========================================

def query_rag(query_str: str, collection_id: Optional[int] = None, top_k: int = 4, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    import time
    start_t = time.time()
    
    # 1. Embed query
    q_vec = get_embedding(query_str, cfg)
    
    # 2. Retrieve top-k context chunks using Hybrid BM25 + Vector Fusion
    sources = db.search_hybrid_chunks(query_str, q_vec, collection_id=collection_id, top_k=top_k)
    
    if not sources:
        context_block = "No relevant context found in the local knowledge base."
    else:
        context_parts = []
        for s in sources:
            context_parts.append(f"[Source: {s['filename']} (Score: {s['similarity_score']:.2f})]\n{s['content']}")
        context_block = "\n\n---\n\n".join(context_parts)
        
    # 3. Generate Augmented Prompt
    system_prompt = (
        "You are TRAG (Terminal RAG Engine), an expert offline document and tabular data intelligence. "
        "Analyze the provided document passages and structured datasets (spreadsheets, tables, records, PDFs, code). "
        "Extract specific factual values, numbers, names, rows, and relationships accurately from the context. "
        "Always cite the source document name when answering. If the context does not contain the answer, state that clearly."
    )
    
    prompt = f"""### Context Documents & Tabular Datasets:
{context_block}

### User Question:
{query_str}

Detailed, direct & accurate answer:"""

    # 4. Invoke LLM (Ollama or Gemini)
    llm_provider = cfg.get("llm_provider", "ollama")
    response_text = ""
    model_name = ""
    
    if llm_provider == "ollama":
        host = cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
        model_name = cfg.get("ollama_llm_model", "llama3.2:3b")
        url = f"{host}/api/generate"
        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "").strip()
        except Exception as e:
            response_text = f"❌ Ollama LLM Error ({model_name}): {e}\nEnsure Ollama is running (`ollama serve`)."
            
    elif llm_provider == "gemini":
        api_key = cfg.get("gemini_api_key", "").strip()
        model_name = cfg.get("gemini_model", "gemini-2.5-flash")
        if not api_key:
            response_text = "❌ Gemini API key is missing. Configure it in Settings."
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]
            }).encode("utf-8")
            
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception as e:
                response_text = f"❌ Gemini API Error: {e}"
                
    latency = time.time() - start_t
    db.log_query(query_str, response_text, f"{llm_provider}:{model_name}", sources, latency, collection_id=collection_id)
    
    return {
        "query": query_str,
        "response": response_text,
        "sources": sources,
        "latency": latency,
        "model_used": f"{llm_provider}:{model_name}"
    }

def stream_query_rag(query_str: str, collection_id: Optional[int] = None, top_k: int = 4, config: Optional[Dict[str, Any]] = None):
    """
    Generator that yields JSON chunks for server-sent events (SSE) and live terminal streaming.
    Yields:
      {"type": "sources", "sources": [...]}
      {"type": "token", "token": "..."}
      {"type": "done", "latency": float, "model": str}
    """
    cfg = config or load_config()
    import time
    start_t = time.time()
    
    # 1. Embed query & retrieve context via Hybrid BM25 + Vector RRF
    q_vec = get_embedding(query_str, cfg)
    sources = db.search_hybrid_chunks(query_str, q_vec, collection_id=collection_id, top_k=top_k)
    
    yield {"type": "sources", "sources": sources}
    
    if not sources:
        context_block = "No relevant context found in the local knowledge base."
    else:
        context_parts = [f"[Source: {s['filename']} (Score: {s['similarity_score']:.2f})]\n{s['content']}" for s in sources]
        context_block = "\n\n---\n\n".join(context_parts)
        
    system_prompt = (
        "You are TRAG (Terminal RAG Engine), an expert offline document and tabular data intelligence. "
        "Analyze the provided document passages and structured datasets (spreadsheets, tables, records, PDFs, code). "
        "Extract specific factual values, numbers, names, rows, and relationships accurately from the context. "
        "Always cite the source document name when answering. If the context does not contain the answer, state that clearly."
    )
    prompt = f"### Context Documents & Tabular Datasets:\n{context_block}\n\n### User Question:\n{query_str}\n\nDetailed, direct & accurate answer:"
    
    llm_provider = cfg.get("llm_provider", "ollama")
    full_response = ""
    model_name = ""
    
    if llm_provider == "ollama":
        host = cfg.get("ollama_host", "http://localhost:11434").rstrip("/")
        model_name = cfg.get("ollama_llm_model", "llama3.2:3b")
        url = f"{host}/api/generate"
        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True
        }).encode("utf-8")
        
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        full_response += token
                        yield {"type": "token", "token": token}
        except Exception as e:
            err_msg = f"\n❌ Ollama Stream Error: {e}"
            full_response += err_msg
            yield {"type": "token", "token": err_msg}
            
    elif llm_provider == "gemini":
        api_key = cfg.get("gemini_api_key", "").strip()
        model_name = cfg.get("gemini_model", "gemini-2.5-flash")
        if not api_key:
            err_msg = "❌ Gemini API Key is missing."
            full_response += err_msg
            yield {"type": "token", "token": err_msg}
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?key={api_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    for item in data:
                        token = item["candidates"][0]["content"]["parts"][0]["text"]
                        full_response += token
                        yield {"type": "token", "token": token}
            except Exception as e:
                err_msg = f"❌ Gemini Error: {e}"
                full_response += err_msg
                yield {"type": "token", "token": err_msg}

    latency = time.time() - start_t
    db.log_query(query_str, full_response, f"{llm_provider}:{model_name}", sources, latency, collection_id=collection_id)
    yield {"type": "done", "latency": latency, "model": f"{llm_provider}:{model_name}"}
