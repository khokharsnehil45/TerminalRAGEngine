"""
TRAG Web & Desktop GUI Server (FastAPI).
Serves document management, live RAG chat, collection switching, and vector analytics.
"""

import os
import shutil
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

import db
import rag_engine

app = FastAPI(title="TRAG GUI", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "web"
STATIC_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = Path(__file__).parent / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

# Models
class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class SessionCreate(BaseModel):
    title: str
    collection_id: Optional[int] = 1

class QueryRequest(BaseModel):
    query: str
    collection_id: Optional[int] = None
    top_k: Optional[int] = 4
    session_id: Optional[int] = None

# API Endpoints
@app.get("/api/sessions")
def list_sessions(collection_id: Optional[int] = None):
    return db.get_sessions(collection_id=collection_id)

@app.post("/api/sessions")
def create_session(data: SessionCreate):
    sid = db.create_session(data.title, collection_id=data.collection_id or 1)
    return {"success": True, "id": sid, "title": data.title}

@app.get("/api/sessions/{sid}/messages")
def get_session_messages(sid: int):
    return db.get_session_messages(sid)

@app.delete("/api/sessions/{sid}")
def delete_session(sid: int):
    success = db.delete_session(sid)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}

@app.get("/api/collections")
def list_collections():
    return db.get_collections()

@app.post("/api/collections")
def create_collection(data: CollectionCreate):
    success, msg, cid = db.create_collection(data.name, data.description)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "id": cid, "name": data.name}

@app.delete("/api/collections/{cid}")
def delete_collection(cid: int):
    success = db.delete_collection(cid)
    if not success:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"success": True}

@app.get("/api/documents")
def list_documents(collection_id: Optional[int] = None):
    return db.get_documents(collection_id=collection_id)

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    success = db.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), collection_id: int = Form(1)):
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        res = rag_engine.ingest_file(save_path, collection_id=collection_id)
        return {"success": True, "document": res}
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

@app.post("/api/query")
def execute_query(data: QueryRequest):
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        res = rag_engine.query_rag(data.query, collection_id=data.collection_id, top_k=data.top_k or 4, session_id=data.session_id)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/stream")
def execute_query_stream(data: QueryRequest):
    if not data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    import json
    def event_generator():
        for chunk in rag_engine.stream_query_rag(data.query, collection_id=data.collection_id, top_k=data.top_k or 4, session_id=data.session_id):
            yield f"data: {json.dumps(chunk)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/history")
def get_history(limit: int = 15):
    return db.get_recent_queries(limit=limit)

@app.get("/api/config")
def get_config():
    return rag_engine.load_config()

@app.post("/api/config")
def save_config(cfg: Dict[str, Any]):
    rag_engine.save_config(cfg)
    return {"success": True}

# Serve SPA
@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def launch_server(port: int = 8450, open_browser: bool = True):
    db.init_db()
    url = f"http://localhost:{port}"
    print(f"\n🚀 TRAG GUI launched at: {url}\n")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    launch_server()
