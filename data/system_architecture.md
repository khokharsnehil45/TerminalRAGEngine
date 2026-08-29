# System Architecture Overview

TRAG (Terminal RAG Engine) is a standalone, local vector intelligence tool built in Python with SQLite, Rich, Questionary, and FastAPI.

## Key Capabilities
1. **Multi-Format Ingestion**: Ingests PDF, Markdown, Plain Text, Code (Python, Rust, JS, Go), and HTML files.
2. **Recursive Semantic Chunking**: Splits documents along logical paragraph and sentence boundaries with configurable overlap (default 800 chars / 150 overlap).
3. **Local Offline Vector Embeddings**: Uses `nomic-embed-text` locally via Ollama with Cosine Similarity vector search stored directly in `~/.trag_database.db`.
4. **Offline RAG Synthesis**: Feeds retrieved top-k context passages to local LLMs (`llama3.2:3b`) or Google Gemini Cloud.
5. **Knowledge Base Collections**: Isolate documents into custom bases (e.g. `legal`, `financial`, `tech_docs`, `research`).
6. **Dual Terminal & Web GUI**: Seamless terminal CLI paired with a minimalist dark dashboard at `http://localhost:8450`.
