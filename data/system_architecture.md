# System Architecture Overview

TRAG (Terminal RAG Engine) is a standalone, local vector intelligence system built in Python with SQLite, Rich, Questionary, and FlashRank.

## Key Capabilities
1. **Multi-Format Ingestion**: Ingests PDF, Markdown, Plain Text, Code (Python, Rust, JS, Go, C++), and Tabular datasets (CSV, TSV, Excel).
2. **Recursive Semantic Chunking**: Splits documents along logical paragraph and sentence boundaries with configurable overlap (default 800 chars / 150 overlap).
3. **3-Stage Hybrid Retrieval & Reranking**: Combines BM25 keyword matching and Dense Cosine Vector search via Reciprocal Rank Fusion (RRF), followed by a local Cross-Encoder (FlashRank) neural reranker.
4. **Local & Multi-Cloud Inference**: Feeds retrieved top-k context passages to local LLMs (`llama3.2:3b`, `mistral`) or Cloud providers (Gemini, OpenAI, Claude 3.5 Sonnet, Groq) with real-time token streaming.
5. **Knowledge Base Collections**: Isolates documents into custom domain bases (e.g. `legal`, `financial`, `tech_docs`, `research`).
6. **Shell Pipe Automation**: Integrates natively with Unix CLI pipelines (`cat error.log | TRAG "Explain this error"`).
