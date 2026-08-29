# TRAG — Terminal RAG Engine (Local Vector Intelligence)

A fast, lightweight, and 100% private **Retrieval-Augmented Generation (RAG)** tool with a retro terminal CLI and minimalist Web GUI built with **FastAPI**, **SQLite Vector Storage**, **Ollama**, and **Rich**.

## ✨ Features

- 🔒 **100% Local & Private**: All document parsing, vector embeddings (`nomic-embed-text`), and LLM synthesis (`llama3.2:3b`) run completely on your machine.
- 📄 **Multi-Format Ingestion**: Ingests **PDFs**, **Markdown (`.md`)**, **Plain Text (`.txt`)**, **Code files (`.py`, `.js`, `.rs`, `.go`, `.cpp`)**, and HTML.
- 🧠 **Recursive Semantic Chunking**: Intelligently breaks long documents into paragraph and sentence chunks with configurable character limits and overlap.
- 📐 **Cosine Similarity Vector Search**: Instant vector search directly inside local SQLite database (`~/.trag_database.db`).
- 🗂️ **Knowledge Base Collections**: Create isolated bases (e.g. `legal`, `financial`, `research`, `codebase`).
- 💻 **Dual Interface (CLI & Web GUI)**:
  - Interactive Terminal CLI wizard with Rich panels & source citation tables.
  - Minimalist dark Web GUI dashboard with live RAG chat and side-by-side cited vector chunks canvas.
- 🔷 **Optional Gemini Cloud Mode**: Switch to Google Gemini (`gemini-2.5-flash` / `text-embedding-004`) for high-speed cloud intelligence.

---

## 🚀 How to Run

You can launch TRAG using either `trag` or `TRAG` from any terminal:

### 1. Interactive CLI Wizard
```bash
TRAG
```
*(or `trag`)*

### 2. Direct Single-Line Query
```bash
TRAG "What are the payment terms mentioned in contract.pdf?"
```

### 3. Launch Web GUI Dashboard
```bash
TRAG gui
```
Opens automatically at: **`http://localhost:8450`**
