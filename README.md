```text
████████╗██████╗  █████╗  ██████╗ 
╚══██╔══╝██╔══██╗██╔══██╗██╔════╝ 
   ██║   ██████╔╝███████║██║  ███╗
   ██║   ██╔══██╗██╔══██║██║   ██║
   ██║   ██║  ██║██║  ██║╚██████╔╝
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
  ⚡ Local Retrieval-Augmented Generation & Vector Intelligence ⚡
```

<div align="center">

# TRAG — Terminal RAG Engine

**Production-Grade, 100% Offline Document & Tabular Vector Intelligence**  
*Grounded Local RAG with Hybrid BM25 Search, Dense Vector Embeddings, Conversational Memory & Anti-Hallucination Guardrails.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-black.svg?logo=ollama&logoColor=white)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview

**TRAG (Terminal RAG Engine)** is a standalone, privacy-first Retrieval-Augmented Generation (RAG) system engineered for developers, researchers, and data teams who need fast, offline search and Q&A over local documents and structured datasets.

Unlike basic RAG prototypes that rely solely on fuzzy vector embeddings and suffer from hallucinations, TRAG incorporates **State-of-the-Art Hybrid Retrieval (BM25 + Cosine Vectors via Reciprocal Rank Fusion)**, row-by-row **Tabular Dataset Understanding**, **Multi-Turn Session Memory**, and **Dynamic Confidence Guardrails**.

---

## ✨ Key Architectural Features

### 1. 🔀 SOTA Hybrid Retrieval (BM25 + Dense Vectors with RRF)
TRAG scores and fuses candidate passages using **Reciprocal Rank Fusion (RRF)**:
$$\text{RRF Score} = \frac{1}{60 + \text{Rank}_{\text{Vector}}} + \frac{1}{60 + \text{Rank}_{\text{BM25}}}$$
- **Exact Keyword Precision (BM25)**: Accurately matches error codes, variable names, serial numbers, names, and transaction IDs.
- **Semantic Understanding (Vectors)**: Captures concepts, ideas, and contextual nuance using local `nomic-embed-text` embeddings.

### 2. 📊 Deep Tabular Dataset Parsing (CSV, Excel, JSON)
- **Spreadsheets (`.csv`, `.tsv`, `.xlsx`, `.xls`)**: Parsed row-by-row into semantic key-value records with Markdown table summaries.
- **Multi-Sheet Workbooks**: Each Excel sheet is extracted and indexed independently with row identification tags.
- **JSON Data**: Auto-detects structured arrays of objects and converts them into searchable records.

### 3. 🧠 Multi-Turn Conversational Memory & Sessions
- Retains context across multiple dialogue turns in both CLI and Web GUI.
- Allows natural follow-up questions (e.g. asking *"Why is it better?"* understands *"it"* refers to the optimizer mentioned in the previous turn).

### 4. 🛡️ Anti-Hallucination Guardrails & Telemetry
- **Dynamic Confidence Scoring (0-100%)**: Evaluates top-k retrieval affinity. If confidence is critically low, TRAG intercepts the query safely instead of fabricating false facts.
- **Observability Telemetry**: Displays latency breakdowns on every response (`Embedding ms`, `Retrieval ms`, `LLM ms`, and `Total ms`).

### 5. 📂 Retro Terminal File Explorer & Dual Interfaces
- **Interactive Terminal CLI (`TRAG` / `trag`)**: Dual-tone Electric Amber banner, Questionary interactive menus, and built-in interactive folder navigator.
- **Minimalist Dark Web GUI (`TRAG gui`)**: 3-pane responsive dashboard with live typewriter token streaming (Server-Sent Events) and real-time cited vector chunks canvas.

---

## 🛠️ Supported File Formats

| Format | Extension | Processing Pipeline |
| :--- | :--- | :--- |
| **PDF Documents** | `.pdf` | Multi-page text extraction & recursive sentence chunking |
| **CSV / TSV** | `.csv`, `.tsv` | Row-by-row structured records & summary tables |
| **Excel Spreadsheets** | `.xlsx`, `.xls` | Multi-sheet parsing with row key-value serialization |
| **JSON Datasets** | `.json` | Array-of-objects tabular normalization |
| **Markdown & Notes** | `.md`, `.txt` | Paragraph boundary splitting with character overlap |
| **Codebases** | `.py`, `.js`, `.ts`, `.rs`, `.go`, `.cpp`, `.html` | Syntax-aware code chunking |

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10+** installed and [Ollama](https://ollama.com) running locally:
```bash
# Pull local embedding and LLM models
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

### 2. Clone and Install
```bash
git clone https://github.com/khokharsnehil45/TerminalRAGEngine.git
cd TerminalRAGEngine

# Install dependencies
pip install -r requirements.txt

# Make launcher executable and link to ~/.local/bin
chmod +x trag
ln -sf $(pwd)/trag ~/.local/bin/TRAG
ln -sf $(pwd)/trag ~/.local/bin/trag
```

---

## 💻 Usage Guide

### 1. Interactive Terminal CLI
Launch the interactive terminal interface:
```bash
TRAG
```
*(or `trag`)*

**Menu Capabilities:**
- `💬 Ask / RAG Chat`: Multi-turn conversational research session with cited source tables.
- `📥 Ingest & Embed Document`: Select files to parse and embed into your vector database.
- `📂 Terminal File Explorer`: Interactive retro folder navigator to browse and ingest files.
- `📜 View Ingested Documents`: Inspect stored documents, character counts, and vector chunk counts.
- `🗂️ Knowledge Base Manager`: Create and switch between isolated domain collections (e.g. `finance`, `legal`, `codebase`).
- `💻 Launch TRAG Web GUI`: Starts the local browser dashboard.

---

### 2. Direct Single-Line Terminal Query
Ask quick questions directly from bash without opening the interactive menu:
```bash
TRAG "What is the transaction status for Amit Verma in sales_records.csv?"
```

---

### 3. Launch Web GUI Dashboard
Launch the minimalist browser dashboard:
```bash
TRAG gui
```
Opens automatically in your browser at: **`http://localhost:8450`**

---

## 🐳 Docker Deployment

TRAG includes a multi-stage production `Dockerfile` and `docker-compose.yml`:

```bash
# Start TRAG with persistent volume mounts
sudo docker compose up -d --build

# Verify container status
sudo docker compose ps

# Check production health endpoint
curl http://localhost:8450/healthz
```

---

## ⚙️ Configuration

TRAG automatically stores settings in `~/.trag_config.json`:

```json
{
  "embedding_provider": "ollama",
  "ollama_host": "http://localhost:11434",
  "embedding_model": "nomic-embed-text",
  "llm_provider": "ollama",
  "ollama_llm_model": "llama3.2:3b",
  "gemini_api_key": "",
  "gemini_model": "gemini-2.5-flash",
  "chunk_size": 800,
  "chunk_overlap": 150
}
```

---

## 📄 License
Released under the [MIT License](LICENSE). Built for high-speed, 100% private document intelligence.
