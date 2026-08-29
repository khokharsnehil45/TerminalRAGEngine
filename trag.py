#!/usr/bin/env python3
"""
TRAG (Terminal RAG Engine) - Standalone Offline Document Vector Intelligence CLI.
Retro dual-tone cyberpunk design with interactive document ingestion, vector searches, and RAG chat.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.markdown import Markdown

import db
import rag_engine

console = Console()

CUSTOM_STYLE = Style([
    ('qmark', 'fg:#00e5ff bold'),
    ('question', 'bold fg:#00e5ff'),
    ('answer', 'fg:#50fa7b bold'),
    ('pointer', 'fg:#ff79c6 bold'),
    ('highlighted', 'fg:#ff79c6 bold'),
    ('selected', 'fg:#50fa7b bold'),
    ('separator', 'fg:#6272a4'),
    ('instruction', 'fg:#8be9fd italic'),
    ('text', 'fg:#f8f8f2'),
])

ACTIVE_COLLECTION = {"id": 1, "name": "default"}

def render_banner(subtitle: str = "⚡ Local Retrieval-Augmented Generation & Vector Intelligence ⚡"):
    """Renders the retro 3D-styled TRAG banner with Electric Amber & Sunset Gold gradient."""
    banner_lines = [
        r"████████╗██████╗  █████╗  ██████╗ ",
        r"╚══██╔══╝██╔══██╗██╔══██╗██╔════╝ ",
        r"   ██║   ██████╔╝███████║██║  ███╗",
        r"   ██║   ██╔══██╗██╔══██║██║   ██║",
        r"   ██║   ██║  ██║██║  ██║╚██████╔╝",
        r"   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ "
    ]
    
    banner_text = Text()
    # Electric Amber & Sunset Gold Gradient
    colors = ["#f59e0b", "#fbbf24", "#fcd34d", "#f59e0b", "#d97706", "#b45309"]
    for i, line in enumerate(banner_lines):
        banner_text.append(line + "\n", style=f"bold {colors[i % len(colors)]}")
        
    banner_text.append(f"  {subtitle}", style="italic #fde68a")
    
    cfg = rag_engine.load_config()
    llm_tag = f"LLM: {cfg.get('llm_provider', 'ollama').upper()} ({cfg.get('ollama_llm_model' if cfg.get('llm_provider')=='ollama' else 'gemini_model')})"
    emb_tag = f"Embeddings: {cfg.get('embedding_model', 'nomic-embed-text')}"
    
    console.print(Panel(
        banner_text,
        border_style="#f59e0b",
        subtitle=f"[bold #fbbf24]v1.0.0 • Base: [{ACTIVE_COLLECTION['name']}] • {llm_tag} • {emb_tag}[/bold #fbbf24]",
        subtitle_align="right",
        padding=(1, 2)
    ))

def print_wizard_box(title: str, subtitle: str):
    content = Text()
    content.append(f"{title}\n", style="bold #fbbf24")
    content.append(f"{subtitle}", style="dim #fde68a")
    console.print(Panel(content, border_style="#f59e0b", padding=(0, 1)))

def pause_prompt():
    questionary.press_any_key_to_continue("Press any key to return to the main menu...").ask()

# ==========================================
# ACTIONS
# ==========================================

def action_query_rag():
    console.clear()
    render_banner()
    print_wizard_box(
        f"💬 Interactive RAG Document Chat — [{ACTIVE_COLLECTION['name']}]",
        "Ask questions grounded in your local vector embeddings with source citations."
    )
    
    while True:
        query_str = questionary.text("Enter your question (or 'back' to return):", style=CUSTOM_STYLE).ask()
        if not query_str or query_str.strip().lower() in ["back", "exit", "q"]:
            return
            
        console.print("\n[bold cyan]🧠 Querying vector database & streaming answer...[/bold cyan]\n")
        sources = []
        full_text = ""
        latency = 0.0
        model_used = "TRAG AI"
        
        try:
            console.print("[bold green]TRAG Answer:[/bold green] ", end="")
            for chunk in rag_engine.stream_query_rag(query_str.strip(), collection_id=ACTIVE_COLLECTION["id"], top_k=4):
                if chunk["type"] == "sources":
                    sources = chunk.get("sources", [])
                elif chunk["type"] == "token":
                    token = chunk["token"]
                    full_text += token
                    console.print(token, end="", style="bright_white")
                elif chunk["type"] == "done":
                    latency = chunk["latency"]
                    model_used = chunk["model"]
            console.print("\n")
            console.print(f"[dim yellow]⏱️ Latency: {latency:.2f}s | Model: {model_used}[/dim yellow]\n")
        except Exception as e:
            console.print(f"\n[bold red]❌ RAG Error:[/bold red] {e}\n")
            continue

        # Sources Table
        if sources:
            src_table = Table(title="[bold cyan]📚 Retrieved Source Passages[/bold cyan]", border_style="cyan", show_lines=True)
            src_table.add_column("Doc / File", style="bold white", width=22)
            src_table.add_column("Similarity", justify="center", style="yellow", width=12)
            src_table.add_column("Context Snippet", style="dim white")
            
            for s in sources:
                snippet = s["content"][:180].replace("\n", " ") + "..."
                src_table.add_row(s["filename"], f"{s['similarity_score']*100:.1f}%", snippet)
            console.print(src_table)
        else:
            console.print("[dim yellow]No relevant context found above threshold.[/dim yellow]")
            
        console.print("\n")

def action_ingest_document():
    console.clear()
    render_banner()
    print_wizard_box(
        f"📥 Ingest & Embed Document — [{ACTIVE_COLLECTION['name']}]",
        "Parses PDF, Markdown, Text, or Code files, splits into chunks, and computes embeddings."
    )
    
    path_str = questionary.text("Enter path to file to ingest (e.g. /home/kevin/notes.pdf):", style=CUSTOM_STYLE).ask()
    if not path_str or not path_str.strip():
        return
        
    target_path = Path(path_str.strip()).expanduser()
    if not target_path.exists() or not target_path.is_file():
        console.print(f"\n[bold red]❌ File not found at: {target_path}[/bold red]\n")
        pause_prompt()
        return

    with console.status(f"[bold cyan]Parsing and generating vector embeddings for {target_path.name}...[/bold cyan]", spinner="dots"):
        try:
            res = rag_engine.ingest_file(target_path, collection_id=ACTIVE_COLLECTION["id"])
            console.print(f"\n[bold green]✓ Ingested '{res['filename']}' successfully![/bold green]")
            console.print(f"[dim]Total Characters: {res['char_count']:,} | Vector Chunks Created: {res['chunk_count']}[/dim]\n")
        except Exception as e:
            console.print(f"\n[bold red]❌ Ingestion Failed:[/bold red] {e}\n")
            
    pause_prompt()

def action_list_documents():
    console.clear()
    render_banner()
    print_wizard_box(f"📜 Ingested Documents — [{ACTIVE_COLLECTION['name']}]", "Browse and manage your vector database.")
    
    docs = db.get_documents(collection_id=ACTIVE_COLLECTION["id"])
    if not docs:
        console.print(f"\n[yellow]No documents in '{ACTIVE_COLLECTION['name']}' knowledge base yet.[/yellow]\n")
        pause_prompt()
        return
        
    table = Table(title="[bold cyan]Document Repository[/bold cyan]", border_style="cyan", show_lines=True)
    table.add_column("ID", justify="center", style="dim cyan", width=5)
    table.add_column("Filename", style="bold white", width=25)
    table.add_column("Type", justify="center", style="magenta", width=8)
    table.add_column("Characters", justify="right", style="white", width=12)
    table.add_column("Chunks", justify="right", style="bold green", width=10)
    table.add_column("Ingested At", justify="center", style="dim white", width=18)
    
    for d in docs:
        table.add_row(
            f"#{d['id']}",
            d["filename"],
            d["file_type"].upper(),
            f"{d['char_count']:,}",
            str(d["chunk_count"]),
            d["created_at"][:16].replace("T", " ")
        )
    console.print(table)
    console.print(f"\n[dim]Total: {len(docs)} documents[/dim]\n")
    
    manage = questionary.select(
        "Manage Documents:",
        choices=["🔙 Back to Main Menu", "🗑️ Delete a Document"],
        style=CUSTOM_STYLE
    ).ask()
    
    if manage == "🗑️ Delete a Document":
        del_id = questionary.text("Enter document ID to delete (e.g. 1):", style=CUSTOM_STYLE).ask()
        if del_id and del_id.isdigit():
            if db.delete_document(int(del_id)):
                console.print(f"[bold green]✓ Deleted document #{del_id} and purged its vector chunks.[/bold green]")
            else:
                console.print(f"[bold red]Document #{del_id} not found.[/bold red]")
            pause_prompt()

def action_manage_collections():
    global ACTIVE_COLLECTION
    console.clear()
    render_banner()
    print_wizard_box("🗂️  Knowledge Base Collections", "Switch or create domain-specific vector stores.")
    
    colls = db.get_collections()
    table = Table(title="[bold cyan]Available Knowledge Bases[/bold cyan]", border_style="cyan", show_lines=True)
    table.add_column("ID", justify="center", style="dim cyan", width=4)
    table.add_column("Base Name", style="bold yellow", width=18)
    table.add_column("Docs", justify="right", style="green", width=8)
    table.add_column("Chunks", justify="right", style="bold green", width=10)
    table.add_column("Description", style="dim white")
    
    for c in colls:
        active_mark = " (Active)" if c["id"] == ACTIVE_COLLECTION["id"] else ""
        table.add_row(
            f"#{c['id']}",
            f"{c['name']}{active_mark}",
            str(c["doc_count"]),
            str(c["chunk_count"]),
            c["description"] or "-"
        )
    console.print(table)
    console.print("\n")
    
    choices = [questionary.Choice(f"📁 Switch to '{c['name']}'", value=("switch", c)) for c in colls]
    choices.append(questionary.Separator())
    choices.append(questionary.Choice("➕ Create New Knowledge Base", value=("new", None)))
    choices.append(questionary.Choice("🔙 Back to Main Menu", value=("back", None)))
    
    act = questionary.select("Choose Collection Action:", choices=choices, style=CUSTOM_STYLE).ask()
    if not act or act[0] == "back":
        return
        
    if act[0] == "switch":
        target = act[1]
        ACTIVE_COLLECTION = {"id": target["id"], "name": target["name"]}
        console.print(f"\n[bold green]✓ Switched active base to '{target['name']}'![/bold green]\n")
        time.sleep(0.8)
        
    elif act[0] == "new":
        name = questionary.text("Enter collection name (e.g. legal, research, dev):", style=CUSTOM_STYLE).ask()
        if not name or not name.strip():
            return
        desc = questionary.text("Description (optional):", style=CUSTOM_STYLE).ask()
        success, msg, cid = db.create_collection(name.strip(), desc)
        if success:
            ACTIVE_COLLECTION = {"id": cid, "name": name.strip().lower()}
            console.print(f"\n[bold green]✓ Created & switched to '{name.strip()}'![/bold green]\n")
            time.sleep(0.8)
        else:
            console.print(f"\n[bold red]❌ {msg}[/bold red]\n")
            pause_prompt()

def action_engine_settings():
    console.clear()
    render_banner()
    print_wizard_box("⚙️  TRAG Model Engine Settings", "Configure local Ollama host, embedding models, and Gemini fallback.")
    
    cfg = rag_engine.load_config()
    
    llm_p = questionary.select(
        f"LLM Generation Engine (Current: {cfg.get('llm_provider', 'ollama').upper()}):",
        choices=[
            questionary.Choice("🦙 Local Ollama (llama3.2:3b - 100% Private)", value="ollama"),
            questionary.Choice("🔷 Google Gemini Cloud (High-speed intelligence)", value="gemini")
        ],
        style=CUSTOM_STYLE
    ).ask()
    if llm_p:
        cfg["llm_provider"] = llm_p
        
    if cfg["llm_provider"] == "ollama":
        host = questionary.text("Ollama Host URL:", default=cfg.get("ollama_host", "http://localhost:11434"), style=CUSTOM_STYLE).ask()
        if host:
            cfg["ollama_host"] = host.strip()
        model = questionary.text("Ollama LLM Model Name:", default=cfg.get("ollama_llm_model", "llama3.2:3b"), style=CUSTOM_STYLE).ask()
        if model:
            cfg["ollama_llm_model"] = model.strip()
    else:
        k = questionary.text("Google Gemini API Key:", default=cfg.get("gemini_api_key", ""), style=CUSTOM_STYLE).ask()
        if k:
            cfg["gemini_api_key"] = k.strip()
        m = questionary.select("Select Gemini Model:", choices=["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"], default=cfg.get("gemini_model", "gemini-2.5-flash"), style=CUSTOM_STYLE).ask()
        if m:
            cfg["gemini_model"] = m

    rag_engine.save_config(cfg)
    console.print("\n[bold green]✓ Configuration saved successfully![/bold green]\n")
    pause_prompt()

def action_launch_gui():
    console.clear()
    render_banner()
    print_wizard_box("🚀 Launching TRAG Web GUI Dashboard", "Starting local server at http://localhost:8450")
    console.print("\n[bold cyan]Opening your browser to TRAG GUI...[/bold cyan]")
    console.print("[dim]Press Ctrl+C anytime to stop GUI server and return to terminal.[/dim]\n")
    import server
    server.launch_server(port=8450, open_browser=True)

# ==========================================
# MAIN INTERACTIVE LOOP
# ==========================================

def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["gui", "web", "--gui", "-g"]:
        action_launch_gui()
        return

    # Direct query CLI flag: trag "what is vector database?"
    if len(sys.argv) > 1 and sys.argv[1].lower() not in ["help", "-h"]:
        q = " ".join(sys.argv[1:])
        db.init_db()
        with console.status(f"[bold cyan]Querying TRAG for: '{q}'...[/bold cyan]", spinner="dots"):
            res = rag_engine.query_rag(q, collection_id=1, top_k=4)
        console.print("\n")
        console.print(Panel(Markdown(res["response"]), title=f"[bold green]✨ TRAG Answer ({res['model_used']})[/bold green]", border_style="green", padding=(1, 2)))
        return

    db.init_db()
    
    while True:
        console.clear()
        render_banner()
        print_wizard_box(
            f"⚡ TRAG Document Vector Intelligence — [{ACTIVE_COLLECTION['name']}]",
            "Chat with PDFs, Markdown, and codebases locally with zero cloud dependencies."
        )
        
        choice = questionary.select(
            "Select TRAG Action: (Use arrow keys)",
            choices=[
                questionary.Choice("💬  Ask / RAG Chat           — Query your documents with cited vector context", value="query"),
                questionary.Choice("📥  Ingest & Embed Document — Parse & vectorize PDF, TXT, MD, or Code file", value="ingest"),
                questionary.Choice("📜  View Ingested Documents — Browse files, character counts & vector chunks", value="docs"),
                questionary.Choice("🗂️   Knowledge Base Manager  — Switch or create domain collections", value="collections"),
                questionary.Choice("💻  Launch TRAG Web GUI     — Minimalist browser dashboard with chat canvas", value="gui"),
                questionary.Choice("⚙️   Engine Configuration    — Ollama host, embedding models, and Gemini API", value="settings"),
                questionary.Separator(),
                questionary.Choice("🚪  Exit TRAG", value="exit")
            ],
            style=CUSTOM_STYLE
        ).ask()
        
        if choice is None or choice == "exit":
            console.print("\n[bold magenta]Thank you for using TRAG! Have a great day! 👋[/bold magenta]\n")
            sys.exit(0)
        elif choice == "query":
            action_query_rag()
        elif choice == "ingest":
            action_ingest_document()
        elif choice == "docs":
            action_list_documents()
        elif choice == "collections":
            action_manage_collections()
        elif choice == "gui":
            action_launch_gui()
        elif choice == "settings":
            action_engine_settings()

if __name__ == "__main__":
    main()
