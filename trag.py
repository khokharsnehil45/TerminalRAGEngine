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
    """Renders the retro 3D-styled TRAG banner with Emerald Matrix gradient."""
    banner_lines = [
        r"████████╗██████╗  █████╗  ██████╗ ",
        r"╚══██╔══╝██╔══██╗██╔══██╗██╔════╝ ",
        r"   ██║   ██████╔╝███████║██║  ███╗",
        r"   ██║   ██╔══██╗██╔══██║██║   ██║",
        r"   ██║   ██║  ██║██║  ██║╚██████╔╝",
        r"   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ "
    ]
    
    banner_text = Text()
    # Emerald Matrix Gradient
    colors = ["#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#34d399", "#10b981"]
    for i, line in enumerate(banner_lines):
        banner_text.append(line + "\n", style=f"bold {colors[i % len(colors)]}")
        
    banner_text.append(f"  {subtitle}", style="italic #6ee7b7")
    
    cfg = rag_engine.load_config()
    provider_str = f"Embedding: {cfg.get('embedding_model', 'nomic-embed-text')} • LLM: {cfg.get('llm_provider', 'ollama')}"
    
    console.print(Panel(
        banner_text,
        border_style="#10b981",
        subtitle=f"[bold #6ee7b7]v1.0.0 • Base: {ACTIVE_COLLECTION['name']} • {provider_str}[/bold #6ee7b7]",
        subtitle_align="right",
        padding=(1, 2)
    ))

def print_wizard_box(title: str, subtitle: str):
    content = Text()
    content.append(f"{title}\n", style="bold #10b981")
    content.append(f"{subtitle}", style="dim #6ee7b7")
    console.print(Panel(content, border_style="#10b981", padding=(0, 1)))

def pause_prompt():
    questionary.press_any_key_to_continue("Press any key to return to the main menu...").ask()

# ==========================================
# ACTIONS
# ==========================================

def action_query_rag():
    console.clear()
    render_banner()
    print_wizard_box(
        f"💬 Interactive Multi-Turn RAG Chat — [{ACTIVE_COLLECTION['name']}]",
        "Continuous dialogue with contextual memory & source grounding. Type 'clear' to reset or 'exit' to return."
    )
    
    # Check existing sessions
    existing_sessions = db.get_sessions(collection_id=ACTIVE_COLLECTION["id"])
    choices = [
        questionary.Choice("✨ Start Brand New Chat Session", value=("new", None))
    ]
    for s in existing_sessions[:6]:
        choices.append(questionary.Choice(
            f"📜 Resume: {s['title']} ({s['message_count']} msgs • {s['updated_at'][:16].replace('T', ' ')})",
            value=("resume", s)
        ))
    choices.append(questionary.Choice("🔙 Back to Main Menu", value=("back", None)))
    
    session_pick = questionary.select("Choose Chat Session:", choices=choices, style=CUSTOM_STYLE).ask()
    if not session_pick or session_pick[0] == "back":
        return
        
    session_id = None
    if session_pick[0] == "new":
        title = questionary.text("Enter Session Topic / Title (optional):", default="New Research Session", style=CUSTOM_STYLE).ask()
        session_id = db.create_session(title or "Research Session", collection_id=ACTIVE_COLLECTION["id"])
    else:
        session_id = session_pick[1]["id"]
        # Print prior messages in session
        past_msgs = db.get_session_messages(session_id)
        if past_msgs:
            console.print("\n[dim]─── Previous Session Context ───[/dim]")
            for m in past_msgs:
                if m["role"] == "user":
                    console.print(f"[bold cyan]You:[/bold cyan] {m['content']}")
                else:
                    console.print(f"[bold green]TRAG:[/bold green] {m['content'][:250]}{'...' if len(m['content'])>250 else ''}\n")
            console.print("[dim]────────────────────────────────[/dim]\n")

    while True:
        query_str = questionary.text("💬 You:", style=CUSTOM_STYLE).ask()
        if not query_str or query_str.strip().lower() in ["back", "exit", "q", "quit"]:
            return
        elif query_str.strip().lower() == "clear":
            session_id = db.create_session("New Research Session", collection_id=ACTIVE_COLLECTION["id"])
            console.print("\n[bold green]✓ Memory reset! New session started.[/bold green]\n")
            continue
            
        console.print("\n[bold cyan]🧠 Querying vector database & streaming answer...[/bold cyan]\n")
        sources = []
        full_text = ""
        latency = 0.0
        model_used = "TRAG AI"
        
        try:
            confidence = {}
            telemetry = {}
            console.print("[bold green]TRAG Answer:[/bold green] ", end="")
            for chunk in rag_engine.stream_query_rag(query_str.strip(), collection_id=ACTIVE_COLLECTION["id"], top_k=4, session_id=session_id):
                if chunk["type"] == "sources":
                    sources = chunk.get("sources", [])
                    confidence = chunk.get("confidence", {})
                elif chunk["type"] == "token":
                    token = chunk["token"]
                    full_text += token
                    console.print(token, end="", style="bright_white")
                elif chunk["type"] == "done":
                    latency = chunk["latency"]
                    model_used = chunk["model"]
                    confidence = chunk.get("confidence", confidence)
                    telemetry = chunk.get("telemetry", {})
            console.print("\n")
            
            # Print Production Telemetry Card
            conf_tier = confidence.get("tier", "UNKNOWN")
            conf_pct = confidence.get("confidence_pct", 0.0)
            conf_color = "green" if conf_tier == "HIGH" else ("yellow" if conf_tier == "MEDIUM" else "red")
            
            telemetry_str = f"⏱️ Latency: {latency:.2f}s (Emb: {telemetry.get('emb_ms', 0)}ms | Ret: {telemetry.get('ret_ms', 0)}ms | LLM: {telemetry.get('llm_ms', 0)}ms)"
            badge_str = f"[{conf_color}]🛡️ Grounded Confidence: {conf_pct}% [{conf_tier}][/{conf_color}]"
            console.print(f"[dim]{telemetry_str} • {badge_str} • Model: {model_used}[/dim]\n")
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

def terminal_file_explorer(start_path: Path = Path.home()) -> Optional[Path]:
    """Interactive retro terminal file explorer to navigate directories and select files for ingestion."""
    current_dir = start_path.resolve()
    
    # Supported file extensions
    SUPPORTED_EXTS = {
        ".pdf": "📕 PDF Document",
        ".csv": "📊 CSV Spreadsheet",
        ".xlsx": "📗 Excel Spreadsheet",
        ".xls": "📗 Excel Spreadsheet",
        ".tsv": "📊 TSV Data",
        ".json": "📦 JSON Dataset",
        ".md": "📝 Markdown Notes",
        ".txt": "📄 Plain Text",
        ".py": "🐍 Python Code",
        ".js": "🟨 JavaScript",
        ".ts": "🔷 TypeScript",
        ".rs": "🦀 Rust Code",
        ".go": "🐹 Go Code",
        ".cpp": "⚙️ C++ Code",
        ".c": "⚙️ C Code",
        ".html": "🌐 HTML Webpage"
    }

    while True:
        console.clear()
        render_banner()
        print_wizard_box(
            f"📂 Terminal File Explorer — [{current_dir}]",
            "Use arrow keys to navigate directories and press Enter on any file to select & ingest."
        )

        try:
            entries = sorted(list(current_dir.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            console.print("[bold red]❌ Permission denied accessing this directory.[/bold red]")
            time.sleep(1.2)
            current_dir = current_dir.parent
            continue

        choices = []
        
        # Parent directory navigation
        if current_dir != current_dir.parent:
            choices.append(questionary.Choice("📁 .. (Go up to parent directory)", value=("dir", current_dir.parent)))
            
        # List subdirectories
        for p in entries:
            if p.is_dir() and not p.name.startswith('.'):
                try:
                    count = len(list(p.iterdir()))
                    choices.append(questionary.Choice(f"📁 {p.name}/ ({count} items)", value=("dir", p)))
                except Exception:
                    choices.append(questionary.Choice(f"📁 {p.name}/", value=("dir", p)))

        # List files
        file_count = 0
        for p in entries:
            if p.is_file() and not p.name.startswith('.'):
                ext = p.suffix.lower()
                size_kb = p.stat().st_size / 1024
                size_str = f"{size_kb:,.1f} KB" if size_kb < 1024 else f"{size_kb/1024:,.1f} MB"
                
                if ext in SUPPORTED_EXTS:
                    icon_label = SUPPORTED_EXTS[ext]
                    choices.append(questionary.Choice(
                        f"✨ {icon_label}: {p.name} [{size_str}]",
                        value=("file", p)
                    ))
                    file_count += 1
                else:
                    choices.append(questionary.Choice(
                        f"📄 {p.name} [{size_str}]",
                        value=("file", p)
                    ))
                    file_count += 1

        choices.append(questionary.Separator())
        choices.append(questionary.Choice("⌨️ Type Custom Path Manually", value=("manual", None)))
        choices.append(questionary.Choice("🔙 Cancel & Return to Menu", value=("cancel", None)))

        selection = questionary.select(
            f"Select a file to ingest (Current Directory: {current_dir.name}/):",
            choices=choices,
            style=CUSTOM_STYLE
        ).ask()

        if not selection or selection[0] == "cancel":
            return None
        elif selection[0] == "dir":
            current_dir = selection[1].resolve()
        elif selection[0] == "file":
            return selection[1]
        elif selection[0] == "manual":
            manual_str = questionary.text("Enter full file path:", style=CUSTOM_STYLE).ask()
            if manual_str and manual_str.strip():
                m_path = Path(manual_str.strip()).expanduser()
                if m_path.exists() and m_path.is_file():
                    return m_path
                else:
                    console.print(f"[bold red]File not found: {m_path}[/bold red]")
                    time.sleep(1.5)

def action_ingest_document():
    console.clear()
    render_banner()
    print_wizard_box(
        f"📥 Ingest & Embed Document — [{ACTIVE_COLLECTION['name']}]",
        "Select how you want to locate files for semantic vector ingestion."
    )
    
    method = questionary.select(
        "Choose Ingestion Method:",
        choices=[
            "📂 Launch Interactive Terminal File Explorer (Browse folders & files)",
            "⌨️ Enter File Path Directly (Type or paste path)",
            "🔙 Back to Main Menu"
        ],
        style=CUSTOM_STYLE
    ).ask()
    
    if not method or "Back" in method:
        return
        
    target_path = None
    if "File Explorer" in method:
        target_path = terminal_file_explorer(start_path=Path.cwd())
    else:
        path_str = questionary.text("Enter path to file to ingest (e.g. /home/kevin/data.csv):", style=CUSTOM_STYLE).ask()
        if path_str and path_str.strip():
            target_path = Path(path_str.strip()).expanduser()

    if not target_path or not target_path.exists() or not target_path.is_file():
        if target_path:
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
                questionary.Choice("📥  Ingest & Embed Document — Parse & vectorize PDF, CSV, Excel, TXT, MD, Code", value="ingest"),
                questionary.Choice("📂  Terminal File Explorer  — Interactive folder browser to find & ingest files", value="explorer"),
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
        elif choice == "explorer":
            selected = terminal_file_explorer(start_path=Path.cwd())
            if selected and selected.is_file():
                with console.status(f"[bold cyan]Parsing and generating vector embeddings for {selected.name}...[/bold cyan]", spinner="dots"):
                    try:
                        res = rag_engine.ingest_file(selected, collection_id=ACTIVE_COLLECTION["id"])
                        console.print(f"\n[bold green]✓ Ingested '{res['filename']}' successfully![/bold green]")
                        console.print(f"[dim]Total Characters: {res['char_count']:,} | Vector Chunks Created: {res['chunk_count']}[/dim]\n")
                    except Exception as e:
                        console.print(f"\n[bold red]❌ Ingestion Failed:[/bold red] {e}\n")
                pause_prompt()
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
