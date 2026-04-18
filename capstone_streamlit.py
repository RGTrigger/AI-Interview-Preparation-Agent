from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import types
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "capstone_streamlit.py.ipynb"
DB_PATH = BASE_DIR / "chat_history.db"
PROJECT_TITLE = "AI Interview Preparation Agent"
DEVELOPER_NAME = "Gaurav (RGtrigger)"
PROJECT_SUMMARY = (
    "Technical interview preparation agent with LangGraph workflow, RAG-backed "
    "answers, interview mode, memory-backed chat history, and focused practice "
    "for coding and core computer science concepts."
)
SUGGESTIONS = [
    "Explain polymorphism in OOP",
    "What is normalization in DBMS?",
    "What is deadlock in operating systems?",
    "Ask me Python interview question",
]


st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles(theme: str) -> None:
    palette = {
        "Dark": {
            "bg": "#1f1f1f",
            "bg1": "#202123",
            "bg2": "#171717",
            "sidebar": "#171717",
            "panel": "#2a2b32",
            "user": "#30323a",
            "user_text": "#f8fafc",
            "line": "rgba(255,255,255,0.08)",
            "text": "#ececec",
            "muted": "#a7aab4",
            "accent": "#10a37f",
            "accent_soft": "rgba(16,163,127,0.15)",
            "input": "#2b2c32",
            "input_shell": "#1c1f26",
            "input_border": "rgba(255,255,255,0.12)",
            "history": "rgba(255,255,255,0.02)",
            "hero": "#b8bcc7",
            "button_bg": "#171b26",
            "button_text": "#f8fafc",
            "button_border": "rgba(71,85,105,0.45)",
            "button_hover": "#202635",
        },
        "Light": {
            "bg": "#f6f7fb",
            "bg1": "#ffffff",
            "bg2": "#eef2f7",
            "sidebar": "#f4f6fb",
            "panel": "#ffffff",
            "user": "#ebf3ff",
            "user_text": "#0f172a",
            "line": "rgba(15,23,42,0.12)",
            "text": "#101828",
            "muted": "#475467",
            "accent": "#10a37f",
            "accent_soft": "rgba(16,163,127,0.12)",
            "input": "#ffffff",
            "input_shell": "#ffffff",
            "input_border": "rgba(15,23,42,0.12)",
            "history": "#ffffff",
            "hero": "#475467",
            "button_bg": "#ffffff",
            "button_text": "#101828",
            "button_border": "rgba(15,23,42,0.12)",
            "button_hover": "#eef2f7",
        },
    }[theme]

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {palette["bg"]};
            --bg1: {palette["bg1"]};
            --bg2: {palette["bg2"]};
            --sidebar: {palette["sidebar"]};
            --panel: {palette["panel"]};
            --user: {palette["user"]};
            --user-text: {palette["user_text"]};
            --line: {palette["line"]};
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --accent: {palette["accent"]};
            --accent-soft: {palette["accent_soft"]};
            --input: {palette["input"]};
            --input-shell: {palette["input_shell"]};
            --input-border: {palette["input_border"]};
            --history: {palette["history"]};
            --hero-copy: {palette["hero"]};
            --button-bg: {palette["button_bg"]};
            --button-text: {palette["button_text"]};
            --button-border: {palette["button_border"]};
            --button-hover: {palette["button_hover"]};
        }}

        html, body, [class*="css"] {{
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }}

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background:
                radial-gradient(circle at top, var(--bg1) 0%, var(--bg2) 58%, var(--bg) 100%);
            color: var(--text);
        }}

        #MainMenu, footer {{
            display: none;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
            border: none;
        }}

        [data-testid="stHeader"]::before {{
            display: none;
        }}

        [data-testid="stHeader"] > div {{
            padding-top: 0.35rem;
        }}

        [data-testid="stExpandSidebarButton"] button {{
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            border-radius: 999px !important;
            color: var(--text) !important;
            box-shadow: 0 10px 25px rgba(15,23,42,0.08);
        }}

        [data-testid="stSidebar"] {{
            background: var(--sidebar);
            border-right: 1px solid var(--line);
            overflow: hidden !important;
            height: 100%;
            min-width: 21rem !important;
            max-width: 21rem !important;
        }}

        section[data-testid="stSidebar"][aria-expanded="true"] {{
            min-width: 21rem !important;
            max-width: 21rem !important;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 0.82rem;
            padding-bottom: 0.28rem;
            height: 100%;
            min-height: 100%;
            flex: 1 1 auto;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            overflow: hidden !important;
            position: relative;
        }}

        [data-testid="stSidebarContent"] {{
            height: 100%;
            overflow: hidden !important;
            display: flex;
            flex-direction: column;
        }}

        [data-testid="stSidebarUserContent"] {{
            flex: 1 1 auto;
            min-height: 0;
            padding-bottom: 0 !important;
            display: flex;
            flex-direction: column;
        }}

        [data-testid="stSidebarUserContent"] > div {{
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }}

        [data-testid="stSidebar"] .block-container > [data-testid="stVerticalBlock"] {{
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 0;
        }}

        section[data-testid="stSidebar"] > div {{
            overflow: hidden !important;
        }}

        [data-testid="stSidebar"] button {{
            border-radius: 0.9rem;
            border: 1px solid var(--line);
            background: transparent;
            color: var(--text);
        }}

        [data-testid="stSidebar"] button[kind="primary"] {{
            background: var(--panel);
        }}

        .brand {{
            padding: 0.08rem 0 0.45rem 0;
        }}

        .brand-title {{
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 0;
        }}

        .sidebar-label {{
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0.58rem 0 0.28rem 0;
        }}

        .footer-card {{
            border-top: 1px solid var(--line);
            padding: 0.78rem 0 0.48rem 0;
            background: var(--sidebar);
        }}

        .footer-title {{
            font-weight: 700;
            font-size: 1.08rem;
            margin-bottom: 0.28rem;
            color: var(--text);
            line-height: 1.2;
        }}

        .footer-copy {{
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.44;
            margin-bottom: 0.44rem;
            white-space: normal;
        }}

        .footer-dev {{
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.2;
            white-space: normal;
        }}

        .footer-dev-label {{
            display: block;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.16rem;
        }}

        .footer-dev-name {{
            display: block;
            margin-top: 0;
            font-size: 0.98rem;
            line-height: 1.24;
            color: var(--text);
        }}

        .hero {{
            max-width: 920px;
            margin: 4.8rem auto 0 auto;
            padding: 0 1rem;
            text-align: center;
        }}

        .hero-title {{
            font-size: clamp(2.8rem, 6vw, 4.8rem);
            line-height: 1.02;
            letter-spacing: -0.04em;
            font-weight: 760;
            margin: 0;
            color: var(--text);
        }}

        .chip-caption {{
            color: var(--muted);
            font-size: 0.85rem;
            margin: 1.15rem 0 0.6rem 0;
        }}

        [data-testid="stChatMessage"] {{
            background: transparent;
            border: none;
            padding-left: 0;
            padding-right: 0;
        }}

        [data-testid="stChatMessageContent"] {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 1.25rem;
            color: var(--text);
            padding: 1rem 1.15rem;
        }}

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{
            background: var(--user);
            color: var(--user-text);
            border-color: var(--input-border);
        }}

        [data-testid="stBottom"],
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] {{
            background: var(--bg1) !important;
        }}

        [data-testid="stBottom"] > div {{
            border-top: 1px solid var(--line);
        }}

        [data-testid="stBottomBlockContainer"] {{
            padding-top: 0.85rem;
            padding-bottom: 0.9rem;
        }}

        .topic-box {{
            margin-top: 0.65rem;
            padding: 0.72rem 0.84rem;
            border-radius: 0.9rem;
            border: 1px solid var(--line);
            background: var(--history);
            color: var(--muted);
            font-size: 0.84rem;
        }}

        [data-testid="stChatInput"] {{
            background: transparent !important;
            padding-top: 0;
            padding-bottom: 0;
        }}

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div,
        [data-testid="stChatInput"] form {{
            background: var(--input-shell) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 1.5rem !important;
            box-shadow: 0 16px 36px rgba(15,23,42,0.08) !important;
        }}

        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="textarea"] > div {{
            background: transparent !important;
        }}

        [data-testid="stChatInput"] textarea {{
            background: transparent !important;
            color: var(--text) !important;
            border-radius: 1.35rem !important;
            border: none !important;
            box-shadow: none !important;
            padding-top: 0.55rem !important;
            -webkit-text-fill-color: var(--text) !important;
            caret-color: var(--text) !important;
        }}

        [data-testid="stChatInputTextArea"] {{
            background: transparent !important;
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }}

        [data-testid="stChatInput"] button {{
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 1rem;
        }}

        [data-testid="stChatInput"] textarea::placeholder {{
            color: var(--muted) !important;
        }}

        .stButton > button {{
            background: var(--button-bg);
            color: var(--button-text);
            border: 1px solid var(--button-border);
        }}

        .stButton > button:hover {{
            background: var(--button-hover);
            color: var(--button-text);
        }}

        [data-baseweb="select"] > div {{
            background: var(--panel) !important;
            border-color: var(--line) !important;
            color: var(--text) !important;
        }}

        [data-testid="stRadio"] label,
        [data-testid="stSelectbox"] label {{
            color: var(--muted) !important;
        }}

        [data-testid="stSidebar"] * {{
            color: var(--text);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                interview_active INTEGER NOT NULL DEFAULT 0,
                current_question TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ui_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                topics TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(ui_conversations)").fetchall()}
        if "interview_active" not in columns:
            conn.execute(
                "ALTER TABLE ui_conversations ADD COLUMN interview_active INTEGER NOT NULL DEFAULT 0"
            )
        if "current_question" not in columns:
            conn.execute(
                "ALTER TABLE ui_conversations ADD COLUMN current_question TEXT NOT NULL DEFAULT ''"
            )


def list_conversations() -> List[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM ui_conversations
            ORDER BY datetime(updated_at) DESC, created_at DESC
            """
        ).fetchall()


def create_conversation(title: str = "New chat") -> str:
    conversation_id = str(uuid.uuid4())
    timestamp = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ui_conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, title, timestamp, timestamp),
        )
    return conversation_id


def ensure_conversation() -> str:
    rows = list_conversations()
    if rows:
        return rows[0]["id"]
    return create_conversation()


def delete_conversation(conversation_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM ui_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM ui_conversations WHERE id = ?", (conversation_id,))


def load_messages(conversation_id: str) -> List[Dict[str, str]]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, topics
            FROM ui_messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [
        {"role": row["role"], "content": row["content"], "topics": row["topics"]}
        for row in rows
    ]


def save_message(conversation_id: str, role: str, content: str, topics: str = "") -> None:
    timestamp = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ui_messages (conversation_id, role, content, topics, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, topics, timestamp),
        )
        conn.execute(
            "UPDATE ui_conversations SET updated_at = ? WHERE id = ?",
            (timestamp, conversation_id),
        )


def get_conversation_meta(conversation_id: str) -> Dict[str, Any]:
    init_db()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id, title, interview_active, current_question
            FROM ui_conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else {}


def update_conversation_meta(
    conversation_id: str,
    *,
    interview_active: bool | None = None,
    current_question: str | None = None,
) -> None:
    init_db()
    updates = []
    values: List[Any] = []
    if interview_active is not None:
        updates.append("interview_active = ?")
        values.append(int(interview_active))
    if current_question is not None:
        updates.append("current_question = ?")
        values.append(current_question)
    updates.append("updated_at = ?")
    values.append(now_iso())
    values.append(conversation_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE ui_conversations SET {', '.join(updates)} WHERE id = ?", values)


def update_title(conversation_id: str, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE ui_conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now_iso(), conversation_id),
        )


def make_title(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "New chat"
    if len(cleaned) <= 38:
        return cleaned
    return cleaned[:35].rstrip() + "..."


def _extract_tool_node(cell_source: str) -> str:
    match = re.search(r"(def tool_node\(state\):[\s\S]*?)\n\s*def answer_node", cell_source)
    if not match:
        raise RuntimeError("Could not extract tool_node from notebook.")
    return match.group(1)


@st.cache_resource(show_spinner=False)
def load_backend() -> Dict[str, Any]:
    load_dotenv(BASE_DIR / ".env", override=False)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    module_name = "__notebook_backend__"
    notebook_module = types.ModuleType(module_name)
    sys.modules[module_name] = notebook_module
    namespace = notebook_module.__dict__
    namespace["__name__"] = module_name

    selected_cells: List[str] = []
    tool_node_code = None
    graph_cell = None

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))

        if "def tool_node" in source:
            tool_node_code = _extract_tool_node(source)
        if "import streamlit as st" in source:
            continue

        if source.startswith("from dotenv import load_dotenv"):
            selected_cells.append(source)
        elif source.startswith("DOMAIN = "):
            selected_cells.append(source)
        elif "class CapstoneState" in source:
            selected_cells.append(source)
        elif "def memory_node" in source:
            selected_cells.append(source)
        elif "def router_node" in source:
            selected_cells.append(source)
        elif "def retrieval_node" in source:
            selected_cells.append(source)
        elif "def answer_node" in source and "INTERVIEW QUESTION" in source:
            selected_cells.append(source)
        elif "def interview_eval_node" in source:
            selected_cells.append(source)
        elif "FAITHFULNESS_THRESHOLD = 0.7" in source:
            selected_cells.append(source)
        elif "def save_node" in source and 'messages.append({' in source:
            selected_cells.append(source)
        elif "# Part 4 â€” Graph Assembly" in source or "# Part 4 — Graph Assembly" in source:
            graph_cell = source

    if not tool_node_code:
        raise RuntimeError("Notebook backend is missing tool_node.")

    for code in selected_cells:
        exec(code, namespace)

    exec(tool_node_code, namespace)

    if not graph_cell:
        raise RuntimeError("Notebook graph assembly cell not found.")

    exec(graph_cell, namespace)

    if "app" not in namespace:
        raise RuntimeError("Notebook backend did not create the compiled app.")

    return namespace


def render_sidebar(active_conversation_id: str) -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand">
                <div class="brand-title">{PROJECT_TITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("+ New chat", use_container_width=True):
            st.session_state.active_conversation_id = create_conversation()
            st.rerun()

        st.markdown('<div class="sidebar-label">Appearance</div>', unsafe_allow_html=True)
        chosen_theme = st.radio(
            "Theme",
            ["Dark", "Light"],
            horizontal=True,
            key="theme_selector",
            label_visibility="collapsed",
        )
        if st.session_state.theme != chosen_theme:
            st.session_state.theme = chosen_theme
            st.rerun()

        st.markdown('<div class="sidebar-label">Chats</div>', unsafe_allow_html=True)
        history_box = st.container(height=500, border=False)
        with history_box:
            for row in list_conversations():
                cols = st.columns([0.85, 0.15], gap="small")
                button_type = "primary" if row["id"] == active_conversation_id else "secondary"
                if cols[0].button(row["title"], key=f"open_{row['id']}", use_container_width=True, type=button_type):
                    st.session_state.active_conversation_id = row["id"]
                    st.rerun()
                if cols[1].button("X", key=f"delete_{row['id']}", use_container_width=True):
                    delete_conversation(row["id"])
                    remaining = list_conversations()
                    st.session_state.active_conversation_id = (
                        remaining[0]["id"] if remaining else create_conversation()
                    )
                    st.rerun()

        st.markdown(
            f"""
            <div class="footer-card">
                <div class="footer-title">{PROJECT_TITLE}</div>
                <div class="footer-copy">{PROJECT_SUMMARY}</div>
                <div class="footer-dev"><span class="footer-dev-label">Developer / Creator</span><span class="footer-dev-name"><strong>{DEVELOPER_NAME}</strong></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.active_conversation_id


def render_empty_state() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1 class="hero-title">{PROJECT_TITLE}</h1>
            <div class="chip-caption">Try one of these prompts</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2, gap="small")
    for index, suggestion in enumerate(SUGGESTIONS):
        with cols[index % 2]:
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
def render_messages(messages: List[Dict[str, str]]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("topics"):
                st.markdown(
                    f"<div class='topic-box'><strong>Knowledge base topics:</strong> {message['topics']}</div>",
                    unsafe_allow_html=True,
                )


def ask_backend(prompt: str, thread_id: str, backend: Dict[str, Any]) -> Dict[str, Any]:
    app = backend["app"]
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"question": prompt}, config=config)


def start_interview_round(conversation_id: str, backend: Dict[str, Any], topic: str) -> str:
    tool_node = backend["tool_node"]
    result = tool_node({"question": f"Start {topic} interview"})
    question = result.get("tool_result", "What is polymorphism in OOP?")
    update_conversation_meta(conversation_id, interview_active=True, current_question=question)
    return (
        f"Interview Mode: {topic}\n\n"
        f"Question:\n{question}\n\n"
        "Reply with your answer in the chat. I will score it out of 10."
    )


def score_interview_answer(answer_text: str, current_question: str, backend: Dict[str, Any]) -> str:
    interview_eval_node = backend["interview_eval_node"]
    result = interview_eval_node({"question": answer_text, "current_question": current_question})
    return result.get("answer", "Unable to evaluate your answer.")


def main() -> None:
    init_db()
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"
    if "active_conversation_id" not in st.session_state:
        # Start each fresh browser session with a brand-new chat instead of
        # reopening the most recently used conversation.
        st.session_state.active_conversation_id = create_conversation()
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    inject_styles(st.session_state.theme)
    backend = load_backend()

    active_conversation_id = render_sidebar(st.session_state.active_conversation_id)
    messages = load_messages(active_conversation_id)
    meta = get_conversation_meta(active_conversation_id)

    if messages:
        render_messages(messages)
    else:
        render_empty_state()

    prompt = st.chat_input("Message the interview agent")
    if st.session_state.pending_prompt and not prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        save_message(active_conversation_id, "user", prompt)
        if len(messages) == 0:
            update_title(active_conversation_id, make_title(prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if meta.get("interview_active"):
                    answer = score_interview_answer(prompt, meta.get("current_question", ""), backend)
                    topics = ""
                    update_conversation_meta(
                        active_conversation_id,
                        interview_active=False,
                        current_question="",
                    )
                else:
                    result = ask_backend(prompt, active_conversation_id, backend)
                    answer = result.get("answer", "No response")
                    topics = ", ".join(result.get("sources", []) or [])
            st.markdown(answer)
            if topics:
                st.markdown(
                    f"<div class='topic-box'><strong>Knowledge base topics:</strong> {topics}</div>",
                    unsafe_allow_html=True,
                )

        save_message(active_conversation_id, "assistant", answer, topics)
        st.rerun()


if __name__ == "__main__":
    main()
