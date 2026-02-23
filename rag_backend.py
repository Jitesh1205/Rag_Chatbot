"""
RAG Backend — LangGraph + SqliteSaver + Dynamic PDF + Persistent FAISS indexes
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import sqlite3
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0, max_retries=2)


# ══════════════════════════════════════════════════════════════════════════════
# Global RAG state
# ══════════════════════════════════════════════════════════════════════════════

_retriever        = None
_current_pdf_name = None

FAISS_INDEX_DIR = "faiss_indexes"


def _index_path_for(pdf_name: str) -> str:
    """Convert a PDF filename to its FAISS index folder path."""
    safe = pdf_name.replace(".pdf", "").replace(" ", "_").replace("/", "_").replace("\\", "_")
    return os.path.join(FAISS_INDEX_DIR, safe)


def _make_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text-v2-moe")


# ══════════════════════════════════════════════════════════════════════════════
# PDF helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(file_path: str, original_filename: str = None) -> str:
    """
    Load a PDF, embed it, and save the FAISS index to disk.
    If the same PDF was indexed before, loads from disk instantly.
    Returns the display name of the PDF.
    """
    global _retriever, _current_pdf_name

    pdf_name   = original_filename or file_path.split("/")[-1]
    index_path = _index_path_for(pdf_name)
    embeddings = _make_embeddings()

    if os.path.exists(index_path):
        # Already indexed — load from disk (fast)
        vectorstore = FAISS.load_local(
            index_path, embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # New PDF — process, embed, save
        loader = PyPDFLoader(file_path)
        docs   = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks   = splitter.split_documents(docs)

        for chunk in chunks:
            chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8")

        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)

    _retriever        = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    _current_pdf_name = pdf_name
    return pdf_name


def restore_pdf_for_thread(pdf_name: str) -> bool:
    """
    Restore the correct FAISS retriever from disk when switching threads
    or on page reload. Returns True if successful, False if index missing.
    """
    global _retriever, _current_pdf_name

    if not pdf_name:
        return False

    # Already loaded — nothing to do
    if pdf_name == _current_pdf_name and _retriever is not None:
        return True

    index_path = _index_path_for(pdf_name)
    if not os.path.exists(index_path):
        return False   # user must re-upload

    try:
        embeddings  = _make_embeddings()
        vectorstore = FAISS.load_local(
            index_path, embeddings,
            allow_dangerous_deserialization=True
        )
        _retriever        = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        _current_pdf_name = pdf_name
        return True
    except Exception as e:
        print(f"Error restoring PDF index for '{pdf_name}': {e}")
        return False


def clear_pdf():
    """
    Reset the global retriever and PDF name.
    Call this when switching to a thread that has no PDF,
    or when creating a new chat — so the LLM acts as a general assistant.
    """
    global _retriever, _current_pdf_name
    _retriever        = None
    _current_pdf_name = None


def get_current_pdf_name() -> str | None:
    return _current_pdf_name


# ══════════════════════════════════════════════════════════════════════════════
# RAG Tool
# ══════════════════════════════════════════════════════════════════════════════

@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the uploaded PDF document.
    Use this when the user asks factual or conceptual questions that
    can be answered from the document.
    """
    if _retriever is None:
        return {
            "query":   query,
            "context": [],
            "error":   "No document loaded. Please upload a PDF first.",
        }

    retrieved_docs = _retriever.invoke(query)
    return {
        "query":    query,
        "context":  [doc.page_content for doc in retrieved_docs],
        "metadata": [doc.metadata     for doc in retrieved_docs],
    }


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph — safe message trimming + chat node
# ══════════════════════════════════════════════════════════════════════════════

tools          = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_system_prompt() -> str:
    if _current_pdf_name:
        return (
            f"You are a helpful document assistant. The user has uploaded: **{_current_pdf_name}**.\n\n"
            f"For greetings and casual messages: respond naturally and warmly.\n"
            f"For factual or conceptual questions: use the rag_tool to retrieve context, "
            f"then answer ONLY based on the retrieved content.\n"
            f"If the context is not relevant, say: \"I don't have that information in {_current_pdf_name}.\"\n\n"
            f"Always be friendly and cite the document when answering factual questions."
        )
    else:
        return (
            "You are a helpful general assistant. No document has been uploaded. "
            "Answer general questions naturally and helpfully. "
            "If the user asks about a specific document, suggest they upload a PDF using the sidebar."
        )


# How many messages to keep in history sent to LLM.
# Must be high enough to never split a tool call pair.
# One full RAG exchange = 4 messages (Human → AIToolCall → Tool → AIAnswer).
# 8 = ~2 full exchanges, safe buffer.
MAX_HISTORY = 8


def _trim_messages_safely(messages: list, max_messages: int) -> list:
    """
    Trim message history to at most max_messages, but NEVER split a
    tool call pair. An AIMessage with tool_calls must always be followed
    by its ToolMessage(s) — splitting them causes Groq 400 errors.
    """
    if len(messages) <= max_messages:
        return messages

    trimmed = list(messages[-max_messages:])

    # Rule 1: Never start with an orphaned ToolMessage
    # (its AIMessage was cut off by the slice)
    while trimmed and isinstance(trimmed[0], ToolMessage):
        trimmed = trimmed[1:]

    # Rule 2: Never start with an AIMessage whose ToolMessage was cut off
    while trimmed:
        first = trimmed[0]
        if (
            isinstance(first, AIMessage)
            and hasattr(first, "tool_calls")
            and first.tool_calls
        ):
            # The next message must be a ToolMessage — if not, it's orphaned
            if len(trimmed) < 2 or not isinstance(trimmed[1], ToolMessage):
                trimmed = trimmed[1:]
            else:
                break
        else:
            break

    return trimmed


def chat_node(state: ChatState):
    system   = SystemMessage(content=_build_system_prompt())
    all_msgs = state["messages"]
    trimmed  = _trim_messages_safely(all_msgs, MAX_HISTORY)
    response = llm_with_tools.invoke([system] + trimmed)
    return {"messages": [response]}


# ══════════════════════════════════════════════════════════════════════════════
# SQLite — checkpointer + thread metadata
# ══════════════════════════════════════════════════════════════════════════════

conn         = sqlite3.connect("rag_chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_metadata (
        thread_id   TEXT PRIMARY KEY,
        thread_name TEXT,
        pdf_name    TEXT,
        created_at  TIMESTAMP,
        updated_at  TIMESTAMP
    )
""")
conn.commit()

# Compile graph
_graph = StateGraph(ChatState)
_graph.add_node("chat_node", chat_node)
_graph.add_node("tools",     ToolNode(tools))
_graph.add_edge(START, "chat_node")
_graph.add_conditional_edges("chat_node", tools_condition)
_graph.add_edge("tools", "chat_node")

chatbot = _graph.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
# Thread metadata helpers
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_all_threads() -> list[dict]:
    rows = conn.execute("""
        SELECT thread_id, thread_name, pdf_name, created_at, updated_at
        FROM thread_metadata
        ORDER BY updated_at DESC
    """).fetchall()
    return [
        {
            "thread_id":   r[0],
            "thread_name": r[1],
            "pdf_name":    r[2],
            "created_at":  r[3],
            "updated_at":  r[4],
        }
        for r in rows
    ]


def create_thread_metadata(thread_id: str, thread_name: str = "New Chat", pdf_name: str = None):
    now = datetime.now().isoformat()
    conn.execute("""
        INSERT OR IGNORE INTO thread_metadata (thread_id, thread_name, pdf_name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    """, (thread_id, thread_name, pdf_name, now, now))
    conn.commit()


def update_thread_metadata(thread_id: str, thread_name: str = None, pdf_name: str = None):
    now = datetime.now().isoformat()
    if thread_name and pdf_name:
        conn.execute(
            "UPDATE thread_metadata SET thread_name=?, pdf_name=?, updated_at=? WHERE thread_id=?",
            (thread_name, pdf_name, now, thread_id)
        )
    elif thread_name:
        conn.execute(
            "UPDATE thread_metadata SET thread_name=?, updated_at=? WHERE thread_id=?",
            (thread_name, now, thread_id)
        )
    elif pdf_name:
        conn.execute(
            "UPDATE thread_metadata SET pdf_name=?, updated_at=? WHERE thread_id=?",
            (pdf_name, now, thread_id)
        )
    else:
        conn.execute(
            "UPDATE thread_metadata SET updated_at=? WHERE thread_id=?",
            (now, thread_id)
        )
    conn.commit()


def generate_thread_name(first_message: str) -> str:
    """Generate a short title from the first message of a thread."""
    try:
        prompt = (
            f'Generate a short, concise title (max 5 words) for a conversation '
            f'that starts with: "{first_message[:100]}"\n'
            f'Respond with ONLY the title, nothing else.'
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().strip('"').strip("'")[:50]
    except Exception:
        words = first_message.split()[:5]
        return " ".join(words) + ("..." if len(first_message.split()) > 5 else "")


def check_if_thread_has_messages(thread_id: str) -> bool:
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state  = chatbot.get_state(config)
        return bool(state and state.values and state.values.get("messages"))
    except Exception:
        return False


def load_thread_messages(thread_id: str) -> list[dict]:
    """
    Load all messages for a thread from the LangGraph checkpointer.
    Handles human, ai, and tool messages so tool calls survive page reloads.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state  = chatbot.get_state(config)

        if not (state and state.values and "messages" in state.values):
            return []

        result       = []
        last_ai_args = {}

        for msg in state.values["messages"]:
            if not hasattr(msg, "type"):
                continue

            # Human
            if msg.type == "human" and msg.content:
                result.append({"role": "user", "content": msg.content})

            # AI — capture tool call args for the ToolMessage that follows
            elif msg.type == "ai":
                last_ai_args = {}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        last_ai_args = tc.get("args", {})
                # Only append if there is visible text content
                if msg.content and isinstance(msg.content, str):
                    result.append({"role": "assistant", "content": msg.content})

            # Tool — rebuild the tool_call dict for the frontend
            elif msg.type == "tool":
                raw = msg.content
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        raw = {"context": [raw]}

                query_sent     = last_ai_args.get("query", "")
                context_chunks = []

                if isinstance(raw, dict):
                    context_chunks = raw.get("context", [])
                    if not query_sent:
                        query_sent = raw.get("query", "")

                result.append({
                    "role":   "tool_call",
                    "query":  query_sent,
                    "chunks": context_chunks,
                })

        return result

    except Exception as e:
        print(f"Error loading thread messages: {e}")
        return []
    