"""
PaperMind — RAG Chatbot Frontend
Run: streamlit run app.py
"""

import os
import uuid
import json
import tempfile
from datetime import datetime, timedelta

import streamlit as st
from langchain_core.messages import HumanMessage

from rag_backend import (
    chatbot,
    load_pdf,
    get_current_pdf_name,
    restore_pdf_for_thread,
    clear_pdf,
    retrieve_all_threads,
    create_thread_metadata,
    update_thread_metadata,
    generate_thread_name,
    check_if_thread_has_messages,
    load_thread_messages,
)


# ══════════════════════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PaperMind — Document Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# Theme Definitions
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "Midnight": {
        "emoji":        "🌑",
        "bg_deep":      "#0f1117",
        "bg_panel":     "#161922",
        "bg_card":      "#1c2030",
        "bg_hover":     "#232840",
        "border":       "#2a3050",
        "border_light": "#3a4568",
        "accent":       "#e8b84b",
        "accent_dim":   "#a07c2a",
        "accent_glow":  "rgba(232,184,75,0.15)",
        "accent_text":  "#0f1117",
        "text_primary": "#f0eee8",
        "text_muted":   "#8890a8",
        "text_faint":   "#404868",
        "user_bg":      "#181e30",
        "ai_bg":        "#131620",
        "gradient":     "radial-gradient(ellipse at 20% 10%, #1a1f35 0%, #0f1117 60%)",
        "dot_color":    "rgba(255,255,255,0.025)",
    },
    "Forest": {
        "emoji":        "🌿",
        "bg_deep":      "#0d1612",
        "bg_panel":     "#121e18",
        "bg_card":      "#182620",
        "bg_hover":     "#1e3028",
        "border":       "#243a2e",
        "border_light": "#2f4e3a",
        "accent":       "#5dca8a",
        "accent_dim":   "#3a8a5a",
        "accent_glow":  "rgba(93,202,138,0.15)",
        "accent_text":  "#0d1612",
        "text_primary": "#e8f0ea",
        "text_muted":   "#7aaa88",
        "text_faint":   "#3a5242",
        "user_bg":      "#141f18",
        "ai_bg":        "#101810",
        "gradient":     "radial-gradient(ellipse at 10% 30%, #152018 0%, #0d1612 65%)",
        "dot_color":    "rgba(93,202,138,0.03)",
    },
    "Ocean": {
        "emoji":        "🌊",
        "bg_deep":      "#081420",
        "bg_panel":     "#0d1d2e",
        "bg_card":      "#122538",
        "bg_hover":     "#172e45",
        "border":       "#1e3a52",
        "border_light": "#2a4e6a",
        "accent":       "#38bdf8",
        "accent_dim":   "#1e7aa8",
        "accent_glow":  "rgba(56,189,248,0.15)",
        "accent_text":  "#081420",
        "text_primary": "#e0f2fe",
        "text_muted":   "#6aaac8",
        "text_faint":   "#2a4858",
        "user_bg":      "#0e1e30",
        "ai_bg":        "#091520",
        "gradient":     "radial-gradient(ellipse at 80% 0%, #0d2540 0%, #081420 60%)",
        "dot_color":    "rgba(56,189,248,0.03)",
    },
    "Crimson": {
        "emoji":        "🔴",
        "bg_deep":      "#140c10",
        "bg_panel":     "#1e1018",
        "bg_card":      "#281420",
        "bg_hover":     "#32182a",
        "border":       "#3a1e28",
        "border_light": "#502838",
        "accent":       "#f87171",
        "accent_dim":   "#b84040",
        "accent_glow":  "rgba(248,113,113,0.15)",
        "accent_text":  "#140c10",
        "text_primary": "#fde8e8",
        "text_muted":   "#c88898",
        "text_faint":   "#5a2e38",
        "user_bg":      "#1c1018",
        "ai_bg":        "#120c10",
        "gradient":     "radial-gradient(ellipse at 90% 10%, #2a1020 0%, #140c10 65%)",
        "dot_color":    "rgba(248,113,113,0.03)",
    },
    "Violet": {
        "emoji":        "💜",
        "bg_deep":      "#100d1a",
        "bg_panel":     "#161228",
        "bg_card":      "#1e1835",
        "bg_hover":     "#251e42",
        "border":       "#2e2650",
        "border_light": "#3e3468",
        "accent":       "#a78bfa",
        "accent_dim":   "#7c5abe",
        "accent_glow":  "rgba(167,139,250,0.15)",
        "accent_text":  "#100d1a",
        "text_primary": "#ede8ff",
        "text_muted":   "#9888c8",
        "text_faint":   "#453a6a",
        "user_bg":      "#16122a",
        "ai_bg":        "#0f0d18",
        "gradient":     "radial-gradient(ellipse at 30% 5%, #1e1840 0%, #100d1a 60%)",
        "dot_color":    "rgba(167,139,250,0.03)",
    },
    "Ember": {
        "emoji":        "🔥",
        "bg_deep":      "#130e08",
        "bg_panel":     "#1e160a",
        "bg_card":      "#281e0e",
        "bg_hover":     "#342514",
        "border":       "#3a2c18",
        "border_light": "#503c22",
        "accent":       "#fb923c",
        "accent_dim":   "#b85a1a",
        "accent_glow":  "rgba(251,146,60,0.15)",
        "accent_text":  "#130e08",
        "text_primary": "#fef3e2",
        "text_muted":   "#c89860",
        "text_faint":   "#5a3e20",
        "user_bg":      "#1c1510",
        "ai_bg":        "#110e08",
        "gradient":     "radial-gradient(ellipse at 5% 90%, #251808 0%, #130e08 65%)",
        "dot_color":    "rgba(251,146,60,0.03)",
    },
}

DEFAULT_THEME = "Midnight"


# ══════════════════════════════════════════════════════════════════════════════
# CSS Builder — injects theme variables dynamically
# ══════════════════════════════════════════════════════════════════════════════

def build_css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;1,500&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {{
    --bg-deep:      {t['bg_deep']};
    --bg-panel:     {t['bg_panel']};
    --bg-card:      {t['bg_card']};
    --bg-hover:     {t['bg_hover']};
    --border:       {t['border']};
    --border-light: {t['border_light']};
    --accent:       {t['accent']};
    --accent-dim:   {t['accent_dim']};
    --accent-glow:  {t['accent_glow']};
    --accent-text:  {t['accent_text']};
    --text-primary: {t['text_primary']};
    --text-muted:   {t['text_muted']};
    --text-faint:   {t['text_faint']};
    --user-bg:      {t['user_bg']};
    --ai-bg:        {t['ai_bg']};
    --tool-bg:      color-mix(in srgb, {t['bg_deep']} 80%, teal 20%);
    --tool-border:  color-mix(in srgb, {t['border']} 60%, teal 40%);
    --tool-accent:  #4ecdc4;
    --gradient:     {t['gradient']};
    --dot-color:    {t['dot_color']};
}}

html, body, [class*="css"] {{
    font-family: 'Source Sans 3', sans-serif;
    color: var(--text-primary);
}}

/* ── App background ── */
.stApp {{
    background: var(--gradient);
    min-height: 100vh;
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image: radial-gradient(var(--dot-color) 1px, transparent 1px);
    background-size: 36px 36px;
    pointer-events: none;
    z-index: 0;
}}

/* ══ Sidebar ══ */
[data-testid="stSidebar"] {{
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-light) !important;
    backdrop-filter: blur(12px);
}}
[data-testid="stSidebar"] > div {{
    padding-top: 0 !important;
}}

/* Sidebar inner scroll container */
[data-testid="stSidebarContent"] {{
    padding: 1.2rem 1rem 2rem 1rem !important;
}}

/* ── Brand ── */
.brand {{
    font-family: 'Playfair Display', serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
    padding: 0.6rem 0 1rem 0;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.brand em {{ color: var(--accent); font-style: italic; }}
.brand-dot {{
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px var(--accent-glow);
    animation: pulse 2.5s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%  {{ opacity: 0.5; transform: scale(0.7); }}
}}

/* ── Section labels ── */
.section-label {{
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-faint);
    margin: 1.25rem 0 0.5rem 0;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 5px;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-light) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s, background 0.2s !important;
}}
[data-testid="stFileUploader"]:hover {{
    border-color: var(--accent-dim) !important;
    background: var(--bg-hover) !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background: var(--border-light) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    transition: background 0.2s !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover {{
    background: var(--accent-dim) !important;
    color: #fff !important;
}}
[data-testid="stFileUploaderFile"] {{
    background: var(--bg-hover) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
}}

/* PDF badge */
@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 4px var(--accent-glow); }}
    50%       {{ box-shadow: 0 0 18px var(--accent-glow); }}
}}
.pdf-badge {{
    background: var(--accent-glow);
    border: 1px solid var(--accent-dim);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: var(--accent);
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0.6rem 0 0.8rem 0;
    word-break: break-all;
    font-weight: 500;
    animation: glowPulse 3s infinite ease-in-out;
}}

/* ── Theme swatches ── */
.theme-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    margin: 0.5rem 0 0.25rem 0;
}}
.theme-swatch {{
    border-radius: 8px;
    padding: 6px 4px;
    font-size: 0.72rem;
    text-align: center;
    cursor: pointer;
    border: 2px solid transparent;
    transition: all 0.15s ease;
    color: var(--text-primary);
    background: var(--bg-card);
}}
.theme-swatch:hover   {{ border-color: var(--accent-dim); transform: translateY(-1px); }}
.theme-swatch.active  {{ border-color: var(--accent); background: var(--accent-glow); }}

/* ── Buttons ── */
.stButton > button {{
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-light) !important;
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    padding: 0.45rem 0.8rem !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}}
.stButton > button:hover {{
    background: var(--bg-hover) !important;
    border-color: var(--accent-dim) !important;
    color: var(--text-primary) !important;
    transform: translateX(2px);
}}
.new-chat-wrap .stButton > button {{
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: var(--accent-text) !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    text-align: center !important;
    padding: 0.55rem 1rem !important;
    letter-spacing: 0.2px !important;
    transform: none !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}}
.new-chat-wrap .stButton > button:hover {{
    filter: brightness(1.1) !important;
    box-shadow: 0 4px 20px var(--accent-glow) !important;
}}
.active-thread .stButton > button {{
    background: var(--accent-glow) !important;
    border-color: var(--accent-dim) !important;
    color: var(--accent) !important;
    transform: none !important;
}}
.active-thread .stButton > button::before {{
    content: "";
}}

/* Sidebar info boxes */
.sidebar-stat {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 4px;
}}
.sidebar-stat strong {{ color: var(--accent); font-weight: 600; }}

/* ══ Main header ══ */
.main-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 2px;
}}
.main-sub {{ font-size: 0.82rem; color: var(--text-muted); margin-bottom: 1rem; }}

/* ══ Chat messages ══ */
[data-testid="stChatMessage"] {{
    border-radius: 14px !important;
    padding: 0.85rem 1.1rem !important;
    margin-bottom: 0.5rem !important;
    border: 1px solid transparent !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}}
[data-testid="stChatMessage"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.3) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: var(--user-bg) !important;
    border-color: var(--border-light) !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
    background: var(--ai-bg) !important;
    border-color: var(--border) !important;
}}

/* ══ Tool call ══ */
.tool-call-box {{
    background: var(--tool-bg);
    border: 1px solid var(--tool-border);
    border-left: 3px solid var(--tool-accent);
    border-radius: 8px;
    padding: 9px 14px;
    margin: 4px 0 2px 0;
    font-size: 0.82rem;
}}
.tool-call-header {{
    color: var(--tool-accent);
    font-weight: 600;
    font-size: 0.78rem;
    margin-bottom: 3px;
}}
.tool-query {{ color: var(--text-muted); font-size: 0.8rem; }}

[data-testid="stExpander"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--tool-border) !important;
    border-radius: 8px !important;
    margin-top: 4px !important;
}}

/* ══ Chat input ══ */
[data-testid="stChatInput"] {{
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 14px !important;
}}
[data-testid="stChatInput"] textarea {{
    color: var(--text-primary) !important;
    font-family: 'Source Sans 3', sans-serif !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--text-faint) !important; }}

/* ══ Welcome card ══ */
.welcome-card {{
    background: var(--bg-panel);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin: 3rem auto;
    max-width: 540px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.3);
}}
.welcome-icon  {{ font-size: 3rem; margin-bottom: 1rem; }}
.welcome-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 0.6rem;
}}
.welcome-body  {{ color: var(--text-muted); font-size: 0.9rem; line-height: 1.75; }}
.hint-row      {{ display: flex; gap: 0.6rem; justify-content: center; flex-wrap: wrap; margin-top: 1.5rem; }}
.hint-pill {{
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.78rem;
    color: var(--text-muted);
    cursor: default;
    transition: border-color 0.2s, color 0.2s;
}}
.hint-pill:hover {{ border-color: var(--accent-dim); color: var(--accent); }}

/* ══ Misc ══ */
hr {{ border-color: var(--border-light) !important; margin: 0.75rem 0 !important; }}
.stSpinner > div {{ border-top-color: var(--accent) !important; }}
.stCaption {{ color: var(--text-faint) !important; font-size: 0.75rem !important; }}
[data-testid="stSidebar"] .stCaption {{ color: var(--text-muted) !important; }}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border-light); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent-dim); }}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def create_new_chat() -> str:
    tid = str(uuid.uuid4())
    create_thread_metadata(tid, "New Chat", pdf_name=None)
    return tid


def get_thread_info(tid: str, threads: list) -> dict | None:
    return next((t for t in threads if t["thread_id"] == tid), None)


def switch_thread(thread_id: str):
    st.session_state.thread_id       = thread_id
    st.session_state.message_history = load_thread_messages(thread_id)

    thread_info = get_thread_info(thread_id, st.session_state.chat_threads)
    pdf_name    = (thread_info or {}).get("pdf_name")

    if pdf_name:
        restored = restore_pdf_for_thread(pdf_name)
        st.session_state.active_pdf_name = pdf_name if restored else None
        if not restored:
            st.session_state.pdf_restore_warning = pdf_name
    else:
        clear_pdf()
        st.session_state.active_pdf_name = None


def categorize_threads(threads: list) -> dict:
    now        = datetime.now()
    today      = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday  = today - timedelta(days=1)
    last_week  = today - timedelta(days=7)
    last_month = today - timedelta(days=30)
    cats = {
        "Today": [], "Yesterday": [],
        "Previous 7 Days": [], "Previous 30 Days": [], "Older": []
    }
    for t in threads:
        try:
            ts = datetime.fromisoformat(t["updated_at"])
            if   ts >= today:      cats["Today"].append(t)
            elif ts >= yesterday:  cats["Yesterday"].append(t)
            elif ts >= last_week:  cats["Previous 7 Days"].append(t)
            elif ts >= last_month: cats["Previous 30 Days"].append(t)
            else:                  cats["Older"].append(t)
        except Exception:
            cats["Older"].append(t)
    return {k: v for k, v in cats.items() if v}


def render_tool_call(query: str, chunks: list):
    st.markdown(f"""
    <div class="tool-call-box">
        <div class="tool-call-header">⚡ RAG Tool Called</div>
        <div class="tool-query">Query: <b>"{query}"</b> &nbsp;·&nbsp; {len(chunks)} passage(s) retrieved</div>
    </div>
    """, unsafe_allow_html=True)
    if chunks:
        with st.expander(f"📄 View retrieved passages ({len(chunks)})", expanded=False):
            for i, chunk in enumerate(chunks, 1):
                st.markdown(f"**Passage {i}**")
                st.caption(chunk[:800] + ("…" if len(chunk) > 800 else ""))


# ══════════════════════════════════════════════════════════════════════════════
# Session State Init
# ══════════════════════════════════════════════════════════════════════════════

if "theme" not in st.session_state:
    st.session_state.theme = DEFAULT_THEME

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

if "thread_id" not in st.session_state:
    if st.session_state.chat_threads:
        st.session_state.thread_id = st.session_state.chat_threads[0]["thread_id"]
    else:
        st.session_state.thread_id = create_new_chat()
        st.session_state.chat_threads = retrieve_all_threads()

if "message_history" not in st.session_state:
    st.session_state.message_history = load_thread_messages(st.session_state.thread_id)

if "active_pdf_name" not in st.session_state:
    st.session_state.active_pdf_name = None

if "pdf_restore_warning" not in st.session_state:
    st.session_state.pdf_restore_warning = None

# Restore PDF on page reload
if "pdf_restored_on_load" not in st.session_state:
    try:
        thread_info = get_thread_info(
            st.session_state.thread_id,
            st.session_state.chat_threads
        )
        pdf_name = (thread_info or {}).get("pdf_name")
        if pdf_name:
            restored = restore_pdf_for_thread(pdf_name)
            st.session_state.active_pdf_name = pdf_name if restored else None
            if not restored:
                st.session_state.pdf_restore_warning = pdf_name
        else:
            clear_pdf()
            st.session_state.active_pdf_name = None
    except Exception as e:
        print(f"PDF restore on load failed: {e}")
        st.session_state.active_pdf_name = None
    st.session_state.pdf_restored_on_load = True


# ── Inject CSS for active theme ───────────────────────────────────────────────
current_theme = THEMES[st.session_state.theme]
st.markdown(build_css(current_theme), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Brand ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="brand">'
        '<span class="brand-dot"></span>'
        'Paper<em>Mind</em>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Re-upload warning ─────────────────────────────────────────────────────
    if st.session_state.pdf_restore_warning:
        st.warning(
            f"⚠️ Could not restore **{st.session_state.pdf_restore_warning}**. "
            "Please re-upload the PDF.",
            icon="⚠️",
        )
        st.session_state.pdf_restore_warning = None

    # ── Stats row ─────────────────────────────────────────────────────────────
    total   = len(st.session_state.chat_threads)
    pdf_str = st.session_state.active_pdf_name or "None"
    short_pdf = (pdf_str[:18] + "…") if len(pdf_str) > 18 else pdf_str
    st.markdown(f"""
    <div class="sidebar-stat">
        <span>💬 Conversations</span>
        <strong>{total}</strong>
    </div>
    <div class="sidebar-stat">
        <span>📄 Active PDF</span>
        <strong>{short_pdf}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Theme Switcher ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🎨 Theme</div>', unsafe_allow_html=True)

    # Build swatch grid using columns
    theme_names = list(THEMES.keys())
    cols = st.columns(3)
    for i, name in enumerate(theme_names):
        t = THEMES[name]
        is_active = st.session_state.theme == name
        with cols[i % 3]:
            # Use a button styled as a swatch
            active_class = "active" if is_active else ""
            st.markdown(
                f'<div class="theme-swatch {active_class}" style="'
                f'background:linear-gradient(135deg,{t["bg_panel"]},{t["bg_deep"]});'
                f'border-color:{"" if not is_active else t["accent"]}'
                f'">{t["emoji"]}<br><span style="font-size:0.62rem;color:{t["text_muted"]}">{name}</span></div>',
                unsafe_allow_html=True,
            )
            if st.button(name, key=f"theme_{name}", use_container_width=True,
                         help=f"Switch to {name} theme"):
                st.session_state.theme = name
                st.rerun()

    # ── PDF Upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📂 Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload a PDF to enable document Q&A",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.active_pdf_name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Processing {uploaded_file.name}…"):
                pdf_name = load_pdf(tmp_path, original_filename=uploaded_file.name)
                os.unlink(tmp_path)

            st.session_state.active_pdf_name = pdf_name

            if not check_if_thread_has_messages(st.session_state.thread_id):
                update_thread_metadata(st.session_state.thread_id, pdf_name=pdf_name)
                st.session_state.chat_threads = retrieve_all_threads()

            st.rerun()

    if st.session_state.active_pdf_name:
        st.markdown(
            f'<div class="pdf-badge">✅&nbsp; {st.session_state.active_pdf_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("⬆️  No document loaded yet")

    # ── New Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">💬 Conversations</div>', unsafe_allow_html=True)
    st.markdown('<div class="new-chat-wrap">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        clear_pdf()
        st.session_state.active_pdf_name = None
        tid = create_new_chat()
        st.session_state.thread_id       = tid
        st.session_state.message_history = []
        st.session_state.chat_threads    = retrieve_all_threads()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Thread list ───────────────────────────────────────────────────────────
    for category, threads in categorize_threads(st.session_state.chat_threads).items():
        st.markdown(
            f'<div class="section-label">🕐 {category}</div>',
            unsafe_allow_html=True,
        )
        for thread in threads:
            tid       = thread["thread_id"]
            name      = thread["thread_name"] or "Untitled Chat"
            pdf       = thread.get("pdf_name") or ""
            is_active = tid == st.session_state.thread_id

            label = f"{'▸ ' if is_active else ''}{name}"
            if pdf:
                short = pdf[:18] + ("…" if len(pdf) > 18 else "")
                label += f"\n📄 {short}"

            wrapper = "active-thread" if is_active else ""
            st.markdown(f'<div class="{wrapper}">', unsafe_allow_html=True)
            if st.button(label, key=f"t_{tid}", use_container_width=True):
                switch_thread(tid)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Sidebar footer ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;font-size:0.65rem;color:{current_theme["text_faint"]};'
        f'padding-top:1rem;border-top:1px solid {current_theme["border"]};">'
        f'PaperMind · LangGraph + FAISS + Groq</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main Chat Area
# ══════════════════════════════════════════════════════════════════════════════

current_info = get_thread_info(st.session_state.thread_id, st.session_state.chat_threads)
chat_name    = (current_info or {}).get("thread_name") or "New Conversation"
chat_pdf     = (current_info or {}).get("pdf_name")    or ""

# Header
col_icon, col_title = st.columns([0.07, 0.93])
with col_icon:
    st.markdown("## 🧠")
with col_title:
    st.markdown(f'<div class="main-title">{chat_name}</div>', unsafe_allow_html=True)
    sub_parts = []
    if chat_pdf:
        sub_parts.append(f"📄 {chat_pdf}")
    if current_info and current_info.get("created_at"):
        try:
            dt = datetime.fromisoformat(current_info["created_at"])
            sub_parts.append(dt.strftime("%b %d, %Y · %I:%M %p"))
        except Exception:
            pass
    sub_line = " &nbsp;·&nbsp; ".join(sub_parts) if sub_parts else "Start a conversation below"
    st.markdown(f'<div class="main-sub">{sub_line}</div>', unsafe_allow_html=True)

st.divider()

# ── Message history ───────────────────────────────────────────────────────────
history = st.session_state.message_history

if history:
    count = sum(1 for m in history if m["role"] != "tool_call")
    st.caption(f"📝 {count} message{'s' if count != 1 else ''} in this conversation")

    for msg in history:
        if msg["role"] == "tool_call":
            render_tool_call(msg["query"], msg["chunks"])
        else:
            avatar = "👤" if msg["role"] == "user" else "🧠"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
else:
    if st.session_state.active_pdf_name:
        doc_hint = f"about <b>{st.session_state.active_pdf_name}</b>"
    else:
        doc_hint = "anything &mdash; or <b>upload a PDF</b> in the sidebar first"

    st.markdown(f"""
    <div class="welcome-card">
        <div class="welcome-icon">🧠</div>
        <div class="welcome-title">Ready to explore</div>
        <div class="welcome-body">
            Ask me {doc_hint}.<br>
            I'll retrieve the most relevant passages and answer from the document.
        </div>
        <div class="hint-row">
            <span class="hint-pill">What is this about?</span>
            <span class="hint-pill">Summarise key points</span>
            <span class="hint-pill">Explain the methodology</span>
            <span class="hint-pill">List the conclusions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────
placeholder = (
    f"Ask about {st.session_state.active_pdf_name}…"
    if st.session_state.active_pdf_name else
    "Ask me anything, or upload a PDF for document Q&A…"
)

user_input = st.chat_input(placeholder)

if user_input:
    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

    if not check_if_thread_has_messages(st.session_state.thread_id):
        with st.spinner("Naming conversation…"):
            thread_name = generate_thread_name(user_input)
        update_thread_metadata(
            st.session_state.thread_id,
            thread_name=thread_name,
            pdf_name=st.session_state.active_pdf_name,
        )
        st.session_state.chat_threads = retrieve_all_threads()
    else:
        update_thread_metadata(st.session_state.thread_id)

    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🧠"):
        tool_placeholder = st.empty()
        text_placeholder = st.empty()

        full_response = ""
        pending_tool  = {}
        tool_result   = {}

        for chunk, _meta in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            chunk_type = type(chunk).__name__

            if chunk_type == "AIMessageChunk":
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc in chunk.tool_call_chunks:
                        if tc.get("name"):
                            pending_tool["name"]     = tc["name"]
                            pending_tool["args_str"] = tc.get("args", "")
                        else:
                            pending_tool["args_str"] = (
                                pending_tool.get("args_str", "") + tc.get("args", "")
                            )
                    tool_placeholder.markdown(
                        '<div class="tool-call-box">'
                        '<div class="tool-call-header">⚡ Calling RAG Tool…</div>'
                        '<div class="tool-query">🔍 Searching document for relevant passages…</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                if isinstance(chunk.content, str) and chunk.content:
                    full_response += chunk.content
                    text_placeholder.markdown(full_response + "▌")

            elif chunk_type == "ToolMessage":
                raw = chunk.content
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except Exception:
                        raw = {"context": [raw]}

                query_sent = pending_tool.get("args_str", "")
                try:
                    args_parsed = json.loads(query_sent)
                    query_sent  = args_parsed.get("query", query_sent)
                except Exception:
                    pass

                context_chunks = []
                if isinstance(raw, dict):
                    context_chunks = raw.get("context", [])
                    if not query_sent:
                        query_sent = raw.get("query", "")

                tool_result = {"query": query_sent, "chunks": context_chunks}

                with tool_placeholder.container():
                    render_tool_call(query_sent, context_chunks)

        text_placeholder.markdown(full_response)

    if tool_result:
        st.session_state.message_history.append({
            "role":   "tool_call",
            "query":  tool_result.get("query",  ""),
            "chunks": tool_result.get("chunks", []),
        })

    if full_response.strip():
        st.session_state.message_history.append({
            "role":    "assistant",
            "content": full_response,
        })

    st.rerun()
