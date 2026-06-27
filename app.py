#!/usr/bin/env python3
"""
app.py — Tariff Navigator: Customs Classification RAG Application

A professional Streamlit interface for Ethiopian customs transit professionals
to accurately classify imported items using the HS 2022 rulebook and Gemini AI.

Architecture:
    User Query (text + optional image)
        → ChromaDB semantic search (retrieve relevant tariff rules)
        → Gemini LLM (classify item with retrieved context)
        → Structured output (Tariff Placement / HS Code / Justification)
"""

import sqlite3
import json
from datetime import datetime
import hashlib
import html
import os
import random
import sys
import time
from pathlib import Path

import streamlit as st

try:
    import chromadb
    from google import genai
    from google.genai import types as genai_types
    from PIL import Image
    from config import get_int_setting, get_path_setting, get_setting
except ImportError as e:
    st.error(f"Missing dependency: {e}. Run: `pip install -r requirements.txt`")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DB_DIR = get_path_setting("DB_DIR", "./db")
COLLECTION_NAME = get_setting("COLLECTION_NAME", "tariff_rules")
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "gemini-embedding-001")
GENERATION_MODEL = get_setting("GENERATION_MODEL", "gemini-3.5-flash")
TOP_K_RESULTS = get_int_setting("TOP_K_RESULTS", 15)  # Number of context chunks to retrieve


def get_api_key() -> str | None:
    """Read Gemini API credentials from Streamlit secrets, session state override, or environment."""
    if "custom_api_key" in st.session_state and st.session_state["custom_api_key"]:
        return st.session_state["custom_api_key"]

    secret_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    for key in secret_keys:
        try:
            if key in st.secrets and st.secrets[key]:
                return st.secrets[key]
        except Exception:
            pass

    return get_setting("GEMINI_API_KEY") or get_setting("GOOGLE_API_KEY")

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tariff Navigator — HS Classification",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — Professional Typography & Layout
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* ── Typography ────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Main Container ────────────────────────────────── */
    .block-container {
        max-width: 920px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* ── Header ────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 2.5rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.03em;
        margin: 0 0 0.4rem;
    }
    .app-header .subtitle {
        font-size: 0.95rem;
        font-weight: 400;
        color: #64748B;
        letter-spacing: 0.01em;
    }
    .app-header .badge {
        display: inline-block;
        margin-top: 0.75rem;
        padding: 0.25rem 0.85rem;
        background: linear-gradient(135deg, #0A66C2 0%, #0E7AE6 100%);
        color: white;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* ── Status Bar ────────────────────────────────────── */
    .status-bar {
        display: flex;
        justify-content: center;
        gap: 2rem;
        padding: 0.75rem 0;
        margin-bottom: 2rem;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: #64748B;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot.green { background: #10B981; }
    .status-dot.amber { background: #F59E0B; }
    .status-dot.red { background: #EF4444; }

    /* ── Input Section ─────────────────────────────────── */
    .input-section {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }

    /* ── Streamlit Overrides ───────────────────────────── */
    .stTextArea textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        background: #FAFBFC !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextArea textarea:focus {
        border-color: #0A66C2 !important;
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.08) !important;
    }

    div.stButton > button {
        width: 100%;
        padding: 0.8rem 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: linear-gradient(135deg, #0A66C2 0%, #0B4F9E 100%);
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0B4F9E 0%, #083A75 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(10, 102, 194, 0.3);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ── Results Section ───────────────────────────────── */
    .result-container {
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 2rem;
    }
    .result-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.25rem 2rem;
        color: white;
    }
    .result-header h3 {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .result-body {
        padding: 2rem;
    }

    .result-section {
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #F1F5F9;
    }
    .result-section:last-child {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }
    .result-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #0A66C2;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    .result-section-content {
        font-size: 0.92rem;
        line-height: 1.75;
        color: #334155;
    }

    .hs-code-display {
        display: inline-block;
        padding: 0.6rem 1.5rem;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        border-radius: 8px;
        letter-spacing: 0.05em;
        margin: 0.5rem 0;
    }

    /* ── File Uploader ─────────────────────────────────── */
    .stFileUploader > div {
        border-radius: 8px !important;
    }

    /* ── Expander (Context) ────────────────────────────── */
    .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
    }

    /* ── Spinner ───────────────────────────────────────── */
    .stSpinner > div {
        border-color: #0A66C2 transparent transparent transparent !important;
    }

    /* ── Footer ────────────────────────────────────────── */
    .app-footer {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-top: 3rem;
        border-top: 1px solid #E5E7EB;
        font-size: 0.75rem;
        color: #94A3B8;
    }

    /* ── Hide Streamlit Defaults ───────────────────────── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Query Cache (SQLite)
# ─────────────────────────────────────────────────────────────


class QueryCache:
    def __init__(self, db_path: str = "./db/query_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        query_text TEXT,
                        response_json TEXT,
                        created_at TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            # Silently fall back if DB is locked or unable to write
            pass

    def _get_hash(self, query_text: str, image_bytes: bytes = None) -> str:
        hasher = hashlib.sha256(query_text.encode('utf-8'))
        if image_bytes:
            hasher.update(image_bytes)
        return hasher.hexdigest()

    def get(self, query_text: str, image_bytes: bytes = None) -> dict | None:
        key = self._get_hash(query_text, image_bytes)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response_json FROM cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception:
            pass
        return None

    def set(self, query_text: str, response_data: dict, image_bytes: bytes = None):
        key = self._get_hash(query_text, image_bytes)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, query_text, response_json, created_at) VALUES (?, ?, ?, ?)",
                    (key, query_text, json.dumps(response_data), datetime.utcnow().isoformat())
                )
                conn.commit()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Service Initialization (cached)
# ─────────────────────────────────────────────────────────────


@st.cache_resource
def init_genai_client():
    """Initialize Google GenAI client (cached across reruns)."""
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB client and get the tariff_rules collection."""
    if not DB_DIR.exists():
        return None, 0
    try:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        return collection, count
    except Exception:
        return None, 0


# ─────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────


def query_chromadb(
    collection, client: genai.Client, query_text: str, n_results: int = TOP_K_RESULTS
) -> list[dict]:
    """
    Embed the query text and retrieve the most relevant tariff rule chunks.
    Returns a list of dicts with 'text', 'page_number', and 'distance'.
    """
    # Embed the query
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query_text,
    )
    query_embedding = result.embeddings[0].values

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append(
            {
                "text": results["documents"][0][i],
                "page_number": results["metadatas"][0][i].get("page_number", "?"),
                "distance": results["distances"][0][i],
            }
        )

    return chunks


def _is_retryable_gemini_error(error: Exception) -> bool:
    """Return True for transient Gemini/API failures worth retrying."""
    error_str = str(error).upper()
    retry_markers = (
        "429",
        "503",
        "500",
        "502",
        "504",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
    )
    return any(marker in error_str for marker in retry_markers)


def _retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff with jitter for transient model failures."""
    base_delay = min(45, 4 * (2**attempt))
    jitter = random.uniform(0.5, 1.5)
    return base_delay * jitter


def build_classification_prompt(
    item_description: str, retrieved_chunks: list[dict]
) -> str:
    """Build the system + context prompt for Gemini classification."""
    # Format retrieved context
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_blocks.append(
            f"[Source: Page {chunk['page_number']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are an expert international customs classification specialist with deep knowledge of the Harmonized System (HS) 2022, the World Customs Organization's Explanatory Notes, General Rules of Interpretation (GRI), and Ethiopian customs regulations.

Your task is to classify the following imported item based STRICTLY on the official tariff rules provided below. You must be precise, legally defensible, and cite specific sources.

═══════════════════════════════════════════════
ITEM TO CLASSIFY:
═══════════════════════════════════════════════
{item_description}

═══════════════════════════════════════════════
OFFICIAL TARIFF RULES (Retrieved from HS 2022 Explanatory Notes):
═══════════════════════════════════════════════
{context_text}

═══════════════════════════════════════════════
CLASSIFICATION INSTRUCTIONS:
═══════════════════════════════════════════════
Analyze the item systematically using the General Rules of Interpretation (GRI 1 through GRI 6). Cite relevant Section/Chapter Notes and GRI rules.

Return ONLY a valid JSON object matching the following structure, without any markdown formatting, backticks, or code blocks (like ```json):
{{
  "conclusion": "A short, direct conclusion in 1-3 sentences stating the outcome and if definitive/provisional.",
  "placement": "Hierarchy path (Section, Chapter, Heading, Subheading)",
  "hs_code": "The final numeric code (e.g. 8471.30.00)",
  "justification": "Detailed explanation of why it is classified here, citing specific GRI rules, Chapter/Section Notes, or Explanatory Notes. Limit to under 100 words."
}}
"""
    return prompt


def classify_item(
    client: genai.Client,
    collection,
    item_description: str,
    image_bytes: bytes | None = None,
    image_mime: str | None = None,
) -> dict:
    """
    Full RAG pipeline: retrieve context → build prompt → call Gemini → return result.

    Returns dict with keys: 'response', 'chunks', 'elapsed'
    """
    start = time.time()

    # Step 1: Retrieve relevant tariff rules from ChromaDB
    chunks = query_chromadb(collection, client, item_description)

    # Step 2: Build classification prompt
    prompt = build_classification_prompt(item_description, chunks)

    # Step 3: Build content parts
    parts = [genai_types.Part.from_text(text=prompt)]

    # Add image if provided
    if image_bytes and image_mime:
        parts.append(
            genai_types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
        )
        parts.append(
            genai_types.Part.from_text(
                text="\n\nThe image above shows the item to be classified. "
                "Use visual details (markings, components, materials, form factor) "
                "to refine your classification alongside the text description."
            )
        )

    # Step 4: Call Gemini (with retry for rate limits)
    last_error = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=[genai_types.Content(parts=parts)],
            )
            elapsed = time.time() - start
            return {
                "response": response.text if response.text else "No response generated.",
                "chunks": chunks,
                "elapsed": elapsed,
            }
        except Exception as e:
            last_error = e
            if _is_retryable_gemini_error(e):
                if attempt < 4:
                    wait = _retry_delay_seconds(attempt)
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    "Gemini generation is temporarily unavailable after "
                    "multiple retries. Please try again in a few minutes."
                ) from e
            else:
                raise

    raise last_error


# ─────────────────────────────────────────────────────────────
# Response Parsing & Display
# ─────────────────────────────────────────────────────────────


def parse_and_display_response(response_text: str):
    """Parse the response (supporting JSON and fallback text splits) and display with styled HTML."""
    sections = {
        "conclusion": "",
        "placement": "",
        "hs_code": "",
        "justification": "",
    }

    # Clean markdown wrappers if returned
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        sections["conclusion"] = data.get("conclusion", "").strip()
        sections["placement"] = data.get("placement", "").strip()
        sections["hs_code"] = data.get("hs_code", "").strip()
        sections["justification"] = data.get("justification", "").strip()
    except Exception:
        # Fallback to regex split if JSON loading fails
        text = response_text
        import re

        # Pattern for Section 0
        conclusion_match = re.search(
            r"(?:SECTION 0|QUICK CONCLUSION)[^\n]*\n(.*?)(?=(?:SECTION 1|TARIFF PLACEMENT))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if conclusion_match:
            sections["conclusion"] = conclusion_match.group(1).strip()

        # Pattern for Section 1
        placement_match = re.search(
            r"(?:SECTION 1|TARIFF PLACEMENT)[^\n]*\n(.*?)(?=(?:SECTION 2|FINAL HS CODE))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if placement_match:
            sections["placement"] = placement_match.group(1).strip()

        # Pattern for Section 2
        hs_match = re.search(
            r"(?:SECTION 2|FINAL HS CODE)[^\n]*\n(.*?)(?=(?:SECTION 3|JUSTIFICATION))",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if hs_match:
            sections["hs_code"] = hs_match.group(1).strip()

        # Pattern for Section 3
        justification_match = re.search(
            r"(?:SECTION 3|JUSTIFICATION)[^\n]*\n(.*)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if justification_match:
            sections["justification"] = justification_match.group(1).strip()

    # Extract just the numeric code from the HS code section
    hs_code_number = ""
    if sections["hs_code"]:
        code_match = re.search(r"\b(\d{4}(?:\.\d{2}(?:\.\d{2,4})?)?)\b", sections["hs_code"])
        if code_match:
            hs_code_number = code_match.group(1)

    # If parsing failed, display raw response
    if not any(sections.values()):
        st.markdown(response_text)
        return

    # Render styled result
    st.markdown(
        """<div class="result-container">
        <div class="result-header">
            <h3>Classification Result</h3>
        </div>
        <div class="result-body">""",
        unsafe_allow_html=True,
    )

    # Quick Conclusion
    if sections["conclusion"]:
        st.markdown(
            f"""<div class="result-section">
                <div class="result-section-title">Quick Conclusion</div>
                <div class="result-section-content">{_md_to_html(sections["conclusion"])}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Tariff Placement
    if sections["placement"]:
        st.markdown(
            f"""<div class="result-section">
                <div class="result-section-title">Tariff Placement</div>
                <div class="result-section-content">{_md_to_html(sections["placement"])}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # HS Code
    if hs_code_number:
        st.markdown(
            f"""<div class="result-section">
                <div class="result-section-title">Final HS Code</div>
                <div class="hs-code-display">{hs_code_number}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    elif sections["hs_code"]:
        st.markdown(
            f"""<div class="result-section">
                <div class="result-section-title">Final HS Code</div>
                <div class="result-section-content">{_md_to_html(sections["hs_code"])}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    # Justification
    if sections["justification"]:
        st.markdown(
            f"""<div class="result-section">
                <div class="result-section-title">Justification</div>
                <div class="result-section-content">{_md_to_html(sections["justification"])}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)


def _md_to_html(text: str) -> str:
    """Minimal markdown-to-HTML for display in styled containers."""
    import re

    # Escape any raw HTML before applying lightweight markdown formatting.
    text = html.escape(text, quote=False)

    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    # Line breaks → paragraphs
    paragraphs = text.split("\n\n")
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if p:
            # Check if it's a list item
            if p.startswith("- ") or p.startswith("• "):
                items = p.split("\n")
                list_html = "<ul style='margin: 0.5rem 0; padding-left: 1.5rem;'>"
                for item in items:
                    item = item.lstrip("- •").strip()
                    if item:
                        list_html += f"<li style='margin-bottom: 0.3rem;'>{item}</li>"
                list_html += "</ul>"
                html_parts.append(list_html)
            else:
                p = p.replace("\n", "<br>")
                html_parts.append(f"<p style='margin: 0 0 0.75rem;'>{p}</p>")
    return "".join(html_parts)


# ─────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────


def main():
    # ── Sidebar API Credentials Override ──────────────────
    st.sidebar.markdown("### 🔑 API Credentials")
    custom_key = st.sidebar.text_input(
        "Gemini API Key Override",
        type="password",
        placeholder="Paste override key if quota is exhausted",
        help="If the default free-tier API key reaches its limit, paste a new key from Google AI Studio (aistudio.google.com) to resume classification immediately.",
    )
    if custom_key:
        if "custom_api_key" not in st.session_state or st.session_state["custom_api_key"] != custom_key:
            st.session_state["custom_api_key"] = custom_key
            init_genai_client.clear()
            st.rerun()
    elif "custom_api_key" in st.session_state:
        del st.session_state["custom_api_key"]
        init_genai_client.clear()
        st.rerun()

    # ── Header ────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-header">
            <h1>Tariff Navigator</h1>
            <div class="subtitle">
                HS 2022 Customs Classification — Powered by AI
            </div>
            <div class="badge">Ethiopian Customs Transit Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Initialize Services ───────────────────────────────
    client = init_genai_client()
    collection, doc_count = init_chromadb()

    # ── Status Bar ────────────────────────────────────────
    api_status = "green" if client else "red"
    api_label = "API Connected" if client else "API Key Missing"
    db_status = "green" if collection and doc_count > 0 else ("amber" if collection else "red")
    db_label = f"{doc_count:,} rules indexed" if doc_count > 0 else ("DB empty — ingestion in progress" if collection else "Database not found")

    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-item">
                <span class="status-dot {api_status}"></span> {api_label}
            </div>
            <div class="status-item">
                <span class="status-dot {db_status}"></span> {db_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Error States ──────────────────────────────────────
    if not client:
        st.sidebar.error("Please enter a valid API key.")
        st.error(
            "**Gemini API key not found.** Set the `GEMINI_API_KEY` environment variable or use the sidebar key override.",
            icon="🔑",
        )
        st.stop()

    if not collection:
        st.warning(
            "**Vector database not found.** Run the ingestion script first to index tariff rules.",
            icon="📦",
        )
        st.code(
            'python ingest.py "path/to/hs2022.pdf"',
            language="bash",
        )
        st.stop()

    # ── Input Section ─────────────────────────────────────
    st.markdown('<div class="section-label">Item Specification</div>', unsafe_allow_html=True)

    item_description = st.text_area(
        "Describe the item to classify",
        placeholder=(
            "Enter detailed specifications of the imported item.\n\n"
            "Example: Samsung Galaxy S24 Ultra smartphone, 256GB storage, "
            "5G capable, with Snapdragon 8 Gen 3 processor, 6.8-inch "
            "Dynamic AMOLED display, titanium frame, imported from South Korea "
            "for commercial resale."
        ),
        height=160,
        label_visibility="collapsed",
    )

    # Optional image upload
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            '<div class="section-label">Product Image (Optional)</div>',
            unsafe_allow_html=True,
        )
        uploaded_image = st.file_uploader(
            "Upload a product image for visual classification",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

    with col2:
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded product image", use_container_width=True)

    # ── Classify Button ───────────────────────────────────
    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
    classify_clicked = st.button(
        "Classify Item",
        disabled=not item_description.strip(),
        use_container_width=True,
    )

    # ── Classification Pipeline ───────────────────────────
    if classify_clicked and item_description.strip():
        # Prepare image bytes first so the cache key includes the actual image content.
        image_bytes = None
        image_mime = None
        image_hash = "no-image"
        if uploaded_image:
            uploaded_image.seek(0)
            image_bytes = uploaded_image.read()
            image_mime = (
                f"image/{uploaded_image.type.split('/')[-1]}"
                if "/" in uploaded_image.type
                else f"image/{uploaded_image.type}"
            )
            image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Create cache key from query + image content.
        cache_key = hashlib.sha256(
            f"{item_description}:{image_hash}".encode()
        ).hexdigest()[:16]

        # Initialize cache
        cache = QueryCache()
        cached_result = cache.get(item_description, image_bytes)

        if cached_result:
            result = cached_result
            st.caption("💾 *Retrieved from local query cache (0ms latency, $0 cost)*")
        elif f"result_{cache_key}" in st.session_state:
            result = st.session_state[f"result_{cache_key}"]
            st.caption("📋 *Session-cached result*")
        else:
            # Run classification
            with st.spinner("Analyzing item against HS 2022 tariff rules..."):
                try:
                    result = classify_item(
                        client=client,
                        collection=collection,
                        item_description=item_description,
                        image_bytes=image_bytes,
                        image_mime=image_mime,
                    )
                    # Cache in both stores
                    cache.set(item_description, result, image_bytes)
                    st.session_state[f"result_{cache_key}"] = result
                except Exception as e:
                    # Graceful local fallback display
                    st.error(
                        "⚠️ **Gemini API limit reached or service unavailable.**\n\n"
                        "Since the AI is currently offline, showing the **most relevant local tariff rules** "
                        "retrieved from ChromaDB. You can read these notes to verify the classification manually."
                    )
                    with st.spinner("Querying local database..."):
                        try:
                            # Note: query_chromadb requires client for embedding, but embedding API might also be blocked.
                            # We wrap this in try-except to catch embedding-specific quota limits.
                            chunks = query_chromadb(collection, client, item_description)
                            st.subheader("📚 Matching Tariff Context (ChromaDB)")
                            for i, chunk in enumerate(chunks, 1):
                                similarity = 1 - chunk["distance"]
                                st.markdown(f"**Rule {i}** — Page {chunk['page_number']} *(Relevance: {similarity:.1%})*")
                                st.info(chunk["text"])
                        except Exception as db_err:
                            st.warning(f"Could not retrieve semantic matches due to embedding key quota: {db_err}")
                            st.markdown("Please insert a new API key in the sidebar override to restore searching.")
                    st.stop()

        # ── Display Results ───────────────────────────────
        parse_and_display_response(result["response"])

        # ── Metadata ──────────────────────────────────────
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.caption(f"⏱ Processed in {result['elapsed']:.1f}s")
        with meta_col2:
            st.caption(f"📄 {len(result['chunks'])} context chunks retrieved")

        # ── Retrieved Context (expandable) ────────────────
        with st.expander("View retrieved tariff rules (context sent to AI)"):
            for i, chunk in enumerate(result["chunks"], 1):
                similarity = 1 - chunk["distance"]  # cosine distance → similarity
                st.markdown(
                    f"**Chunk {i}** — Page {chunk['page_number']} "
                    f"(relevance: {similarity:.1%})"
                )
                st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                st.divider()

    # ── Footer ────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-footer">
            Tariff Navigator &middot; HS 2022 Classification System
            &middot; For professional customs transit use only
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
