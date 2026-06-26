#!/usr/bin/env python3
"""
ingest.py — Memory-Safe PDF Ingestion for Tariff Navigator

Parses a massive HS 2022 customs classification PDF (scanned/image-based)
and populates a local ChromaDB vector store with embedded text chunks.

Designed for 900 MB+ scanned PDFs:
  - Detects image-only pages and uses Gemini Vision for OCR
  - Lazy batch loading via PyMuPDF memory-mapped I/O
  - Local text cache to avoid re-OCR on resume
  - Forced garbage collection between batches
  - Resumable progress tracking
  - Exponential-backoff retries for Gemini API calls

Usage:
    python ingest.py <path_to_pdf>
    python ingest.py <path_to_pdf> --batch-size 30
    python ingest.py <path_to_pdf> --reset
    python ingest.py <path_to_pdf> --dry-run
"""

import argparse
import gc
import io
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
from google import genai
from google.genai import types as genai_types
from tqdm import tqdm

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb is not installed. Run: pip install chromadb")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DB_DIR = Path("./db")
TEXT_CACHE_DIR = DB_DIR / ".text_cache"
PROGRESS_FILE = DB_DIR / ".ingest_progress"
COLLECTION_NAME = "tariff_rules"

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
OCR_MODEL = "gemini-2.5-flash-lite"

# Batching
EMBED_BATCH_SIZE = 100  # Max texts per embed_content call
MAX_RETRIES = 5
BASE_RETRY_DELAY = 2  # seconds

# OCR rendering
OCR_DPI = 200  # Resolution for rendering PDF pages to images
OCR_RATE_LIMIT_DELAY = 2  # seconds between OCR calls (free tier: ~1500 RPD)

# OCR prompt — instructs Gemini to faithfully transcribe the page
OCR_SYSTEM_PROMPT = """You are a precision document OCR system for customs classification rulebooks.
Extract ALL text from this scanned page image EXACTLY as it appears. Follow these rules strictly:

1. Preserve all HS codes, heading numbers, subheading numbers, and article numbers exactly.
2. Preserve table structures — use pipes (|) to separate columns and newlines for rows.
3. Preserve section titles, chapter titles, and all hierarchical numbering.
4. Preserve all footnotes, notes, and legal references verbatim.
5. If the page contains a table of contents, list of sections, or index, transcribe it fully.
6. Do NOT summarize, interpret, paraphrase, or omit any content.
7. Do NOT add commentary, headers, or formatting not present in the original.
8. If a page is blank or contains only decorative elements, respond with: [BLANK PAGE]
9. For any text you cannot read clearly, use [?] to mark uncertain characters.

Output ONLY the extracted text, nothing else."""

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ─────────────────────────────────────────────────────────────
# Text Processing
# ─────────────────────────────────────────────────────────────


def clean_text(raw: str) -> str:
    """Normalize unicode, collapse whitespace, strip control characters."""
    text = unicodedata.normalize("NFC", raw)
    # Remove control characters except newlines and tabs
    text = re.sub(r"[^\S\n\t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(
    text: str,
    page_num: int,
    source_file: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.

    Each chunk is a dict with keys:
        - id: unique identifier (source_page_chunk)
        - text: the chunk content
        - metadata: {page_number, source_file, chunk_index}
    """
    if not text or len(text.strip()) < 10:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            search_start = start + int(chunk_size * 0.8)
            para_break = text.rfind("\n\n", search_start, end)
            if para_break != -1:
                end = para_break + 2
            else:
                sentence_break = text.rfind(". ", search_start, end)
                if sentence_break != -1:
                    end = sentence_break + 2

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= 20:
            chunk_id = f"p{page_num:05d}_c{chunk_index:03d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "page_number": page_num,
                        "source_file": source_file,
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


# ─────────────────────────────────────────────────────────────
# Text Cache (avoid re-OCR on resume)
# ─────────────────────────────────────────────────────────────


def get_cached_text(page_num: int) -> str | None:
    """Retrieve cached OCR text for a page, or None if not cached."""
    cache_file = TEXT_CACHE_DIR / f"page_{page_num:05d}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    return None


def save_cached_text(page_num: int, text: str) -> None:
    """Cache OCR text for a page to avoid re-processing."""
    TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = TEXT_CACHE_DIR / f"page_{page_num:05d}.txt"
    cache_file.write_text(text, encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Gemini Client
# ─────────────────────────────────────────────────────────────


def create_genai_client() -> genai.Client:
    """Initialize the Google GenAI client with API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error(
            "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
        sys.exit(1)
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────
# Gemini Vision OCR
# ─────────────────────────────────────────────────────────────


def render_page_to_png(page: fitz.Page, dpi: int = OCR_DPI) -> bytes:
    """Render a PDF page to PNG bytes at the specified DPI."""
    zoom = dpi / 72.0  # 72 DPI is the PDF default
    mat = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pixmap.tobytes("png")
    # Explicitly free the pixmap to reclaim memory
    pixmap = None
    return png_bytes


def ocr_page_with_gemini(
    client: genai.Client, png_bytes: bytes, page_num: int
) -> str:
    """
    Send a page image to Gemini Vision for OCR text extraction.
    Returns the extracted text, with retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=OCR_MODEL,
                contents=[
                    genai_types.Content(
                        parts=[
                            genai_types.Part.from_text(text=OCR_SYSTEM_PROMPT),
                            genai_types.Part.from_bytes(
                                data=png_bytes,
                                mime_type="image/png",
                            ),
                        ],
                    ),
                ],
            )

            extracted = response.text.strip() if response.text else ""

            if extracted == "[BLANK PAGE]":
                return ""

            return extracted

        except Exception as e:
            delay = BASE_RETRY_DELAY * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                log.warning(
                    "OCR error page %d (attempt %d/%d): %s — retrying in %ds",
                    page_num,
                    attempt + 1,
                    MAX_RETRIES,
                    str(e)[:120],
                    delay,
                )
                time.sleep(delay)
            else:
                log.error(
                    "OCR FAILED page %d after %d attempts: %s",
                    page_num,
                    MAX_RETRIES,
                    str(e)[:200],
                )
                return ""  # Skip this page rather than crash


def extract_page_text(
    client: genai.Client,
    doc: fitz.Document,
    page_num: int,
    dry_run: bool = False,
) -> str:
    """
    Extract text from a PDF page. Tries native text first;
    falls back to Gemini Vision OCR for scanned/image pages.
    Results are cached locally to avoid redundant API calls.
    """
    # Check cache first
    cached = get_cached_text(page_num)
    if cached is not None:
        return cached

    page = doc.load_page(page_num)

    # Try native text extraction first
    native_text = page.get_text("text").strip()
    if len(native_text) > 30:
        cleaned = clean_text(native_text)
        if not dry_run:
            save_cached_text(page_num, cleaned)
        return cleaned

    # Fall back to Gemini Vision OCR
    if dry_run:
        return "[DRY RUN — OCR skipped]"

    png_bytes = render_page_to_png(page)
    extracted = ocr_page_with_gemini(client, png_bytes, page_num)
    cleaned = clean_text(extracted)

    # Cache the result
    save_cached_text(page_num, cleaned)

    # Rate limit to avoid hitting API quota
    time.sleep(OCR_RATE_LIMIT_DELAY)

    return cleaned


# ─────────────────────────────────────────────────────────────
# Gemini Embedding (with retry)
# ─────────────────────────────────────────────────────────────


def embed_texts_with_retry(
    client: genai.Client, texts: list[str]
) -> list[list[float]]:
    """
    Embed a list of texts using Gemini, with exponential backoff retry.
    Handles batching internally (max EMBED_BATCH_SIZE per API call).
    """
    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                )
                batch_embeddings = [e.values for e in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                break

            except Exception as e:
                delay = BASE_RETRY_DELAY * (2**attempt)
                if attempt < MAX_RETRIES - 1:
                    log.warning(
                        "Embed API error (attempt %d/%d): %s — retrying in %ds",
                        attempt + 1,
                        MAX_RETRIES,
                        str(e)[:120],
                        delay,
                    )
                    time.sleep(delay)
                else:
                    log.error(
                        "Embed API failed after %d attempts: %s",
                        MAX_RETRIES,
                        str(e)[:200],
                    )
                    raise

    return all_embeddings


# ─────────────────────────────────────────────────────────────
# ChromaDB Operations
# ─────────────────────────────────────────────────────────────


def get_or_create_collection(reset: bool = False):
    """
    Get or create the ChromaDB collection with persistent storage.
    If reset=True, deletes the existing collection first.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    if reset:
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            log.info("Deleted existing collection '%s'", COLLECTION_NAME)
        except Exception:
            pass

        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            log.info("Cleared ingestion progress marker")

        # Clear text cache too
        if TEXT_CACHE_DIR.exists():
            import shutil
            shutil.rmtree(TEXT_CACHE_DIR)
            log.info("Cleared text cache")

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def upsert_chunks(
    collection, chunks: list[dict], embeddings: list[list[float]]
) -> None:
    """Upsert a batch of chunks with their embeddings into ChromaDB."""
    if not chunks:
        return

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )


# ─────────────────────────────────────────────────────────────
# Progress Tracking (Resumability)
# ─────────────────────────────────────────────────────────────


def load_progress() -> int:
    """Load the last successfully processed page number. Returns 0 if none."""
    if PROGRESS_FILE.exists():
        try:
            data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
            return data.get("last_completed_page", 0)
        except (json.JSONDecodeError, KeyError):
            return 0
    return 0


def save_progress(last_completed_page: int, total_pages: int) -> None:
    """Persist the last successfully completed page number."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "last_completed_page": last_completed_page,
                "total_pages": total_pages,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────
# Main Ingestion Pipeline
# ─────────────────────────────────────────────────────────────


def ingest(
    pdf_path: str,
    batch_size: int = 50,
    chunk_size: int = 1000,
    overlap: int = 200,
    reset: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Main ingestion pipeline for scanned/image-based PDFs.

    1. Opens the PDF with memory-mapped I/O (fitz.open)
    2. Iterates pages in batches of `batch_size`
    3. For each page: tries native text → falls back to Gemini Vision OCR
    4. Caches extracted text locally (avoids re-OCR on resume)
    5. Chunks text with overlap
    6. Embeds via Gemini embedding API (batched, with retry)
    7. Upserts into ChromaDB
    8. Forces gc.collect() after every batch
    9. Tracks progress for resumability
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        log.error("PDF file not found: %s", pdf_path)
        sys.exit(1)

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    log.info("Opening PDF: %s (%.1f MB)", pdf_path.name, file_size_mb)

    # ── Open PDF (memory-mapped) ──────────────────────────────
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    log.info("Total pages: %d", total_pages)
    if total_pages == 0:
        log.error("PDF contains no pages: %s", pdf_path)
        sys.exit(1)

    # ── Quick scan: detect if PDF is scanned ──────────────────
    sample_pages = sorted(
        {
            page_num
            for page_num in (0, 1, min(10, total_pages - 1))
            if 0 <= page_num < total_pages
        }
    )
    native_text_found = False
    for sp in sample_pages:
        if len(doc.load_page(sp).get_text("text").strip()) > 30:
            native_text_found = True
            break

    if native_text_found:
        log.info("PDF has native text — using direct extraction")
    else:
        log.info("PDF is SCANNED (image-only) — using Gemini Vision OCR")
        log.info(
            "This will take longer due to per-page OCR. "
            "Extracted text is cached locally for resumability."
        )

    # ── Initialize services ───────────────────────────────────
    if not dry_run:
        client = create_genai_client()
        collection = get_or_create_collection(reset=reset)
        existing_count = collection.count()
        log.info(
            "ChromaDB collection '%s' has %d existing documents",
            COLLECTION_NAME,
            existing_count,
        )
    else:
        client = None
        collection = None
        log.info("DRY RUN — no API calls or DB writes will occur")

    # ── Determine start page ──────────────────────────────────
    start_page = 0 if reset else load_progress()
    if start_page > 0:
        log.info("Resuming from page %d (previously completed)", start_page)
    if start_page >= total_pages:
        log.info("Ingestion already complete for this PDF. Nothing to do.")
        doc.close()
        return

    # ── Statistics ────────────────────────────────────────────
    total_chunks_created = 0
    total_chunks_embedded = 0
    pages_ocr = 0
    pages_native = 0
    pages_cached = 0
    empty_pages = 0
    failed_batches = 0
    start_time = time.time()

    # ── Main loop: iterate page batches ───────────────────────
    num_batches = (total_pages - start_page + batch_size - 1) // batch_size
    pbar = tqdm(
        range(start_page, total_pages, batch_size),
        desc="Ingesting",
        unit="batch",
        total=num_batches,
        ncols=100,
    )

    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, total_pages)
        batch_chunks: list[dict] = []

        # ── Extract text from this batch of pages ─────────────
        for page_num in range(batch_start, batch_end):
            try:
                # Check if text is already cached
                was_cached = get_cached_text(page_num) is not None

                text = extract_page_text(
                    client=client,
                    doc=doc,
                    page_num=page_num,
                    dry_run=dry_run,
                )

                if was_cached:
                    pages_cached += 1
                elif text and not dry_run:
                    # Determine if we used OCR or native
                    native = doc.load_page(page_num).get_text("text").strip()
                    if len(native) > 30:
                        pages_native += 1
                    else:
                        pages_ocr += 1

            except Exception as e:
                log.warning("Failed to process page %d: %s", page_num, e)
                continue

            if not text or len(text.strip()) < 10:
                empty_pages += 1
                continue

            page_chunks = split_into_chunks(
                text=text,
                page_num=page_num,
                source_file=pdf_path.name,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            batch_chunks.extend(page_chunks)

        total_chunks_created += len(batch_chunks)

        elapsed = time.time() - start_time
        rate = (batch_end - start_page) / (elapsed / 60) if elapsed > 0 else 0
        pbar.set_postfix(
            chunks=total_chunks_created,
            pg=f"{batch_end}/{total_pages}",
            rate=f"{rate:.0f}p/m",
        )

        # ── Embed & upsert ────────────────────────────────────
        if batch_chunks and not dry_run:
            try:
                texts = [c["text"] for c in batch_chunks]
                embeddings = embed_texts_with_retry(client, texts)
                upsert_chunks(collection, batch_chunks, embeddings)
                total_chunks_embedded += len(batch_chunks)
            except Exception as e:
                failed_batches += 1
                log.error(
                    "BATCH FAILED (pages %d–%d): %s — skipping this batch",
                    batch_start,
                    batch_end - 1,
                    str(e)[:200],
                )
                continue

        # ── Persist progress & free memory ────────────────────
        if not dry_run:
            save_progress(batch_end, total_pages)

        gc.collect()

    # ── Cleanup ───────────────────────────────────────────────
    doc.close()
    elapsed = time.time() - start_time

    # ── Final report ──────────────────────────────────────────
    log.info("═" * 60)
    log.info("  INGESTION COMPLETE")
    log.info("═" * 60)
    log.info("  Total pages processed : %d", total_pages - start_page)
    log.info("  Pages via OCR         : %d", pages_ocr)
    log.info("  Pages via native text : %d", pages_native)
    log.info("  Pages from cache      : %d", pages_cached)
    log.info("  Empty/skipped pages   : %d", empty_pages)
    log.info("  Chunks created        : %d", total_chunks_created)
    log.info("  Chunks embedded       : %d", total_chunks_embedded)
    log.info("  Failed batches        : %d", failed_batches)
    log.info("  Elapsed time          : %.1f minutes", elapsed / 60)
    if not dry_run and collection:
        log.info("  ChromaDB total docs   : %d", collection.count())
    log.info("  Database location     : %s", DB_DIR.resolve())
    log.info("  Text cache location   : %s", TEXT_CACHE_DIR.resolve())
    log.info("═" * 60)


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a customs classification PDF into ChromaDB for the "
            "Tariff Navigator RAG system. Supports scanned/image-based PDFs "
            "via Gemini Vision OCR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py hs2022.pdf
  python ingest.py hs2022.pdf --batch-size 30
  python ingest.py hs2022.pdf --reset
  python ingest.py hs2022.pdf --dry-run
        """,
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the HS 2022 customs classification PDF",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of PDF pages to process per batch (default: 50)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target character count per text chunk (default: 1000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Character overlap between adjacent chunks (default: 200)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing database and re-ingest from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline without calling the API or writing to the DB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(
        pdf_path=args.pdf_path,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset=args.reset,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
