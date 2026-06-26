from pathlib import Path
import json

import chromadb


DB_DIR = Path("./db")
TEXT_CACHE_DIR = DB_DIR / ".text_cache"
PROGRESS_FILE = DB_DIR / ".ingest_progress"


def main() -> None:
    if not DB_DIR.exists():
        print("Database directory not found")
        return

    try:
        c = chromadb.PersistentClient(path=str(DB_DIR))
        col = c.get_collection("tariff_rules")
        print(f"ChromaDB docs: {col.count()}")
    except Exception as exc:
        print(f"Could not open ChromaDB: {exc}")
        return

    if PROGRESS_FILE.exists():
        p = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        print(f"Pages completed: {p.get('last_completed_page', 0)}/{p.get('total_pages', 0)}")
    else:
        print("No progress file yet")

    if TEXT_CACHE_DIR.exists():
        cached = len([f for f in TEXT_CACHE_DIR.iterdir() if f.is_file() and f.suffix == ".txt"])
    else:
        cached = 0
    print(f"Pages cached (OCR done): {cached}")

    if col.count() > 0:
        sample = col.peek(limit=1)
        print(f"\nSample chunk (first 300 chars):\n{sample['documents'][0][:300]}")


if __name__ == "__main__":
    main()
