import chromadb, json, os

c = chromadb.PersistentClient(path="./db")
col = c.get_collection("tariff_rules")
print(f"ChromaDB docs: {col.count()}")

if os.path.exists("./db/.ingest_progress"):
    p = json.loads(open("./db/.ingest_progress").read())
    print(f"Pages completed: {p['last_completed_page']}/{p['total_pages']}")
else:
    print("No progress file yet")

cached = len([f for f in os.listdir("./db/.text_cache") if f.endswith(".txt")])
print(f"Pages cached (OCR done): {cached}")

if col.count() > 0:
    sample = col.peek(limit=1)
    print(f"\nSample chunk (first 300 chars):\n{sample['documents'][0][:300]}")
