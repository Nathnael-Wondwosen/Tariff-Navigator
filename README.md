# Tariff Navigator

Local RAG app for tariff classification using ChromaDB, Gemini embeddings, and Streamlit.

## What’s in the repo

- `app.py` runs the Streamlit UI.
- `ingest.py` builds the local vector database from the HS 2022 PDF.
- `check_progress.py` prints the current ingestion state.
- `db/` stores the persistent ChromaDB index used by the app.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Local config

Copy `.env.example` to `.env` and fill in the values you want to change.
The app and ingestion script both read from that file automatically.

```bash
copy .env.example .env
```

At minimum, set:

- `GEMINI_API_KEY`

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app from this repo.
3. Set the main file path to `app.py`.
4. Add a secret named `GEMINI_API_KEY` in the app settings.
5. Deploy.

Notes:

- The app reads the Gemini key from Streamlit secrets, environment variables, or `.env` locally.
- The bundled `db/` directory is used as the persistent local vector store.
- If you update the tariff corpus, re-run `ingest.py` locally and commit the updated `db/` contents if you want the cloud app to use the new index.
- Keep `.env` out of Git. Use Streamlit secrets for the cloud deployment.

## Rebuild the index

```bash
python ingest.py path/to/hs2022.pdf
```

Optional flags:

- `--batch-size` to adjust ingestion batch size.
- `--chunk-size` to change chunk length.
- `--overlap` to control chunk overlap.
- `--reset` to rebuild from scratch.
- `--dry-run` to test the pipeline without API calls or writes.
