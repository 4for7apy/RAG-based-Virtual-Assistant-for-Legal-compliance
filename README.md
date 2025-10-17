# RAG-Based Virtual Assistant

A modern, minimal RAG assistant using LangChain + OpenAI + Qdrant with a Streamlit UI.


## ✨ Features
- Web scrape and PDF ingestion
- Qdrant vector search store
- Chat via OpenAI + LangChain
- Streamlit UI with optional voice (STT/TTS)

## 🗂️ Repository Structure
```
.
├─ app/                  # Source (UI, scrapers, voice)
│  ├─ streamlit_ui.py
│  ├─ voice_chat2.py
│  ├─ Data_Scrapping.py
│  └─ ...
├─ data/                 # Example datasets
│  └─ mcq.pdf
├─ docs/
│  ├─ assets/            # Images & screenshots
│  └─ README.md          # Extra docs & quickstart
├─ requirements.txt
├─ Procfile
├─ LICENSE
└─ CONTRIBUTING.md
```

## 🚀 Quickstart
1. Create a virtualenv
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. Install deps: `pip install -r requirements.txt`
3. Add `.env` with at least:
   - `OPENAI_API_KEY=...`
4. Run the app
   - UI: `streamlit run app/streamlit_ui.py`
   - Voice demo: `streamlit run app/voice_chat2.py`


## 📦 Deployment
- Procfile provided (`web: python app/deploy.py`).
- Ensure `app/deploy.py` starts a server (Flask/FastAPI) for your platform.

## 📜 License
MIT — see `LICENSE`.
