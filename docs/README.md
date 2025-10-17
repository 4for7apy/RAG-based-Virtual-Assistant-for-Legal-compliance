## Docs Overview

- **What it is**: A Streamlit-based RAG assistant using LangChain, OpenAI, and Qdrant.
- **Key features**:
  - Web-scrape content and load PDFs
  - Embed to Qdrant vector DB
  - Chat UI via Streamlit; optional voice I/O

### Screenshots

![Web Login](./assets/web login3.jpg)
![Open Eyes](./assets/open_eyes.jpg)

### Quickstart

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Create `.env` with `OPENAI_API_KEY` and Qdrant settings if needed.
4. `streamlit run app/streamlit_ui.py`
