## Chat Bot (LangChain + Google Gemini) — Retrieval-Augmented Chat

A simple, Node.js-like Python project that runs a conversational chatbot with chat history and retrieval over local knowledge using LangChain, Google Gemini, and Chroma.

### Features
- **Conversational memory** per session
- **RAG**: answers grounded on `knowledge.txt`
- **Google Gemini** as the LLM
- **Local vector store** with Chroma
- Clean, modern LangChain setup (no deprecated `ConversationChain`)

## Tech Stack
- **Python** 3.10+
- **LangChain** core + community integrations
- **Google Gemini** via `langchain-google-genai`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **Vector DB**: Chroma (local)
- **Env**: `python-dotenv`

## Project Structure
```text
chat-bot/
  ├─ .env                   # holds GOOGLE_API_KEY
  ├─ .gitignore
  ├─ knowledge.txt          # documents used for retrieval
  ├─ main.py                # entrypoint
  ├─ requirements.txt       # dependencies
  └─ env/                   # virtual environment (created locally)
```

## Prerequisites
- Python 3.10+ installed
- A Google API key for Gemini (get one from Google AI Studio)

## Setup

### 1) Create and activate a virtual environment
```bash
cd /Users/mojtaba/Desktop/project/chat-bot
python -m venv env
source env/bin/activate
```

### 2) Add your API key
Create `.env`:
```bash
echo 'GOOGLE_API_KEY=your_google_api_key_here' > .env
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```
- Type your question at the prompt.
- Type `exit` to quit.

## How It Works
1. Loads docs from `knowledge.txt`, splits into chunks.
2. Builds embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
3. Stores vectors in a local Chroma DB.
4. Uses a history-aware retriever to rewrite your question based on chat history.
5. Sends context + question to Gemini; returns concise answers.
6. Maintains per-session memory using LangChain’s `RunnableWithMessageHistory`.

## Customize

### Update Knowledge
- Edit `knowledge.txt` with your content.
- Restart the app to re-ingest.

### Use Google Embeddings (optional, fewer deps)
Replace HuggingFace embeddings with Google’s to avoid `sentence-transformers`:
```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
```
And you can remove `langchain-huggingface` and `sentence-transformers` from `requirements.txt`.

## Troubleshooting

- Missing `sentence_transformers`:
  - Ensure `sentence-transformers` is in `requirements.txt`
  - Reinstall: `pip install -r requirements.txt`

- Chroma/SQLite issues:
  - Ensure `chromadb` is installed.
  - Delete local Chroma folder (if created) and re-run.

- API key errors:
  - Check `.env` has `GOOGLE_API_KEY` and you ran `source env/bin/activate`.

- Version conflicts:
  - Upgrade pip tooling: `python -m pip install --upgrade pip setuptools wheel`
  - Reinstall deps: `pip install -r requirements.txt --upgrade`

## Commands Cheat Sheet
```bash
# Activate / Deactivate
source env/bin/activate
deactivate

# Install / Freeze exact versions (like package-lock.json)
pip install -r requirements.txt
pip freeze > requirements.txt

# Upgrade a single package
pip install -U langchain

# Run
python main.py
```

## License
MIT (or your preferred license)


