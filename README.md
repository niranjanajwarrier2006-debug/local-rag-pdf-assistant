# 🤖 AI PDF Assistant

An offline AI-powered PDF knowledge assistant built using Streamlit, Ollama, Phi-3, LangChain, and ChromaDB.

The system uses Retrieval-Augmented Generation (RAG) to semantically search uploaded documents and generate context-aware responses with reduced hallucinations — all running locally without cloud dependency.

---

## 🚀 Features

- Local LLM inference using Ollama + Phi3
- PDF upload and processing
- Retrieval-Augmented Generation (RAG)
- Semantic search using vector embeddings
- ChromaDB vector database
- ChatGPT-style interface
- Citation-aware responses
- Privacy-focused offline execution

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Ollama
- Phi3
- LangChain
- ChromaDB

---

## ▶️ Run Locally

### Start Ollama

```bash
ollama run phi3
```

### Run Streamlit

```bash
python -m streamlit run app.py
```

---

## 📌 Future Improvements

- Voice input
- Multiple PDF support
- Persistent chat history
- Better citations
- Multi-model support