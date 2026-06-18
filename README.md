# 🤖 AI PDF Assistant

An offline, privacy-focused AI-powered PDF knowledge assistant. This system uses Retrieval-Augmented Generation (RAG) to semantically search uploaded documents and generate context-aware responses with reduced hallucinations—all running locally without cloud dependency.

## 🚀 Features

* **Local LLM Inference:** Powered by `Phi-3` via Ollama.
* **Persistent Storage:** Uses `ChromaDB` to save vectors locally, allowing your data to persist across application restarts.
* **Page-Level Citations:** Automatically cites source content by page number.
* **Smart Embeddings:** Dedicated `nomic-embed-text` model for high-quality semantic vectorization.
* **Privacy-Focused:** 100% offline execution; no data leaves your machine.
* **Clean UI:** Streamlit-based chat interface.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Phi-3 (via Ollama)
* **Embeddings:** nomic-embed-text (via Ollama)
* **Orchestration:** LangChain, LangChain-Ollama, LangChain-Chroma
* **Vector Database:** ChromaDB
* **PDF Processing:** PyPDF

## ⚙️ Prerequisites

1.  **Install Ollama:** Download and install it from [ollama.com](https://ollama.com/).
2.  **Pull Required Models:**
    Open your terminal and run:
    ```bash
    ollama pull phi3
    ollama pull nomic-embed-text
    ```

## 🚀 Setup & Run Locally

1.  **Clone the repository/Navigate to project folder:**
    ```bash
    cd path/to/your/project
    ```

2.  **Install dependencies:**
    ```bash
    python -m pip install -U langchain-ollama langchain-chroma pypdf streamlit
    ```

3.  **Run the application:**
    ```bash
    python -m streamlit run app.py
    ```

## 📌 Future Improvements

* **Voice Input:** Integration with Whisper for speech-to-text.
* **Document Management:** A dashboard to manage, delete, or switch between uploaded PDFs.
* **Chat History Persistence:** Saving chat sessions to a database.
* **Multi-Model Support:** Ability to switch between different LLMs (e.g., Llama 3) via the UI.
