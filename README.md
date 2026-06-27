🤖 AI PDF Assistant

An offline, privacy-focused AI-powered PDF knowledge assistant. This system uses Retrieval-Augmented Generation (RAG) to semantically search your PDFs and generate context-aware responses with minimal hallucinations—all running locally on your hardware.

🚀 Key Features

Context Isolation: Designed to handle one PDF at a time with a "fresh state," preventing context pollution from previous documents.

Local LLM Inference: Powered by Phi-3 via Ollama.

Persistent Vector Storage: Uses ChromaDB for local indexing, ensuring performance is fast and data stays on your machine.

Smart Citations: Automatically cites source content by page number to verify AI claims.

Privacy-First: 100% offline; no data ever leaves your machine.

Clean & Responsive UI: Built with Streamlit for a smooth, casual-but-professional experience.

🛠️ Tech Stack

Frontend: Streamlit

LLM: Phi-3 (via Ollama)

Embeddings: nomic-embed-text

Orchestration: LangChain

Vector Database: ChromaDB

PDF Processing: PyPDF

⚙️ Prerequisites

Install Ollama: Download and install it from ollama.com.

Pull Required Models:
Open your terminal and run:

Bash
ollama pull phi3
ollama pull nomic-embed-text

🚀 Setup & Run Locally

Clone the repository and enter the folder:

Bash
cd path/to/your/project
Install dependencies:

Bash
pip install -U langchain-ollama langchain-chroma pypdf streamlit ollama
Run the application:

Bash
streamlit run app.py

💡 How it works

Upload: Use the sidebar to upload your PDF.

Process: The app automatically indexes the PDF into a local chroma_db directory.

Chat: Ask questions. If you need to switch documents, the "Clear Everything" button in the sidebar wipes the index safely, ensuring your next chat is perfectly accurate to the new file.

📌 Future Improvements

Voice Input: Integration with Whisper for speech-to-text.

Enhanced Document Management: A dashboard to manage, delete, or switch between multiple saved PDFs.

Multi-Model Support: Add a dropdown to switch between LLMs (e.g., Llama 3) via the UI.
