import streamlit as st
import ollama
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI PDF Assistant", page_icon="🤖", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main { padding-top: 1rem; }
.stChatMessage { border-radius: 15px; padding: 10px; }
.user-box { background-color: #DCF8C6; padding: 14px; border-radius: 12px; margin-bottom: 10px; color: black; }
.bot-box { background-color: #F1F3F4; padding: 14px; border-radius: 12px; margin-bottom: 10px; color: black; }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG & INITIALIZATION ----------------
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3"

# Initialize Embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Setup Persistent VectorDB
vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

st.title("🤖 AI PDF Assistant")
st.caption(f"Local RAG using {LLM_MODEL} + {EMBEDDING_MODEL}")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.divider()
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- PDF PROCESSING ----------------
if uploaded_file:
    # Save uploaded file temporarily to process
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    with st.spinner("📚 Processing PDF..."):
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            
            # Add to VectorDB
            vectorstore.add_documents(split_docs)
            st.success("✅ PDF processed and indexed!")
            os.remove(temp_path) # Clean up temp file
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    role_class = "user-box" if msg["role"] == "user" else "bot-box"
    st.markdown(f'<div class="{role_class}"><b>{"🧑 You" if msg["role"] == "user" else "🤖 Assistant"}</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Ask something about your PDF...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-box"><b>🧑 You</b><br><br>{prompt}</div>', unsafe_allow_html=True)

    # RAG SEARCH
    docs = vectorstore.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate Citations based on metadata page number (starts at 0, so add 1)
    citations = [f"Page {doc.metadata.get('page', 0) + 1}" for doc in docs]

    system_prompt = f"""
    You are an intelligent PDF assistant. Answer ONLY using the provided PDF context.
    If the answer is not in the context, say 'I could not find that information in the uploaded PDF.'
    PDF Context: {context}
    """

    with st.spinner("🤖 Thinking..."):
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                options={"temperature": temperature}
            )
            reply = response["message"]["content"]
            
            # Add Citations
            reply += "\n\n---\n📚 **Sources:** " + ", ".join(set(citations))

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.markdown(f'<div class="bot-box"><b>🤖 Assistant</b><br><br>{reply}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating response: {e}")