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
.user-box { background-color: #DCF8C6; padding: 14px; border-radius: 12px; margin-bottom: 10px; color: black; }
.bot-box { background-color: #F1F3F4; padding: 14px; border-radius: 12px; margin-bottom: 10px; color: black; }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "pdf_assistant_collection" # We use a specific name to target
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "phi3"

st.title("🤖 AI PDF Assistant")
st.caption(f"Local RAG using {LLM_MODEL}")

# ---------------- INITIALIZATION ----------------
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.divider()
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)
    
    if st.button("🗑️ Clear Everything"):
        st.session_state.messages = []
        # Clear collection without deleting folder
        try:
            temp_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, collection_name=COLLECTION_NAME)
            temp_db.delete_collection()
        except:
            pass
        st.rerun()

# ---------------- PDF PROCESSING ----------------
# We handle the database locally within the logic flow to avoid lock issues
if uploaded_file:
    # 1. Clean the collection (Wipes the data, keeps the folder)
    try:
        temp_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, collection_name=COLLECTION_NAME)
        temp_db.delete_collection()
    except:
        pass

    # 2. Re-initialize fresh
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings, 
        collection_name=COLLECTION_NAME
    )
    
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    with st.spinner("📚 Processing new PDF..."):
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            vectorstore.add_documents(split_docs)
            st.success("✅ PDF processed! Context is fresh.")
            os.remove(temp_path) 
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
else:
    # Load existing (or create if empty)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings, 
        collection_name=COLLECTION_NAME
    )

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    role_class = "user-box" if msg["role"] == "user" else "bot-box"
    st.markdown(f'<div class="{role_class}"><b>{"🧑 You" if msg["role"] == "user" else "🤖 Assistant"}</b><br><br>{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Ask something about your PDF...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-box"><b>🧑 You</b><br><br>{prompt}</div>', unsafe_allow_html=True)

    docs = vectorstore.similarity_search(prompt, k=3)
    
    if not docs:
        reply = "Please upload a PDF first."
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
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
                reply += "\n\n---\n📚 **Sources:** " + ", ".join(set(citations))
            except Exception as e:
                reply = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.markdown(f'<div class="bot-box"><b>🤖 Assistant</b><br><br>{reply}</div>', unsafe_allow_html=True)