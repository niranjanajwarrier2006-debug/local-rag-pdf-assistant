import streamlit as st
import ollama
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

.user-box {
    background-color: #DCF8C6;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: black;
}

.bot-box {
    background-color: #F1F3F4;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: black;
}

.small-text {
    font-size: 12px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🤖 AI PDF Assistant")
st.caption("Local RAG Chatbot using Ollama + Phi3")

# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    st.divider()

    temperature = st.slider(
        "Creativity",
        0.0,
        1.0,
        0.2,
        0.1
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- PDF PROCESSING ----------------

if uploaded_file and not st.session_state.pdf_processed:

    with st.spinner("📚 Processing PDF..."):

        try:

            pdf_reader = PdfReader(uploaded_file)

            text = ""

            for page in pdf_reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted

            if not text.strip():

                st.error("❌ Could not extract text from PDF")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )

            chunks = splitter.split_text(text)

            docs = [
                Document(page_content=chunk)
                for chunk in chunks
            ]

            embeddings = OllamaEmbeddings(
                model="phi3"
            )

            vectorstore = Chroma.from_documents(
                docs,
                embeddings
            )

            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_processed = True

            st.success("✅ PDF processed successfully!")

        except Exception as e:

            st.error(f"Error processing PDF: {e}")

# ---------------- DISPLAY CHAT ----------------

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f"""
            <div class="user-box">
            <b>🧑 You</b><br><br>
            {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-box">
            <b>🤖 Assistant</b><br><br>
            {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- CHAT INPUT ----------------

prompt = st.chat_input("Ask something about your PDF...")

if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Show user message
    st.markdown(
        f"""
        <div class="user-box">
        <b>🧑 You</b><br><br>
        {prompt}
        </div>
        """,
        unsafe_allow_html=True
    )

    context = ""
    citations = []

    # ---------------- RAG SEARCH ----------------

    if st.session_state.vectorstore:

        docs = st.session_state.vectorstore.similarity_search(
            prompt,
            k=3
        )

        context = "\n\n".join([
            doc.page_content for doc in docs
        ])

        citations = [
            f"Chunk {i+1}"
            for i in range(len(docs))
        ]

    # ---------------- SYSTEM PROMPT ----------------

    system_prompt = f"""
You are an intelligent PDF assistant.

Answer ONLY using the provided PDF context.

If answer is not found in the context, say:
'I could not find that information in the uploaded PDF.'

Keep answers clear and professional.

PDF Context:
{context}
"""

    # ---------------- GENERATE RESPONSE ----------------

    with st.spinner("🤖 Thinking..."):

        try:

            response = ollama.chat(
                model="phi3",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": temperature
                }
            )

            reply = response["message"]["content"]

            # ---------------- ADD CITATIONS ----------------

            if citations:

                reply += "\n\n---\n📚 Sources:\n"

                for cite in citations:
                    reply += f"- {cite}\n"

            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })

            # Show assistant message
            st.markdown(
                f"""
                <div class="bot-box">
                <b>🤖 Assistant</b><br><br>
                {reply}
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(f"Error generating response: {e}")

# ---------------- FOOTER ----------------

st.divider()

st.caption("🚀 Built with Streamlit + Ollama + Phi3 + LangChain + ChromaDB")