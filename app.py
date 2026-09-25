import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from rag_engine import CourseMateRAGEngine

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="CourseMate AI",
    page_icon="🎓",
    layout="wide"
)

# Initialize Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = CourseMateRAGEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar UI
with st.sidebar:
    st.title("⚙️ Configuration")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        st.success("⚡ Gemini API Key Active")
    else:
        user_key = st.text_input("Enter Google Gemini API Key:", type="password")
        if user_key:
            os.environ["GOOGLE_API_KEY"] = user_key
            st.success("API Key set successfully!")

    st.divider()
    st.title("📚 Upload Course Documents")
    uploaded_files = st.file_uploader(
        "Select PDF materials:",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
        elif not os.getenv("GOOGLE_API_KEY"):
            st.error("Please configure your Gemini API Key first.")
        else:
            with st.spinner("Processing documents into vector store..."):
                all_docs = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    os.remove(tmp_path)

                st.session_state.rag_engine.ingest_documents(all_docs)
                st.success(f"Successfully processed {len(uploaded_files)} PDF(s) into vector store!")

    st.divider()
    st.title("🛠️ Quick Actions")
    if st.button("📄 Generate Study Guide"):
        if st.session_state.rag_engine.retriever:
            with st.spinner("Generating Study Guide..."):
                guide = st.session_state.rag_engine.generate_study_guide()
                st.session_state.messages.append({"role": "assistant", "content": guide})
        else:
            st.warning("Please process documents first.")

    if st.button("📖 Extract Glossary"):
        if st.session_state.rag_engine.retriever:
            with st.spinner("Extracting Key Terms..."):
                glossary = st.session_state.rag_engine.extract_key_terms()
                st.session_state.messages.append({"role": "assistant", "content": glossary})
        else:
            st.warning("Please process documents first.")

# Main Interface
st.title("🎓 CourseMate AI")
st.caption("Your RAG Study Assistant Powered by Google Gemini")

tab1, tab2 = st.tabs(["💬 Chat & Query", "ℹ️ About CourseMate AI"])

with tab1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Chat Input
    if user_query := st.chat_input("Ask a question about your course materials..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer = st.session_state.rag_engine.query(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.markdown("""
    ### Welcome to CourseMate AI!
    CourseMate AI helps you chat with your academic materials, extract key glossaries, and automatically build comprehensive study guides.
    
    **How to use:**
    1. Ensure your Gemini API Key is configured in `.env` or the sidebar.
    2. Upload your PDF course materials using the sidebar.
    3. Click **Process Documents**.
    4. Start asking questions or use the **Quick Actions** buttons!
    """)