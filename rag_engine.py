import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class CourseMateRAGEngine:
    def __init__(self):
        # Local fast embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Updated to gemini-3.6-flash to resolve 404 deprecation error
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0.3
        )
        
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.processed_docs = []

    def ingest_documents(self, docs):
        """
        Splits PDF documents into chunks, generates embeddings,
        stores them in ChromaDB, and builds the retrieval chain.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        self.processed_docs = chunks

        # Store embeddings in Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Set up similarity retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # RAG Prompt Template
        template = """You are CourseMate AI, an intelligent study assistant for course materials.
Answer the question clearly using only the provided context.
If the answer is not present in the context, say "I don't have enough information in the provided materials to answer this."

Context:
{context}

Question:
{question}

Answer:"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Build execution chain
        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """
        Queries the RAG chain with a user question.
        """
        if not self.chain:
            return "Please upload and process your course documents first."
        
        return self.chain.invoke(question)

    def generate_study_guide(self) -> str:
        """
        Generates a structured study guide based on ingested context.
        """
        if not self.retriever:
            return "Please upload and process documents first."

        sample_docs = self.retriever.invoke("overview main topics summary key points")
        context_text = "\n\n".join(doc.page_content for doc in sample_docs)

        prompt = f"""Based on the following course context, generate a comprehensive Study Guide.
Include:
1. Executive Summary
2. Key Topics & Core Concepts
3. Practice Review Questions with Answers

Context:
{context_text}
"""
        return self.llm.invoke(prompt).content

    def extract_key_terms(self) -> str:
        """
        Extracts key terms and definitions from course context.
        """
        if not self.retriever:
            return "Please upload and process documents first."

        sample_docs = self.retriever.invoke("definitions terminology core concepts formulas")
        context_text = "\n\n".join(doc.page_content for doc in sample_docs)

        prompt = f"""Extract a Glossary of Key Terms and Definitions from the following course materials:

Context:
{context_text}
"""
        return self.llm.invoke(prompt).content