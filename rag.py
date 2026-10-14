import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def create_vector_store(documents):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def retrieve_context(vector_store, question, k=4):

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results