
import os
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from manager_agent import ManagerAgent
from rag import create_vector_store, retrieve_context
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="AI Document Scanner",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Scanner")
st.caption("Agentic AI | RAG | Gemini | Streamlit")

st.write(
    "Upload a PDF, CSV, Excel, or DOCX document. "
    "The appropriate Reader Agent will process it."
)

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "csv", "xlsx", "xls", "docx"]
)

if uploaded_file is not None:

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        st.session_state.document_name = None
        st.session_state.processed_name = None
        st.session_state.processed_text = ""
        st.session_state.processed_result = None

    if st.session_state.processed_name != uploaded_file.name:

        st.session_state.vector_store = None
        st.session_state.document_name = None
        st.session_state.processed_name = None
        st.session_state.processed_text = ""
        st.session_state.processed_result = None

        suffix = Path(uploaded_file.name).suffix.lower()
        temp_path = None

        try:
            with NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner("Reading your document..."):
                result = ManagerAgent(temp_path)

            # Limit embedding requests for the initial test.
            documents = result["documents"][:50]

            if not documents:
                st.error("No readable text was found.")
                st.stop()

            st.session_state.processed_name = uploaded_file.name
            st.session_state.processed_result = result
            st.session_state.processed_text = result["text"]

            st.info(
                f"Document contains {result['total_chunks']} chunks. "
                f"This test will embed only {len(documents)} chunks."
            )

            with st.spinner("Creating RAG knowledge base..."):
                vector_store = create_vector_store(documents)

            st.session_state.vector_store = vector_store
            st.session_state.document_name = uploaded_file.name

            st.success("Document processed and RAG is ready!")

        except Exception as error:
            st.error(f"Document processing failed: {error}")

        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    if st.session_state.processed_result is not None:

        result = st.session_state.processed_result

        col1, col2, col3 = st.columns(3)

        col1.metric("Document", uploaded_file.name)
        col2.metric("Total RAG Chunks", result["total_chunks"])
        col3.metric(
            "Document Type",
            Path(uploaded_file.name).suffix.upper().replace(".", "")
        )

        with st.expander("View extracted text"):
            st.text_area(
                "Extracted content",
                st.session_state.processed_text,
                height=250
            )

        st.subheader("💬 Ask Questions")

        question = st.text_input(
            "Enter your question about the document",
            key="document_question",
            placeholder="Summarize the main points..."
        )

        if st.button("Get Answer", type="primary"):

            if not question.strip():
                st.warning("Please enter a question.")

            elif st.session_state.vector_store is None:
                st.warning(
                    "The RAG knowledge base is not ready. "
                    "Please upload the document again."
                )

            else:
                try:
                    with st.spinner("Finding relevant information..."):

                        context_docs = retrieve_context(
                            st.session_state.vector_store,
                            question,
                            k=4
                        )

                        context = "\n\n".join(
                            doc.page_content
                            for doc in context_docs
                        )

                        prompt = ChatPromptTemplate.from_template(
                            """Answer the question using only the context.
If the answer is not available, say the document does not provide
enough information.

Context:
{context}

Question:
{question}

Answer:"""
                        )

                        llm = ChatGoogleGenerativeAI(
                           model="gemini-3.1-flash-lite",
                            google_api_key=os.getenv("GEMINI_API_KEY"),
                            temperature=0.2
                        )

                        chain = prompt | llm

                        response = chain.invoke({
                            "context": context,
                            "question": question
                        })

                    st.markdown("### Answer")
                    st.write(response.content)

                    with st.expander("View retrieved context"):
                        for index, doc in enumerate(
                            context_docs, start=1
                        ):
                            st.markdown(f"**Source {index}**")
                            st.write(doc.page_content)
                            st.caption(str(doc.metadata))

                except Exception as error:
                    st.error(f"Answer generation failed: {error}")

st.divider()
st.caption("AI Document Scanner | Python • Streamlit • Gemini • RAG")
