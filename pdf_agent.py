from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def PDFReaderAgent(file_path):
    """
    Reads a PDF, extracts text page by page,
    and prepares document chunks for RAG.
    """

    path = Path(file_path)

    # 1. Validate the file
    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    # 2. Read the PDF
    reader = PdfReader(str(path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    # 3. Extract text page by page
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "file_type": "PDF"
                    }
                )
            )

    if not documents:
        raise ValueError(
            "No readable text found. This PDF may require OCR."
        )

    # 4. Split text into chunks for RAG
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # 5. Return extracted pages and RAG chunks
    return {
        "file_name": path.name,
        "total_pages": len(reader.pages),
        "pages_with_text": len(documents),
        "total_chunks": len(chunks),
        "text": "\n\n".join(
            doc.page_content for doc in documents
        ),
        "documents": chunks
    }