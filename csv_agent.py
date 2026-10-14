import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def CSVReaderAgent(file_path):
    """Reads a CSV file and prepares its content for RAG."""

    path = Path(file_path)

    # 1. Validate file
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Please provide a CSV file.")

    # 2. Read CSV
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The CSV file is empty.")

    # 3. Convert rows into documents
    documents = []

    for index, row in df.iterrows():
        row_text = "\n".join(
            f"{column}: {row[column]}"
            for column in df.columns
        )

        documents.append(
            Document(
                page_content=row_text,
                metadata={
                    "source": path.name,
                    "row": int(index) + 1,
                    "file_type": "CSV"
                }
            )
        )

    # 4. Split documents for RAG
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # 5. Return results
    return {
        "file_name": path.name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_chunks": len(chunks),
        "text": df.to_string(index=False),
        "documents": chunks
    }