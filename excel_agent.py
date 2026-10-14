import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def ExcelReaderAgent(file_path):
    """Reads Excel files and prepares content for RAG."""

    path = Path(file_path)

    # 1. Validate file
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    if path.suffix.lower() not in [".xlsx", ".xls"]:
        raise ValueError("Please provide an Excel file.")

    # 2. Read all sheets
    sheets = pd.read_excel(path, sheet_name=None)

    documents = []
    total_rows = 0

    # 3. Convert each row into a document
    for sheet_name, df in sheets.items():

        total_rows += len(df)

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
                        "sheet": str(sheet_name),
                        "row": int(index) + 1,
                        "file_type": "Excel"
                    }
                )
            )

    if not documents:
        raise ValueError("No readable data found in the Excel file.")

    # 4. Split documents for RAG
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # 5. Return results
    return {
        "file_name": path.name,
        "total_sheets": len(sheets),
        "total_rows": total_rows,
        "total_chunks": len(chunks),
        "text": "\n\n".join(
            f"Sheet: {name}\n{df.to_string(index=False)}"
            for name, df in sheets.items()
        ),
        "documents": chunks
    }