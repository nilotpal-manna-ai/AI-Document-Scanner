from pathlib import Path

from pdf_agent import PDFReaderAgent
from csv_agent import CSVReaderAgent
from excel_agent import ExcelReaderAgent
from word_agent import WordReaderAgent


def ManagerAgent(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return PDFReaderAgent(file_path)

    elif extension == ".csv":
        return CSVReaderAgent(file_path)

    elif extension in [".xlsx", ".xls"]:
        return ExcelReaderAgent(file_path)

    elif extension == ".docx":
        return WordReaderAgent(file_path)

    else:
        raise ValueError("Unsupported file format.")


if __name__ == "__main__":
    print("AI Document Scanner - Manager Agent Ready")