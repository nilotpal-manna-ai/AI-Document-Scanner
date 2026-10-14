# AI Document Scanner

An AI-powered document scanner built with Python, Streamlit, Google Gemini, and Retrieval-Augmented Generation (RAG).

## Features
- PDF document reading
- CSV file reading
- Excel file reading
- Word (DOCX) file reading
- Automatic document routing using Reader Agents
- Text extraction and chunking
- FAISS vector search
- Ask questions about uploaded documents using Gemini AI

## Technologies
Python, Streamlit, LangChain, Google Gemini, FAISS, Pandas, PyPDF, OpenPyXL, Python-docx.

## Project Files
- `app.py` — Main Streamlit application
- `manager_agent.py` — Document routing
- `pdf_agent.py` — PDF reader
- `csv_agent.py` — CSV reader
- `excel_agent.py` — Excel reader
- `word_agent.py` — Word reader
- `rag.py` — RAG and vector search
- `requirements.txt` — Dependencies

## Installation
1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Create a `.env` file with your Gemini API key.
4. Run `streamlit run app.py`.

## Live Demo
https://ai-document-scanner-yvgnssg4lbjdgvssg9ddds.streamlit.app/

## GitHub Repository
https://github.com/nilotpal-manna-ai/AI-Document-Scanner

## Author
Nilotpal Manna
