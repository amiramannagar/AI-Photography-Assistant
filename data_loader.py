from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import os


# =========================
# PDF LOADER
# =========================

def load_pdfs(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            documents.extend(docs)

    return documents


# =========================
# CSV LOADER
# =========================

def load_csv(file_path):
    df = pd.read_csv(file_path)

    documents = []

    for _, row in df.iterrows():
        text = f"""
Scenario: {row.get('scenario', '')}
Settings: {row.get('settings', '')}
Tips: {row.get('tips', '')}
"""
        documents.append(text)

    return documents


# =========================
# NORMALIZE CSV
# =========================

def normalize_csv(csv_docs):
    return [
        Document(
            page_content=text,
            metadata={"source": "csv"}
        )
        for text in csv_docs
    ]


# =========================
# MERGE DATA
# =========================

def load_all_data(pdf_folder, csv_path):
    pdf_docs = load_pdfs(pdf_folder)
    csv_docs = load_csv(csv_path)

    csv_docs = normalize_csv(csv_docs)

    return pdf_docs + csv_docs


# =========================
# CHUNKING
# =========================

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


# =========================
# PIPELINE RUN
# =========================

def build_chunks(pdf_path, csv_path):

    docs = load_all_data(pdf_path, csv_path)
    chunks = chunk_documents(docs)

    print("Total docs:", len(docs))
    print("Total chunks:", len(chunks))

    return chunks