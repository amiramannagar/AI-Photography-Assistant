from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# ✅ import pipeline
from data_loader import build_chunks


# =========================
# BUILD DATA
# =========================

PDF_PATH = "RAG Data/pdfs"
CSV_PATH = "RAG Data/photography_rag_dataset_clean.csv"

chunks = build_chunks(PDF_PATH, CSV_PATH)


# =========================
# EMBEDDINGS + VECTORSTORE
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# =========================
# MODEL
# =========================

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# =========================
# GENERATION
# =========================

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# PROMPT
# =========================

template = """
You are a photography assistant.

Return ONLY:

ISO: <number>
Aperture: f/<number>
Shutter Speed: 1/<number>

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate.from_template(template)


# =========================
# PARSER
# =========================

def parse_output(text):

    iso = None

    iso_match = re.search(r'ISO[:\s]*([0-9]{2,7})', text)
    if iso_match:
        raw_iso = iso_match.group(1)
        iso = raw_iso[:3] if len(raw_iso) > 4 else raw_iso

    aperture = re.search(r'f/\d+\.?\d*', text)
    shutter = re.search(r'1/\d+', text)

    return {
        "settings": {
            "iso": iso,
            "aperture": aperture.group(0) if aperture else None,
            "shutter_speed": shutter.group(0) if shutter else None
        }
    }


# =========================
# RAG FUNCTION
# =========================

def ask_rag(question):

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    response = generate_text(final_prompt)

    return parse_output(response)
