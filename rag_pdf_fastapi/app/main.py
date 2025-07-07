from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os

# Import app modules
from app.config import PINECONE_API_KEY
from app.rag_utils import (
    extract_text_from_pdf,
    split_text,
    store_chunks,
    ask_question,
    initialize
)
from app.database import SessionLocal
from app.models import ChunkMetadata

# Initialize FastAPI app
app = FastAPI()

# Triggered when FastAPI starts
@app.on_event("startup")
def on_startup():
    print("🚀 FastAPI startup: calling initialize()")
    initialize(pinecone_api_key=PINECONE_API_KEY)

# Ensure upload directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================
# 📥 Upload PDF Endpoint
# ======================
@app.post("/upload/")
async def upload_file(file: UploadFile):
    print("📥 /upload/ endpoint hit")

    contents = await file.read()
    print(f"📦 Received file: {file.filename} ({len(contents)} bytes)")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    print(f"✅ File saved to {file_path}")

    text = extract_text_from_pdf(contents)
    print(f"📄 Extracted text length: {len(text)}")

    chunks = split_text(text)
    print(f"✂️ Split into {len(chunks)} chunks")

    count = store_chunks(chunks, filename=file.filename)
    print(f"📌 Stored {count} chunks")

    return {
        "filename": file.filename,
        "text_length": len(text),
        "chunks_stored": count
    }

# ============================
# ❓ Ask Question Endpoint
# ============================
@app.post("/ask/")
async def question_answer(question: str = Form(...)):
    print("❓ /ask/ endpoint hit")
    print("🧠 Question received:", question)

    answer, sources = ask_question(question)
    print("✅ Answer generated")

    return JSONResponse(content={
        "question": question,
        "answer": answer,
        "sources": sources
    })

# ================================
# 📄 View All Chunks from MySQL
# ================================
'''@app.get("/chunks/")
def get_all_chunks():
    db = SessionLocal()
    chunks = db.query(ChunkMetadata).all()
    db.close()

    return [
        {
            "id": chunk.id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text[:100],  # limit preview
            "filename": chunk.filename,
            "created_at": chunk.created_at.isoformat()
        }
        for chunk in chunks
    ] '''

