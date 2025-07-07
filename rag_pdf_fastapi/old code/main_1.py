from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os

from app.rag_utils import initialize, extract_text_from_pdf, split_text, store_chunks, ask_question
from app.config import PINECONE_API_KEY  # make sure this contains your Pinecone API key

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Run once when FastAPI app starts
@app.on_event("startup")
def on_startup():
    print("🚀 FastAPI startup: calling initialize()")
    initialize(pinecone_api_key=PINECONE_API_KEY)

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

    count = store_chunks(chunks)
    print(f"📌 Stored {count} chunks")

    return {
        "filename": file.filename,
        "text_length": len(text),
        "chunks_stored": count
    }

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
