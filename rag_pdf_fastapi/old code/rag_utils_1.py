import io
import time
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pinecone  # updated import style

# Initialize global variables
embedding_model = None
qa_pipeline = None
index = None

def initialize(pinecone_api_key, index_name="simple-free-rag", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", llm_model_name="google/flan-t5-base", index_dimension=384):
    """Initialize embedding model, LLM pipeline, Pinecone client and index."""
    global embedding_model, qa_pipeline, index

    print("🔄 Loading embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)
    print("✅ Embedding model loaded!")

    print("🔄 Loading language model...")
    qa_pipeline = pipeline(
        "text2text-generation",
        model=llm_model_name,
        max_length=512,
        temperature=0.1
    )
    print("✅ Language model loaded!")

    print("🔄 Initializing Pinecone...")
    pinecone.init(api_key=pinecone_api_key)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=index_dimension,
            metric="cosine"
        )
        print(f"✅ Created Pinecone index: {index_name}")
        time.sleep(30)  # wait for index to be ready

    index = pinecone.Index(index_name)
    print("✅ Connected to Pinecone index")

def extract_text_from_pdf(file_bytes):
    """Extract all text from a PDF bytes object."""
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

def split_text(text, chunk_size=800, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def store_chunks(chunks, batch_size=100):
    """Create embeddings for chunks and upload them to Pinecone index."""
    if embedding_model is None or index is None:
        raise RuntimeError("Call initialize() before storing chunks.")

    print("🔄 Creating embeddings for chunks...")
    chunk_embeddings = embedding_model.encode(chunks)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding.tolist() if not isinstance(embedding, list) else embedding,
            "metadata": {"text": chunk}
        })

    print("🔄 Uploading chunks to Pinecone...")
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"✅ Stored {len(vectors)} chunks in Pinecone.")
    return len(vectors)

def ask_question(question, k=3):
    """Given a question, query Pinecone for context and generate answer via LLM."""
    if embedding_model is None or qa_pipeline is None or index is None:
        raise RuntimeError("Call initialize() before asking questions.")

    try:
        question_embedding = embedding_model.encode([question])[0].tolist()
        results = index.query(
            vector=question_embedding,
            top_k=k,
            include_metadata=True
        )
        relevant_chunks = [match.metadata['text'] for match in results.matches]
        context = "\n\n".join(relevant_chunks)

        prompt = f"Answer the question based on the following context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        answer = qa_pipeline(prompt)[0]['generated_text']
        return answer, relevant_chunks

    except Exception as e:
        return f"Error: {e}", []

