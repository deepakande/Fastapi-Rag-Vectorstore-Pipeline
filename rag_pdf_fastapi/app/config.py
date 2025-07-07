import os
from dotenv import load_dotenv

# Load environment variables from a .env file (if exists)
load_dotenv()

# 🔐 Pinecone API Key
PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
    "pcsk_6KFiZf_jHoSM45Bi8vVS8dNrQrpQbQknoLNdkTfqLPymzS8RrkYT8GjvKdsY7tPZbjjSX"
)

# 📌 Pinecone Index Configuration
INDEX_NAME = "simple-free-rag"
INDEX_DIMENSION = 384

# 🔎 Embedding & LLM Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

