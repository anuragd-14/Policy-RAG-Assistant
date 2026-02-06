import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# CHANGED: Import HuggingFace embeddings instead of OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 1. Load Data
loader = TextLoader("./data/policies.txt")
documents = loader.load()

# 2. Chunk Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

# 3. Embed & Store
# CHANGED: Use a free local model ("all-MiniLM-L6-v2")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

print("Ingestion complete. Vector store created in './chroma_db'.")