Policy RAG Assistant
1. Overview
This is a Retrieval-Augmented Generation (RAG) system developed as an AI Engineer intern assignment. It serves as a question-answering assistant for company policy documents (e.g., Refund, Shipping, Cancellation policies). The system is built using Python and LangChain, utilizing Groq for high-speed inference and ChromaDB for vector storage.

The core focus of this project is on retrieval accuracy, hallucination control, and robust prompt engineering.

2. Architecture
The system follows a standard RAG pipeline:

Ingestion: Policy documents are loaded and split into smaller chunks.

Embedding: Text chunks are converted into vector embeddings using HuggingFace (all-MiniLM-L6-v2) and stored locally in ChromaDB.

Retrieval: When a user asks a question, the system performs a semantic similarity search to find the top 3 most relevant context chunks.

Generation: The retrieved context and the user's question are passed to the Groq LLM (llama-3.1-8b-instant) with a strictly engineered prompt to generate the final answer.

3. Setup Instructions

> Prerequisites:
    1. Python 3.8+
    2. A Free Groq API Key

> Installation:
    1. Clone the repository
