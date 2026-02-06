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

        git clone https://github.com/anuragd-14/Policy-RAG-Assistant.git
        cd RAG_assignment
        
    2. Create a virtual environment:
        python -m venv venv
        # Windows
        .\venv\Scripts\activate
        # Mac/Linux
        source venv/bin/activate

    3. Install dependencies:
        pip install -r requirements.txt

    4. Configure Environment: Create a file named .env in the root directory and add your API key:
        GROQ_API_KEY=gsk_your_actual_key_here

> Usage:

     1. Process the Data: Run the ingestion script to create the vector database.
        python ingest.py
        (Ensure your data is in data/policies.txt)
    
    2. Run the Chatbot: Start the interactive assistant.
        python main.py
        
<img width="1301" height="493" alt="Screenshot 2026-02-06 191734" src="https://github.com/user-attachments/assets/86b17e1a-20b3-4e3e-99ba-a399c11db422" />


    3. Run Evaluation: Test the system against the predefined question set.
        python evaluate.py
        

<img width="1301" height="493" alt="Screenshot 2026-02-06 192140" src="https://github.com/user-attachments/assets/b04f0c83-46b2-483f-82c6-0ff62e3d8134" />


4. Design Decisions

   > Data Chunking Strategy:
    I chose a chunk size of 500 characters with a 50-character overlap.
            Reasoning: Company policies are often dense and clause-heavy. A massive chunk size (e.g., 2000 chars) might mix unrelated policies (like Shipping and Refunds) into the same context, confusing the model. A size of 500 chars captures complete individual rules while maintaining precise retrieval context.

    > Vector Storage
        - ChromaDB: Selected for its simplicity and ability to run locally without needing a cloud vector database account.
        - Embeddings: all-MiniLM-L6-v2 (HuggingFace) was chosen over OpenAI embeddings to keep the project cost-efficient and capable of running offline for the embedding step.

5. Prompt Engineering (Iteration Process)
    One of the key requirements was to improve the prompt to handle edge cases.

   > Iteration 1: The Baseline (Initial Attempt)
        My first prompt was a generic instruction.
           Template: "Answer the question based on the context below: {context} Question: {question}"
        Critique: This prompt failed the "Negative Constraint" test. When asked "Do you ship to Mars?", the model attempted to be helpful by using general knowledge or making up a plausible answer, rather than                 admitting the policy didn't exist.

   > Iteration 2: The Final Version (Implemented)
            I refined the prompt to strictly bound the model to the provided text.
            Template:
            "You are a helpful customer support assistant. Your task is to answer the user's question based STRICTLY on the provided context below. Rules:
            - If the answer is not in the context, say 'I cannot answer this based on the available policies.'
            - Use bullet points for list-based answers.
            - Keep answers concise."
            Improvements:
            - Role Prompting: Sets a professional tone.
            - Negative Constraints: Explicitly forbade answering if the context was missing.
            - Formatting: Forced structured output (bullet points) for readability.

6. Evaluation Results:
I evaluated the system using a set of 8 questions covering factual retrieval, edge cases, and missing information.

<img width="843" height="509" alt="image" src="https://github.com/user-attachments/assets/78bc087b-5c43-46c8-848e-4da319e6fce1" />

7. Future Improvements
Given more time, I would implement the following upgrades:
    > Hybrid Search: Currently, the system relies purely on semantic similarity. I would add a keyword-based search (BM25) to better handle specific product names or exact terms that semantic search might miss.
    > Source Citations: Modify the prompt to require the model to cite the specific document or section name (e.g., [Source: Refund Policy]) for every claim it makes.
    > Dockerization: Containerize the application to ensure it runs consistently across any environment without dependency conflicts.

