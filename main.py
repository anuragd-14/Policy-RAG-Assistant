
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# CHANGED: Import Groq instead of OpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Setup Vector Store
# Must use the same embedding model as ingest.py
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 2. Define Prompt
template = """
You are a helpful customer support assistant. 
Your task is to answer the user's question based STRICTLY on the provided context below.

Rules:
1. If the answer is not in the context, say "I cannot answer this based on the available policies."
2. Use bullet points for list-based answers.
3. Keep answers concise.

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# 3. Setup LLM (Using Free Groq Model)
# We use Llama3-8b-8192 which is fast and free
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

# 4. Build Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run Query
# if __name__ == "__main__":
#     # Test Question 1
#     question1 = "What is the refund policy duration?"
#     print(f"\n--- Question: {question1} ---")
#     response = rag_chain.invoke(question1)
#     print(f"Answer:\n{response}")

#     # Test Question 2 (Edge Case)
#     question2 = "Do you ship to the moon?"
#     print(f"\n--- Question: {question2} ---")
#     response = rag_chain.invoke(question2)
#     print(f"Answer:\n{response}")

# ... (Keep all your imports and chain setup code above unchanged) ...

# 5. Interactive Loop
if __name__ == "__main__":
    print("---------------------------------------------------------")
    print("🤖 Policy Assistant Ready! (Type 'exit' or 'quit' to stop)")
    print("---------------------------------------------------------")
    
    while True:
        # Get user input
        user_input = input("\n👉 You: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
            
        # Ignore empty inputs
        if not user_input:
            continue
            
        # Run the RAG chain
        print("🤖 AI: Thinking...", end="\r") # Simple loading effect
        try:
            response = rag_chain.invoke(user_input)
            # Clear "Thinking..." line and print result
            print(f"🤖 AI: {response}") 
        except Exception as e:
            print(f"❌ Error: {e}")