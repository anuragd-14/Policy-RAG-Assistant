from main import rag_chain

# Evaluation Set (8 Questions) 
# We include answerable, partially answerable, and unanswerable questions.
test_questions = [
    "What is the return policy period?",            # Answerable (Fact)
    "How much does express shipping cost?",         # Answerable (Fact)
    "What is the cancellation deadline?",           # Answerable (Fact)
    "Can I return a gift card?",                    # Answerable (Negative constraint)
    "Do you ship to Canada?",                       # Answerable (List)
    "What happens if I cancel after 12 hours?",     # Answerable (Condition)
    "Do you offer shipping to Mars?",               # Unanswerable (Hallucination check) [cite: 57]
    "Can I change my order items after placement?"  # Missing Info (Edge case) [cite: 58]
]

print("--- STARTING EVALUATION ---")
results = []

for q in test_questions:
    print(f"\nQuestion: {q}")
    try:
        # Invoke the RAG chain
        res = rag_chain.invoke(q)
        print(f"Answer: {res}")
        results.append((q, res))
    except Exception as e:
        print(f"Error: {e}")

print("\n--- EVALUATION COMPLETE ---")