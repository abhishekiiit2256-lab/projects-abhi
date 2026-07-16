import math
import re
from collections import Counter
import ollama

# 1. Sample Knowledge Base (Context)
documents = [
    "The Gemini 2.5 Flash model is optimized for high-frequency tasks where speed and cost are highly critical.",
    "Retrieval-Augmented Generation (RAG) is a technique that grants LLMs access to external facts to ground their responses.",
    "Cosine similarity measures the cosine of the angle between two non-zero vectors, determining how similar they are.",
    "To cook the perfect soft-boiled egg, lower large eggs into boiling water and cook them gently for exactly six minutes.",
    "Python is a high-level, interpreted programming language known for its readability and vast ecosystem."
]

def tokenize(text: str) -> list[str]:
    """Cleans text into a list of lowercase words."""
    return re.findall(r'\w+', text.lower())

def run_manual_rag(query: str, model_name: str = "llama3:latest"):
    print(f"User Query: '{query}'\n")
    
    # Step A: Tokenize and count the query words
    query_tokens = tokenize(query)
    query_counter = Counter(query_tokens)
    
    stored_results = []
    
    print("STEP-BY-STEP MATH BREAKDOWN FOR EACH DOCUMENT")
    
    # Step B: Compute cosine similarity for each document
    for doc in documents:
        doc_tokens = tokenize(doc)
        doc_counter = Counter(doc_tokens)
        
        unique_words = list(set(query_counter.keys()).union(set(doc_counter.keys())))
        
        dot_product = sum(query_counter[w] * doc_counter[w] for w in unique_words)
        
        magnitude_query = math.sqrt(sum(c ** 2 for c in query_counter.values()))
        magnitude_doc = math.sqrt(sum(c ** 2 for c in doc_counter.values()))
        
        similarity_score = (
            dot_product / (magnitude_query * magnitude_doc)
            if magnitude_query and magnitude_doc else 0.0
        )
        
        print(f"Doc: '{doc[:45]}...'")
        print(f"   ↳ Dot Product (Top): {dot_product}")
        print(f"   ↳ Magnitudes (Bottom): {magnitude_query:.4f} * {magnitude_doc:.4f} = {magnitude_query * magnitude_doc:.4f}")
        print(f"   ↳ Resulting Score: {similarity_score:.4f}\n")
        
        stored_results.append({
            "score": similarity_score,
            "text": doc,
            "math_metadata": {
                "dot_product": dot_product,
                "magnitude": magnitude_doc
            }
        })
        
    # Step C: Sort results
    stored_results.sort(key=lambda item: item["score"], reverse=True)
    winner = stored_results[0]
    best_context = winner["text"]
    best_score = winner["score"]
    
    print("-------------------------------------------------------")
    print(f" WINNING CONTEXT SELECTED (Score: {best_score:.4f}):\n\"{best_context}\"\n")
    
    # Step D: Construct prompt
    system_instruction = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the answer cannot be found in the context, say 'I cannot find the answer.'"
    )
    user_prompt = f"Context: {best_context}\n\nQuestion: {query}"
    
    # Step E: Forward to Ollama
    print("-> Forwarding prompt package to local Ollama...")
    try:
        available_models = [m["model"] for m in ollama.list()["models"]]
        if model_name not in available_models:
            print(f"\n Model '{model_name}' not found. Installed models: {available_models}")
            print(f" Run `ollama pull {model_name}` before retrying.\n")
            return
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.0}
        )
        
        print("\n=== Final Local AI Response ===")
        print(response["message"]["content"])
        
    except Exception as e:
        print(f"\nOllama Error: {e}")

#Execute the Pipeline
if __name__ == "__main__":
    user_query = "what is cosine similarity?"
    run_manual_rag(user_query, model_name="llama3:latest")
