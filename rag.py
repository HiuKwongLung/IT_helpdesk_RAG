import requests
import faiss
import pickle
from sentence_transformers import SentenceTransformer


def load_vector_db():
    index = faiss.read_index("vector_store/faiss_index.index")
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    return index, chunks, model


def get_answer(query):
    index, chunks, model = load_vector_db()

    # Retrieval
    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, k=3)

    results = [chunks[i] for i in I[0]]

    # Build prompt
    context = "\n\n".join(results)

    prompt = f"""
    Use the following context to answer the question.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """

    # Call LLM
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# --- For Testing only --- Remove after app.py is done
query = "My Microsoft excel is not working"
answer = get_answer(query)
print(answer)
