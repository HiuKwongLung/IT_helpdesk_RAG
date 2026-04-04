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

def retrieve(query, k=3):
    index, chunks, model = load_vector_db()

    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=k)

    results = [chunks[i] for i in I[0]]

    return results

def get_answer(query, retrieved_chunks):
    # Build prompt
    context = "\n".join(retrieved_chunks)

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