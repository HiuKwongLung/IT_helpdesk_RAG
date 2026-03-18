import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

def load_doc(csv_path):
    df = pd.read_csv(csv_path)
    docs = []

    for _, row in df.iterrows():
        topic = str(row.get("ki_topic", "")).strip()
        main_text = str(row.get("ki_text", "")).strip()
        alt_text = str(row.get("alt_ki_text", "")).strip()

        #Add documents for both main and alt text
        if main_text:
            doc_main = f"""
Topic: {topic}

Details:
{main_text}
"""
            docs.append(doc_main)

        # Document 2: alternative text
        if alt_text:
            doc_alt = f"""
Topic: {topic}

Details:
{alt_text}
"""
            docs.append(doc_alt)
    return docs

#Turn document in to small chunks
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

#Load CSV
csv_path = "data/synthetic_knowledge_items.csv"
docs= load_doc(csv_path)
"""
print(f"total documents: {len(docs)}")
print("Sample document:\n)")
print(docs[0])
"""

#Chunking
all_chunks = []
for doc in docs:
    chunks = chunk_text(doc, 128)
    all_chunks.extend(chunks)

"""
print(f"Total chunks: {len(all_chunks)}\n")
for i in range(3):
    print(f"--- Chunk {i} ---\n")
    print(all_chunks[i])
"""

#Generate embeddings
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
embeddings = model.encode(all_chunks)

#Store in vector DB
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

#Retrieval
query = "Troubleshooting Issues with Microsoft Office"

query_embedding = model.encode([query])

D, I = index.search(query_embedding, k=3)

results = [all_chunks[i] for i in I[0]]

#Build prompt
context = "\n\n".join(results)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

#Call LLM
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt
    }
)

answer = response.json()["response"]
print(answer)

