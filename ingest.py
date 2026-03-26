import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Load data
def load_doc(csv_path):
    df = pd.read_csv(csv_path)
    docs = []

    for _, row in df.iterrows():
        topic = str(row.get("ki_topic", "")).strip()
        main_text = str(row.get("ki_text", "")).strip()
        alt_text = str(row.get("alt_ki_text", "")).strip()

        # Add documents for both main and alt text
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

# Turn document in to small chunks
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

# Load CSV
csv_path = "data/synthetic_knowledge_items.csv"
docs= load_doc(csv_path)
"""
print(f"total documents: {len(docs)}")
print("Sample document:\n)")
print(docs[0])
"""

# Chunking
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

# Generate embeddings
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
embeddings = model.encode(all_chunks)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# Save index
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/faiss_index.index")
with open("vector_store/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("FAISS index saved")

