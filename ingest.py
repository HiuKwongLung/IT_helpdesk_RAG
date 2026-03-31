import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import re

# Remove ** in the orignal document
def clean_data(text):
    text = re.sub(r"\*+", "", text)
    return text.strip()

# Load data
def load_doc(csv_path):
    df = pd.read_csv(csv_path)
    docs = []

    for _, row in df.iterrows():
        topic = str(row.get("ki_topic", "")).strip()
        main_text = str(row.get("ki_text", "")).strip()
        alt_text = str(row.get("alt_ki_text", "")).strip()

        # Clean text
        main_text = clean_data(main_text)
        alt_text = clean_data(alt_text)

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

# Chunking
all_chunks = []
for doc in docs:
    chunks = chunk_text(doc, 128)
    all_chunks.extend(chunks)

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