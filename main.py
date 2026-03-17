import pandas as pd

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




# Test Load CSV
csv_path = "data/synthetic_knowledge_items.csv"
docs= load_doc(csv_path)

print(f"total documents: {len(docs)}")
print("Sample document:\n)")
print(docs[0])
