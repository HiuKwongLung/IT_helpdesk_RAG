import time
from rag import retrieve, get_answer

# 20 test queries with expected keywords
test_queries = [
    {
        "question": "I forgot my PIN, how do i reset it?",
        "keywords": ["pin", "reset", "credentials", "verify"]
    },
    {
        "question": "How to set up conference call on Cisco Webx?",
        "keywords": ["meetings", "audio", "video", "cisco"]
    },
    {
        "question": "How to back up a file?",
        "keywords": ["file", "drive", "backup", "copy"]
    },
    {
        "question": "I am having problem with the company tablet, what do i do?",
        "keywords": ["tablet", "updates", "reset", "settings"]
    },
    {
        "question": "How to set up a secure wireless network?",
        "keywords": ["protocol", "password", "network", "firewall"]
    },
    {
        "question": "How to reset a jammed printer?",
        "keywords": ["printer", "remove", "paper", "clean"]
    },
    {
        "question": "I am having problems with Citrix, what do i do?",
        "keywords": ["update", "verify", "authentication", "login"]
    },
    {
        "question": "how do i set up network on a new printer?",
        "keywords": ["ip", "driver", "configuring", "printer"]
    },
    {
        "question": "how to create a restore point on window?",
        "keywords": ["window", "restore", "system", "protection"]
    },
    {
        "question": "I am having audio problem with my laptop, what do i do?",
        "keywords": ["sound", "audio", "update", "manager"]
    },
    {
        "question": "How to set up a secure FTP connection?",
        "keywords": ["ftp", "ip", "password", "server"]
    },
    {
        "question": "I forgot my username, how do i reset it?",
        "keywords": ["forgot", "portal", "email", "employee"]
    },
    {
        "question": "I am having problem with java, what do i do?",
        "keywords": ["java", "update", "settings", "logs"]
    },
    {
        "question": "How to create a new distribution list in exchange?",
        "keywords": ["exchange", "create", "configure", "members"]
    },
    {
        "question": "I am having problem with Adobe Flash, what do i do?",
        "keywords": ["version", "browser", "extensions", "cache"]
    }
]
def evaluate_retrieval(keywords, retrieved_chunks):
    text = " ".join(retrieved_chunks).lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits / len(keywords)

def evaluate_answer(answer, keywords):
    answer = answer.lower()
    hits = sum(1 for kw in keywords if kw in answer)
    return hits / len(keywords)

def run_evaluation():
    results = []

    for item in test_queries:
        query = item["question"]
        keywords = item["keywords"]

        print(f"\n Query: {query}")

        # Retrieval evaluation
        start_retrieval = time.time()
        retrieved_chunks = retrieve(query)
        end_retrieval = time.time()

        retrieval_score = evaluate_retrieval(keywords, retrieved_chunks)

        # LLM Answer evaluation
        start_generation = time.time()
        answer = get_answer(query, retrieved_chunks)
        end_generation = time.time()

        answer_score = evaluate_answer(answer, keywords)

        total_time = end_generation - start_retrieval

        retrieval_time = end_retrieval - start_retrieval
        generation_time = end_generation - start_generation

        print("Top Retrieved Chunks:")
        for chunk in retrieved_chunks[:2]:
            print("-", chunk[:300], "...")

        print("Answer:")
        print(answer)

        print(f"Retrieval Time: {retrieval_time:.2f}s")
        print(f"Generation Time: {generation_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")

        print(f"Retrieval Score: {retrieval_score:.2f}")
        print(f"Answer Score: {answer_score:.2f}")

        results.append({
            "query": query,
            "retrieval_score": retrieval_score,
            "answer_score": answer_score,
            "latency_retrieval": retrieval_time,
            "latency_generation": generation_time
        })
    return results

def summarize(results):
    avg_retrieval = sum(r["retrieval_score"] for r in results) / len(results)
    avg_answer = sum(r["answer_score"] for r in results) / len(results)
    avg_latency_retrieval = sum(r["latency_retrieval"] for r in results) / len(results)
    avg_latency_generation = sum(r["latency_generation"] for r in results) / len(results)

    print("Final Results")
    print(f"Avg Retrieval Score: {avg_retrieval:.2f}")
    print(f"Avg Answer Score: {avg_answer:.2f}")
    print(f"Avg Latency: {avg_latency_retrieval:.2f}s")
    print(f"Avg Latency: {avg_latency_generation:.2f}s")

if __name__ == "__main__":
    results = run_evaluation()
    summarize(results)


