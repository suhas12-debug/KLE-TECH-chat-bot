import numpy as np
import json
from sentence_transformers import SentenceTransformer

# NOTE: This diagnostic tool MUST use the same embedding model and query prefix
# as chat.py (EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5") to produce valid scores.
# Using a different model will compare embeddings from different vector spaces,
# making all scores inaccurate and useless for debugging.
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"

def diagnostic(query: str = "give thursday timetable for 6th sem d division", top_k: int = 10):
    print(f"Loading embedding model: '{EMBED_MODEL_ID}'...")
    model = SentenceTransformer(EMBED_MODEL_ID, device='cpu')

    # Load knowledge base
    print("Loading knowledge base...")
    embeddings = np.load('embeddings.npy')
    with open('facts.json', 'r', encoding='utf-8') as f:
        facts = json.load(f)

    # BGE models require this instruction prefix on the query side (NOT on the facts side)
    # This matches the prefix used in chat.py's retrieve() method.
    instruction = "Represent this sentence for searching relevant passages: "
    query_with_instruction = instruction + query
    query_emb = model.encode([query_with_instruction])

    # Compute cosine similarity (same formula as chat.py)
    similarities = np.dot(embeddings, query_emb.T).flatten() / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    # Find top K
    best_indices = np.argsort(similarities)[-top_k:][::-1]

    print(f"\nQuery: {query}")
    print(f"Model: {EMBED_MODEL_ID}")
    print(f"Knowledge base size: {len(facts)} facts")
    print(f"\nTop {top_k} Retrieved Facts (valid scores):")
    print("-" * 80)
    for rank, i in enumerate(best_indices, 1):
        print(f"[#{rank}] Score: {similarities[i]:.4f} | {facts[i][:120]}...")
    print("-" * 80)

if __name__ == "__main__":
    # You can change this query to debug any retrieval issue
    diagnostic(query="give thursday timetable for 6th sem d division", top_k=10)
