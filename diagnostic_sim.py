import torch
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def diagnostic():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # Load knowledge
    embeddings = np.load('embeddings.npy')
    with open('facts.json', 'r', encoding='utf-8') as f:
        facts = json.load(f)
        
    query = "give thursday timetable for 6th sem d division"
    query_emb = model.encode([query])
    
    # Calculate similarities
    similarities = np.dot(embeddings, query_emb.T).flatten() / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    
    # Find top 10
    best_indices = np.argsort(similarities)[-10:][::-1]
    
    print(f"Query: {query}\n")
    print("Top 10 Retrieved Facts:")
    for i in best_indices:
        print(f"Score: {similarities[i]:.4f} | Fact: {facts[i]}")

if __name__ == "__main__":
    diagnostic()
