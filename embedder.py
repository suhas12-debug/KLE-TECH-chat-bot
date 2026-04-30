from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

def generate_embeddings():
    print("Loading SentenceTransformer model ('BAAI/bge-small-en-v1.5')...")
    # SBERT runs on CPU as per constraints
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    
    facts = []
    input_file = 'college_data.jsonl'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Reading facts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                facts.append(data['text'])
    
    print(f"Encoding {len(facts)} facts... (This may take a minute on CPU)")
    embeddings = model.encode(facts, show_progress_bar=True)
    
    print("Saving embeddings to 'embeddings.npy'...")
    np.save('embeddings.npy', embeddings)
    
    print("Saving fact texts to 'facts.json'...")
    with open('facts.json', 'w', encoding='utf-8') as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)
    
    print("Done! Embeddings and facts saved successfully.")

if __name__ == "__main__":
    generate_embeddings()
