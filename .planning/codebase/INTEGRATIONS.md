# Integrations

This document tracks all external services, APIs, and data sources integrated into the KLE Tech Chatbot.

## External Services
- **Hugging Face Hub**: Used to download pre-trained LLM weights (`Qwen`) and embedding models (`Sentence Transformers`).
  - Connectivity: Requires HTTPS access to `huggingface.co`.

## Data Sources
- **`generate_dataset.py`**: The "Source of Truth" script containing raw university knowledge in Python dict format.
- **`college_data.jsonl`**: The intermediate factual database extracted from the raw dataset.

## Internal Components
- **Vector Index**: `embeddings.npy` (Numpy-based cosine similarity index).
- **Text Mapping**: `facts.json` (Maps vector indices back to textual facts).
