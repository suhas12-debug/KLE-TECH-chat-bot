# Architecture

The KLE Tech Chatbot uses a **Retrieval-Augmented Generation (RAG)** architecture with a custom high-precision filtering layer.

## System Workflow

1.  **Ingestion**: `generate_dataset.py` contains raw academic data. This is processed into granular facts.
2.  **Indexing**: `embedder.py` converts facts into vector embeddings using `Sentence-BERT`.
3.  **Retrieval**:
    - **Semantic Search**: Uses cosine similarity to find the Top 20 most relevant facts.
    - **Hybrid Filter**: A rule-based layer in `chat.py` that validates the Semester (e.g., Semester 4 vs 6) and discards mismatching facts.
4.  **Generation**: The filtered facts are passed as context to the `Qwen2.5-0.5B-Instruct` model.
    - **Zero-Tolerance Prompting**: The LLM is instructed to refuse answering if the context doesn't match the specific day/semester requested.

## Component Diagram
- **`KLETechChatbot` Class**: Manages the loading of both models (SBERT and Qwen) and coordinates the Chat-Retrieve-Generate loop.
- **`bitsandbytes` Engine**: Handles the 4-bit memory mapping for the LLM.

## Design Patterns
- **Singleton-like Bot**: The `KLETechChatbot` instance maintains all model weights in memory during the session.
- **Context Normalization**: Prepending `[KLE Tech University Knowledge]` to facts to ensure high semantic anchoring.
