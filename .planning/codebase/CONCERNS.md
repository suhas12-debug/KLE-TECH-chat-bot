# Concerns

This document tracks technical debt, fragile areas, and security considerations for the KLE Tech Chatbot.

## Technical Debt
- **Extraction Logic**: The extraction of facts from `generate_dataset.py` relies on regex parsing of a Python file. This is fragile; if the dictionary structure changes, extraction may fail.
- **Hardcoded Semesters**: The Hybrid Filter in `chat.py` has hardcoded check for semesters 4, 5, 6, and 7. This should be generalized to handle arbitrary semester numbers.

## Fragile Areas
- **Roman Numeral Conversion**: The data migration depends on successfully converting `IV` to `Semester 4`. Any missed patterns (e.g., `iv` lowercase or `fourth`) might still cause retrieval hallucinations.
- **Dependency on Local GPU**: The current 4-bit `bitsandbytes` setup requires a CUDA-capable GPU. There is no automated fallback to CPU/OpenVINO if a GPU is missing.

## Security & Ethics
- **Data Privacy**: The knowledge base is extracted from public university schedules. No PII (Personally Identifiable Information) is currently stored.
- **Hallucination Risk**: Despite filtering, the LLM might still hallucinate if the retrieved facts are ambiguous.

## Scalability
- **CPU Retrieval**: Performing cosine similarity on 200 facts on CPU is extremely fast. However, if the knowledge base grows to >10,000 facts, we may need a dedicated vector database like FAISS or Chroma.
