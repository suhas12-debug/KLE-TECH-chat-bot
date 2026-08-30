# Technology Stack

This document outlines the core technologies, frameworks, and dependencies used in the KLE Tech Chatbot.
It is kept in sync with the active source code in `chat.py` and `embedder.py`.

## Core Language & Runtime
- **Language**: Python 3.10+
- **Environment**: Local GPU execution (requires `torch` with CUDA support)

## Frameworks & Libraries
- **Deep Learning**: `torch` (PyTorch)
- **Transformers**: `transformers` (Hugging Face)
- **Retrieval (SBERT)**: `sentence-transformers`
- **Optimization**: `bitsandbytes` (4-bit NF4 quantization for GPU VRAM efficiency)
- **Acceleration**: `accelerate`
- **Data Handling**: `numpy`

## AI Models (Current Live Configuration)
- **Generation Model**: `Qwen/Qwen2.5-1.5B-Instruct`
  - Quantization: 4-bit (NF4) with double quantization
  - Compute dtype: `bfloat16`
  - Device: GPU via `device_map="auto"`
- **Embedding Model (Bi-Encoder)**: `BAAI/bge-small-en-v1.5` (Running on CPU)
  - Query prefix: `"Represent this sentence for searching relevant passages: "`
- **Reranking Model (Cross-Encoder)**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Running on CPU)
  - Applied after Stage 1 retrieval to re-score the top 20 candidates

## Configuration
- **Retrieval Threshold**: 0.35 (below this, the "contact office" fallback is triggered)
- **Stage 1 Candidates**: Top 80 semantic matches considered before hard filtering
- **Stage 2 Final Top-K**: 5 facts sent to the LLM after reranking
- **Quantization Config**: NF4 with double quantization and `bfloat16` compute dtype

