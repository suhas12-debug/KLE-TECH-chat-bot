# Technology Stack

This document outlines the core technologies, frameworks, and dependencies used in the KLE Tech Chatbot.

## Core Language & Runtime
- **Language**: Python 3.10+
- **Environment**: Local GPU execution (requires `torch` with CUDA support)

## Frameworks & Libraries
- **Deep Learning**: `torch` (PyTorch)
- **Transformers**: `transformers` (Hugging Face)
- **Retrieval (SBERT)**: `sentence-transformers`
- **Optimization**: `bitsandbytes` (4-bit quantization for GPU VRAM efficiency)
- **Acceleration**: `accelerate`
- **Data Handling**: `numpy`

## AI Models
- **Generation Model**: `Qwen/Qwen2.5-0.5B-Instruct`
  - Quantization: 4-bit (NF4)
- **Embedding Model**: `all-MiniLM-L6-v2` (Running on CPU for parallelizability)

## Configuration
- **Retrieval Threshold**: 0.35
- **Top K retrieval**: 5
- **Quantization Config**: NF4 with double quantization and `bfloat16` compute.
