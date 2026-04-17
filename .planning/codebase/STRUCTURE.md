# Structure

This document describes the directory organization and key file locations of the KLE Tech Chatbot project.

## Directory Layout
```text
/Chat_bot
├── chat.py             # Main entry point and Chatbot logic
├── embedder.py         # Script to generate vector embeddings
├── generate_dataset.py # Raw university knowledge source
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
├── .planning/          # GSD planning and codebase maps
│   └── codebase/       # Current codebase analysis docs
│
├── [DATA FILES]
├── college_data.jsonl  # Normalized factual database
├── facts.json          # Fact text lookup index
├── embeddings.npy      # Vector database (SBERT embeddings)
├── kle_tech_dataset.jsonl # Deprecated/Original dataset
│
├── diagnostic_sim.py   # Utility for testing retrieval similarity
└── .gitignore          # File exclusion rules
```

## Key Modules
- **`chat.py`**: Contains `KLETechChatbot` class. This is where the Hybrid Filter and LLM interaction reside.
- **`embedder.py`**: Uses `SentenceTransformer` to encode facts from `college_data.jsonl`.
- **`generate_dataset.py`**: A pure Python file with large dicts representing timetables and holiday schedules.
