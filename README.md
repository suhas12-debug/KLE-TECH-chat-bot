# KLE Tech University Chatbot (Hybrid Semantic RAG)

A high-precision, custom AI assistant designed for **KLE Technological University**. This bot uses a **Hybrid Retrieval-Augmented Generation (RAG)** architecture that guarantees 100% factual accuracy for structured academic data while utilizing **Qwen2.5-1.5B-Instruct** for conversational queries.

---

## 🏗️ System Architecture: The "Bypass" Pipeline

Traditional LLMs often hallucinate dates, fees, and schedules. This system solves that using a **Deterministic Bypass Architecture**. 

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        HYBRID INFERENCE PIPELINE                        │
│                                                                         │
│                            [ User Query ]                               │
│                                  │                                      │
│                                  ▼                                      │
│                      [ Tag & Intent Detector ]                          │
│                                  │                                      │
│              ┌───────────────────┴───────────────────┐                  │
│     Contains Factual Tag                       No Factual Tag           │
│   (e.g., 'fee', 'timetable')                   (General Query)          │
│              │                                       │                  │
│              ▼                                       ▼                  │
│ [ Deterministic Bypass ]                [ Vector Search Retriever ]     │
│  (Regex & Keyword Lock)                 (SBERT + Cosine Similarity)     │
│              │                                       │                  │
│              ▼                                       ▼                  │
│ [ Hard Filter: Day/Div/Sem ]               [ Retrieve Top 80 Facts ]    │
│              │                                       │                  │
│              ▼                                       ▼                  │
│    [ Format Direct Fact ]                  [ Prompt Context Builder ]   │
│              │                                       │                  │
│              │                                       ▼                  │
│              │                          [ Qwen2.5-1.5B-Instruct LLM ]   │
│              │                                       │                  │
│              │ 1. Direct Truth                       │ 2. AI Generated  │
│              └───────────────────┬───────────────────┘                  │
│                                  ▼                                      │
│                       [ Final Formatted Answer ]                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Recent Hardening & Accuracy Features

### 1. Robust Timetable Algorithm
The bot now implements an **Implicit Intent Detection** algorithm. 
- **Typo Tolerance:** Specifically handles common typos like `"timeteble"` and `"time table"`.
- **Implicit Lock:** If a **Day** (e.g., Monday) and a **Division** (e.g., D) appear in the same sentence, the bot automatically locks the search to the Academic Timetable dataset, even if the word "timetable" isn't used.

### 2. Deep Vector Search
To prevent schedule facts from being "crowded out" by similar-looking calendar events (like working day swaps), the search depth has been increased from **Top 30 to Top 80 candidates**. This ensures the specific schedule for your division is always found and processed.

### 3. Factual Integrity Tags
- **[FEE]**: Direct bypass for tuition and admission costs.
- **[PLACEMENT]**: Strict year-based filtering (2022-2024) to ensure no batch-data leakage.
- **[ACADEMIC]**: Regex-locked weekly schedules.
- **[CALENDAR]**: University holidays and event dates.

---

## 🛠️ Technical Stack

- **Embedding Engine**: `SentenceTransformer` (all-MiniLM-L6-v2) - Runs on **CPU** for stability.
- **Language Model**: `Qwen2.5-1.5B-Instruct` - Runs on **GPU** (4-bit NF4 quantization).
- **Inference Library**: `transformers`, `bitsandbytes`, `accelerate`.
- **Logic**: Hybrid Python engine with strict Regex post-processing.

---

## 💻 System Requirements

Because the bot runs a 1.5B parameter model locally, it has specific resource needs:
- **VRAM**: Minimum 2GB (Runs comfortably on RTX 3050 Laptop).
- **System RAM**: Minimum 8GB (Needs at least **2GB of free Physical RAM** to start the loading sequence).
- **Storage**: ~5GB for model weights and embeddings.

> [!IMPORTANT]
> If the bot fails to start silently, ensure you have closed heavy background apps like Chrome or Edge to free up System RAM.

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `chat.py` | The main engine. Handles retrieval, filtering, and the chat loop. |
| `generate_dataset.py` | The raw knowledge source (Timetables, Fees, etc.). |
| `extract_facts.py` | Normalizes raw data into a searchable JSON format. |
| `embedder.py` | Rebuilds the mathematical vector space for the bot. |

---

## 📖 How to Update & Run

1. **Update Data**: Add facts to `generate_dataset.py`.
2. **Rebuild DB**: 
   ```bash
   python scratch/extract_facts.py
   python embedder.py
   ```
3. **Start Bot**: 
   ```bash
   python chat.py
   ```
