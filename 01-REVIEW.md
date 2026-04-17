# Phase 1: Code Review (RAG Architecture)

**Status:** Completed
**Depth:** Standard
**Files Reviewed:** `chat.py`, `embedder.py`, `generate_dataset.py`, `requirements.txt`

---

## 🛑 Critical Findings

### 1. Hardcoded Semester Filter (Robustness)
- **Location**: `chat.py:77`
- **Issue**: The `other_sems` list is hardcoded as `["Semester 4", "Semester 6", "Semester 5", "Semester 7"]`. 
- **Impact**: If the university adds a Semester 8 or changes the naming to just "Sem 4", the hybrid filter will fail to block incorrect data.
- **Recommendation**: Dynamically extract the semester names from the knowledge base or query a config file.

---

## ⚠️ Warning Findings

### 2. Lack of GPU Error Handling
- **Location**: `chat.py:45`
- **Issue**: `BitsAndBytesConfig` and `AutoModelForCausalLM` loading is perform in `__init__` without memory checks.
- **Impact**: On machines with limited VRAM (~4GB), the bot will crash with a cryptic "CUDA Out of Memory" error instead of suggesting `load_in_8bit` or `cpu` offloading.
- **Recommendation**: Wrap model loading in a `try...except` block and check available VRAM before loading.

### 3. Extraction Regex Fragility
- **Location**: `extract_facts.py` (and previous extraction logic)
- **Issue**: Facts are extracted using regex patterns looking for `answer: "..."`.
- **Impact**: If `generate_dataset.py` is updated with a different indentation or key name, the knowledge base will be empty or malformed.
- **Recommendation**: Use `json.loads` or a formal AST parser if the dataset becomes a JSON file.

---

## ℹ️ Info Findings

### 4. Greeting Fact Coverage
- **Location**: `college_data.jsonl`
- **Issue**: Only 4 generic greetings were added.
- **Impact**: Users asking "How are you?" or "Thanks" might still get the university office fallback.
- **Recommendation**: Expand the greeting fact set to include basic conversational turns.

---

## ▶ Next Steps

1.  **Auto-Fix**: Run `gsd-code-review-fix 1` to resolve the hardcoded semesters and add GPU error handling.
2.  **Deployment**: Verify the bot runs on a fresh environment.
