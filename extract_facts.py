"""
extract_facts.py — Build Pipeline Step 2
==========================================
Converts the conversational Q&A dataset (kle_tech_dataset.jsonl) into
the granular factual database (college_data.jsonl) used by the RAG engine.

Full Build Chain:
  Step 1: generate_dataset.py  →  kle_tech_dataset.jsonl  (raw Q&A training pairs)
  Step 2: extract_facts.py     →  college_data.jsonl       (unique, normalized facts)
  Step 3: embedder.py          →  embeddings.npy, facts.json  (vector index)
  Step 4: chat.py              →  interactive chatbot

Usage:
  python extract_facts.py

If kle_tech_dataset.jsonl is missing, run generate_dataset.py first.
"""

import json
import re
import os

# ── File Paths ────────────────────────────────────────────────────────────────
INPUT_FILE  = "kle_tech_dataset.jsonl"
OUTPUT_FILE = "college_data.jsonl"

# Semantic anchor prefix added to every fact so the embedding model
# anchors them firmly in the KLE Tech knowledge domain.
PREFIX = "[KLE Tech University Knowledge]"

# ── Supplementary Facts ───────────────────────────────────────────────────────
# These facts are NOT present as unique assistant answers in kle_tech_dataset.jsonl
# but ARE required in the RAG database (e.g. individual [HOLIDAY] entries for
# per-holiday queries, and greeting/conversational starters).
SUPPLEMENTARY_FACTS = [
    # Individual holiday facts — tagged [HOLIDAY] for per-event queries
    f"{PREFIX} [HOLIDAY]: Chandramana Ugadi is on 19th March 2026.",
    f"{PREFIX} [HOLIDAY]: Compensatory Holiday is on 20th March 2026.",
    f"{PREFIX} [HOLIDAY]: Ramzan (Eid-ul-Fitr) is on 21st March 2026.",
    f"{PREFIX} [HOLIDAY]: Mahavir Jayanti is on 31st March 2026.",
    f"{PREFIX} [HOLIDAY]: Good Friday is on 3rd April 2026.",
    f"{PREFIX} [HOLIDAY]: Ambedkar Jayanti is on 14th April 2026.",
    f"{PREFIX} [HOLIDAY]: Basava Jayanti is on 20th April 2026.",
    f"{PREFIX} [HOLIDAY]: May Day is on 1st May 2026.",
    f"{PREFIX} [HOLIDAY]: Bakrid is on 28th May 2026.",
    # Conversational starters / greeting responses
    f"{PREFIX} Hello! I am the KLE Tech University assistant. How can I help you today?",
    f"{PREFIX} Hi there! I can provide info about KLE Tech timetables, minor exams, and placements.",
    f"{PREFIX} Greetings! I am a chatbot trained on KLE Tech University data.",
    f"{PREFIX} Who are you? I am an AI assistant for KLE Tech University students.",
    f"{PREFIX} What can you do? I can help you with class schedules, faculty info, and placement records.",
    f"{PREFIX} Thank you! You're welcome! Feel free to ask more about KLE Tech university.",
    f"{PREFIX} Thanks. No problem! I'm here to help with your university queries.",
    f"{PREFIX} Goodbye. Have a great day! Reach out if you need more info about KLE Tech.",
]

# ── Normalisation Rules ───────────────────────────────────────────────────────
# Convert Roman numeral division identifiers inside [ACADEMIC] timetable facts
# to word-form semester strings so the hybrid semester filter in chat.py works
# correctly and avoids "VI" being confused with the word "division".
# Order matters: longer patterns first (VIII before VI, VI before IV).
ROMAN_TO_SEM = [
    (r'\bVIII\b', 'Semester 8'),
    (r'\bVII\b',  'Semester 7'),
    (r'\bVI\b',   'Semester 6'),
    (r'\bIV\b',   'Semester 4'),
    (r'\bII\b',   'Semester 2'),
]


def normalize_fact(text: str) -> str:
    """Replace Roman numeral semester identifiers with word-form equivalents."""
    for pattern, replacement in ROMAN_TO_SEM:
        text = re.sub(pattern, replacement, text)
    return text


def extract_facts():
    # ── Validate Input ────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file '{INPUT_FILE}' not found.")
        print("Please run 'python generate_dataset.py' first to create it.")
        return

    print(f"Reading Q&A pairs from '{INPUT_FILE}'...")
    seen: set[str] = set()
    facts: list[dict] = []
    skipped = 0

    # ── Extract Unique Facts from Q&A Pairs ──────────────────────────────────
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed JSON on line {lineno}: {e}")
                skipped += 1
                continue

            raw_fact = entry.get("assistant", "").strip()
            if not raw_fact:
                skipped += 1
                continue

            # Normalize Roman numerals to word-form semester identifiers
            raw_fact = normalize_fact(raw_fact)

            prefixed = f"{PREFIX} {raw_fact}"
            if prefixed not in seen:
                seen.add(prefixed)
                facts.append({"text": prefixed})

    unique_count = len(facts)
    print(f"  [OK] Extracted {unique_count} unique facts from Q&A pairs "
          f"({skipped} lines skipped).")

    # ── Append Supplementary Facts ────────────────────────────────────────────
    added = 0
    for supp_fact in SUPPLEMENTARY_FACTS:
        if supp_fact not in seen:
            seen.add(supp_fact)
            facts.append({"text": supp_fact})
            added += 1

    print(f"  [OK] Added {added} supplementary facts "
          f"(individual holidays + greetings).")

    # ── Write Output ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + '\n')

    total = len(facts)
    print(f"\n[DONE] Wrote {total} total facts to '{OUTPUT_FILE}'.")
    print("   Next step: run 'python embedder.py' to rebuild the vector index.")


if __name__ == "__main__":
    extract_facts()
