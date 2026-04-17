import torch
import numpy as np
import json
import os
import sys
import time
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"
THRESHOLD = 0.35
TOP_K = 5

# --- ANSI Color Codes ---
CLR_CYAN = "\033[1;36m"
CLR_GREEN = "\033[0;32m"
CLR_YELLOW = "\033[1;33m"
CLR_RESET = "\033[0m"

class KLETechChatbot:
    def __init__(self):
        print(f"{CLR_YELLOW}Loading SentenceTransformer model (CPU)...{CLR_RESET}")
        self.embed_model = SentenceTransformer(EMBED_MODEL_ID, device='cpu')
        
        print(f"{CLR_YELLOW}Loading Knowledge Base...{CLR_RESET}")
        if not os.path.exists('embeddings.npy') or not os.path.exists('facts.json'):
            print("Error: embeddings.npy or facts.json not found. Please run embedder.py first.")
            sys.exit(1)
            
        self.embeddings = np.load('embeddings.npy')
        with open('facts.json', 'r', encoding='utf-8') as f:
            self.facts = json.load(f)
            
        # DYNAMIC SEMESTER DETECTION  (limit to valid sem numbers 1-8 only)
        self.available_sems = sorted(list(set(
            s for s in re.findall(r'Semester \d+', " ".join(self.facts))
            if int(s.split()[1]) <= 8
        )))
        print(f"{CLR_YELLOW}Detected semesters: {', '.join(self.available_sems)}{CLR_RESET}")
            
        print(f"{CLR_YELLOW}Loading Qwen 2.5-0.5B-Instruct (4-bit GPU)...{CLR_RESET}")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.llm = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"\n{CLR_RESET}[!] ERROR: Could not load GPU model. Error: {e}")
            print(f"Possible fixes:\n1. Ensure you have a CUDA GPU.\n2. Try reducing VRAM usage.\n3. Run with CPU fallback (future update).")
            sys.exit(1)
        print(f"{CLR_GREEN}Chatbot Ready!{CLR_RESET}\n")

    def _normalize_query(self, query):
        """Expand shorthand in the query to match the normalized fact text format."""
        q = query
        # Expand semester shorthands
        for pat, repl in [
            (r'\b8th\s*sem\b', 'Semester 8'), (r'\b7th\s*sem\b', 'Semester 7'),
            (r'\b6th\s*sem\b', 'Semester 6'), (r'\b5th\s*sem\b', 'Semester 5'),
            (r'\b4th\s*sem\b', 'Semester 4'),
        ]:
            q = re.sub(pat, repl, q, flags=re.IGNORECASE)
        # Expand division shorthand: "div A" -> "division A"
        q = re.sub(r'\bdiv\b', 'division', q, flags=re.IGNORECASE)
        return q

    def retrieve(self, query):
        normalized_query = self._normalize_query(query)
        query_emb = self.embed_model.encode([normalized_query])
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_emb.T).flatten() / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # HYBRID FILTER: Detect explicit semester mentions
        q_lower = query.lower()
        required_sem = None
        
        # Step 1: Check word-form semester mentions.
        # Use \b word boundaries on roman numerals to avoid matching 'vi' inside 'division' etc.
        sem_map = [
            ("eighth", "8"), ("seventh", "7"), ("sixth", "6"), ("fifth", "5"), ("fourth", "4"),
            ("8th sem", "8"), ("7th sem", "7"), ("6th sem", "6"), ("5th sem", "5"), ("4th sem", "4"),
            ("sem 8", "8"), ("sem 7", "7"), ("sem 6", "6"), ("sem 5", "5"), ("sem 4", "4"),
            ("semester 8", "8"), ("semester 7", "7"), ("semester 6", "6"), ("semester 5", "5"), ("semester 4", "4"),
        ]
        for phrase, num in sem_map:
            if phrase in q_lower:
                required_sem = f"Semester {num}"
                break
        
        # Step 2: Word-boundary roman numeral check (must be a whole word, not inside 'division')
        if not required_sem:
            roman_map = [("viii", "8"), ("vii", "7"), ("vi", "6"), ("iv", "4")]
            for roman, num in roman_map:
                if re.search(r'\b' + roman + r'\b', q_lower):
                    required_sem = f"Semester {num}"
                    break
        
        # Step 3: Fallback — look for patterns like "4th" or "6th" standalone
        if not required_sem:
            match = re.search(r'\b([4-8])(?:th|st|nd|rd)?\s*(?:sem|semester)\b', q_lower)
            if match:
                required_sem = f"Semester {match.group(1)}"
            
        # HYBRID FILTER: Detect day of week
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
        required_day = None
        for day in days:
            if day in q_lower:
                required_day = day.capitalize()
                break

        # HYBRID FILTER: Detect division letter (A-F)
        required_div = None
        div_match = re.search(r'\b(?:div(?:ision)?\s*([a-f])|([a-f])\s*div(?:ision)?)\b', q_lower)
        if div_match:
            required_div = (div_match.group(1) or div_match.group(2)).upper()

        # Get top indices
        best_indices = np.argsort(similarities)[-30:][::-1]  # Look at top 30 for filtering
        
        filtered_facts = []
        for i in best_indices:
            fact_text = self.facts[i]
            
            # SEMESTER FILTER: Skip facts from OTHER semesters
            if required_sem and required_sem in self.available_sems:
                if any(s in fact_text for s in self.available_sems if s != required_sem) and required_sem not in fact_text:
                    continue
            
            # DAY FILTER: Skip timetable facts from OTHER days
            if required_day and "[ACADEMIC]" in fact_text:
                if not fact_text.startswith(f"[KLE Tech University Knowledge] [ACADEMIC]: {required_day}"):
                    continue
            
            # DIVISION FILTER: Skip timetable facts from OTHER divisions
            if required_div and "[ACADEMIC]" in fact_text:
                # Fact format: [ACADEMIC]: Monday (Semester 6 D): ...
                # Match the division letter inside the parentheses
                div_in_fact = re.search(r'\((?:Semester \d+ )?([A-F])\)', fact_text)
                if div_in_fact and div_in_fact.group(1) != required_div:
                    continue
            
            filtered_facts.append(fact_text)
            if len(filtered_facts) >= TOP_K:
                break
        
        if not filtered_facts or similarities[best_indices[0]] < THRESHOLD:
            return None
        
        return filtered_facts

    def _format_timetable(self, facts, query):
        """Format timetable facts exclusively to avoid messiness."""
        academic_facts = [f for f in facts if "[ACADEMIC]" in f]
        
        # If we have specific schedule facts, only show those to keep it clean
        if academic_facts:
            lines = []
            for f in academic_facts:
                # Extract schedule part: Monday (Semester 6 D): ...
                match = re.search(r'\[ACADEMIC\]:\s*.+?\):(.+)', f)
                if match:
                    schedule = match.group(1).strip().rstrip('.')
                    # Split by comma ONLY if not inside parentheses
                    subjects = []
                    current = []
                    depth = 0
                    for char in schedule:
                        if char == '(': depth += 1
                        elif char == ')': depth -= 1
                        if char == ',' and depth == 0:
                            subjects.append("".join(current).strip())
                            current = []
                        else:
                            current.append(char)
                    if current:
                        subjects.append("".join(current).strip())
                    lines.extend([s for s in subjects if s])
                elif "on Monday/Tuesday/Thursday" in f or "Div" in f:
                    # Keep general room info only if no specific schedule was found
                    if not lines:
                         lines.append(re.sub(r'\[KLE Tech University Knowledge\]\s*\[ACADEMIC\]:\s*', '', f).strip())
            return "\n".join(list(dict.fromkeys(lines))) # Unique lines only

        # Fallback to general cleaning if no academic facts found
        lines = []
        for f in facts:
            clean = re.sub(r'\[KLE Tech University Knowledge\]\s*(\[[A-Z]+\]:)?\s*', '', f).strip()
            lines.append(clean)
        return "\n".join(list(dict.fromkeys(lines)))

    def generate(self, query, facts):
        # For timetable queries, bypass LLM entirely — format directly for accuracy
        q_lower = query.lower()
        timetable_keywords = ["timetable", "schedule", "class", "monday", "tuesday",
                              "wednesday", "thursday", "friday", "saturday"]
        if any(k in q_lower for k in timetable_keywords):
            return self._format_timetable(facts, query)

        # For general queries (placement, holidays, greetings) use Qwen
        context = "\n".join([f"- {f}" for f in facts])
        prompt = f"""You are a helpful assistant for KLE Tech University.
Answer using ONLY the facts provided below. 
STRICT RULES:
1. Be extremely concise. Max 1-2 sentences.
2. Direct answers only. No filler like "Based on the info..." or "Here is the answer...".
3. Use a friendly but professional tone.

Facts:
{context}

Question: {query}
Answer:"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the answer — split on last occurrence of "Answer:"
        split_marker = "Answer:"
        if split_marker in full_text:
            return full_text.split(split_marker)[-1].strip()
        return full_text.strip()

    def chat(self):
        while True:
            try:
                user_input = input(f"{CLR_CYAN}You:{CLR_RESET} ").strip()
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                if not user_input:
                    continue
                
                facts = self.retrieve(user_input)
                
                if facts is None:
                    print(f"{CLR_GREEN}Bot:{CLR_RESET} Please contact the university office\n")
                else:
                    response = self.generate(user_input, facts)
                    print(f"{CLR_GREEN}Bot:{CLR_RESET} {response}\n")
                    
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    bot = KLETechChatbot()
    bot.chat()
