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
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
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
            
        print(f"{CLR_YELLOW}Loading Qwen 2.5-1.5B-Instruct (4-bit GPU)...{CLR_RESET}")
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
        # BGE models perform better with an instruction prefix for the query
        instruction = "Represent this sentence for searching relevant passages: "
        query_to_embed = instruction + normalized_query
        query_emb = self.embed_model.encode([query_to_embed])
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

        # HYBRID FILTER: Detect month (for holidays)
        months = ["january", "february", "march", "april", "may", "june", 
                  "july", "august", "september", "october", "november", "december"]
        required_month = None
        for month in months:
            if month in q_lower:
                required_month = month.capitalize()
                break
        
        # HYBRID FILTER: Detect tags for factual queries
        tag_keywords = {
            "fee": "[FEE]", "fees": "[FEE]", "kcet": "[FEE]", "comedk": "[FEE]",
            "cost": "[FEE]", "tuition": "[FEE]",
            "placement": "[PLACEMENT]", "placed": "[PLACEMENT]", "package": "[PLACEMENT]",
            "recruiter": "[PLACEMENT]", "recruiters": "[PLACEMENT]", "hiring": "[PLACEMENT]",
            "companies": "[PLACEMENT]", "company": "[PLACEMENT]",
            "rank": "[RANKING]", "nirf": "[RANKING]", "ranking": "[RANKING]",
            "location": "[LOCATION]", "address": "[LOCATION]", "campus": "[LOCATION]",
            "minor exam": "[CALENDAR]", "minor 1": "[CALENDAR]", "minor 2": "[CALENDAR]",
            "registration": "[CALENDAR]", "counselling": "[CALENDAR]",
            "formative feedback": "[CALENDAR]", "summative feedback": "[CALENDAR]",
            "pleiades": "[CALENDAR]", "fest": "[CALENDAR]",
            "working days": "[CALENDAR]", "term commencement": "[CALENDAR]",
            "course drop": "[CALENDAR]", "course withdrawal": "[CALENDAR]",
            "attendance report": "[CALENDAR]", "class committee": "[CALENDAR]",
        }
        matched_tag = None
        for keyword, tag in tag_keywords.items():
            if keyword in q_lower:
                matched_tag = tag
                break

        # Get top indices
        best_indices = np.argsort(similarities)[-80:][::-1]  # Look at top 80 for filtering
        
        filtered_facts = []

        # YEAR-PRIORITY SCAN: For placement year queries, do a full-database keyword scan
        requested_year = None
        if matched_tag == "[PLACEMENT]":
            year_match = re.search(r'\b(20\d{2})\b', q_lower)
            if year_match:
                requested_year = year_match.group(1)
                known_years = ["2024", "2023", "2022"]
                
                if requested_year not in known_years:
                    # Year not in our database — return honest "no data" response
                    return ["NO_DATA_FOR_YEAR"]
                
                # Scan entire DB for this year + tag
                for f in self.facts:
                    if matched_tag in f and requested_year in f:
                        # Use regex to avoid "2023-2024" matching a search for just "2024"
                        # "In 2023" fact contains "2023-2024" which falsely matches "2024"
                        if requested_year == "2024":
                            if f.find("In 2023") != -1: continue
                        if requested_year == "2023":
                            # Only include facts that talk ABOUT 2023, not just mention it in passing
                            if "In 2023" not in f and "2023 batch" not in f.lower(): continue
                        if f not in filtered_facts:
                            filtered_facts.append(f)
                
                # If year-scan found facts, return them directly (skip vector search)
                if filtered_facts:
                    return filtered_facts
        
        # TAG-PRIORITY SCAN: If a factual tag is detected, scan the full DB for matching facts
        # Return ALL matching tagged facts — generate() will sub-filter to the specific one needed
        if matched_tag and not filtered_facts:
            tag_facts = [f for f in self.facts if matched_tag in f]
            if tag_facts:
                return tag_facts  # Return ALL, not TOP_K — sub-filter in generate() will narrow
        
        # Normal vector-search loop for non-year queries
        for i in best_indices:
            if len(filtered_facts) >= TOP_K: break
            
            fact_text = self.facts[i]
            
            # Skip duplicates from year-scan above
            if fact_text in filtered_facts:
                continue
            
            # SEMESTER FILTER: Skip facts from OTHER semesters
            if required_sem and required_sem in self.available_sems:
                if any(s in fact_text for s in self.available_sems if s != required_sem) and required_sem not in fact_text:
                    continue
            
            # TIMETABLE FILTER: If they ask for a timetable, skip non-academic facts
            is_timetable_query = any(k in q_lower for k in ["timetable", "timeteble", "time table", "schedule", "class"])
            
            # If the user mentions both a day and a division, it's definitely a timetable query
            if required_day and required_div:
                is_timetable_query = True
                
            if is_timetable_query and "[ACADEMIC]" not in fact_text:
                continue
                
            # DAY FILTER: Skip timetable facts from OTHER days
            if required_day and "[ACADEMIC]" in fact_text:
                if not fact_text.startswith(f"[KLE Tech University Knowledge] [ACADEMIC]: {required_day}"):
                    continue
            
            # DIVISION FILTER: Skip timetable facts from OTHER divisions
            if required_div and "[ACADEMIC]" in fact_text:
                div_in_fact = re.search(r'\((?:Semester \d+ )?([A-F])\)', fact_text)
                if div_in_fact and div_in_fact.group(1) != required_div:
                    continue
            
            # MONTH FILTER: If a month is asked, skip holiday facts that don't match that month
            if required_month and ("[CALENDAR]" in fact_text or "[HOLIDAY]" in fact_text or "holiday" in fact_text.lower()):
                if required_month not in fact_text:
                    continue
            
            # TOTAL HOLIDAYS FILTER: If they ask for ALL holidays, prioritize the summary fact
            if "list all" in q_lower and "holiday" in q_lower and not required_month:
                if "9 holidays" not in fact_text and "All holidays:" not in fact_text:
                    if "[HOLIDAY]" in fact_text: continue
            
            filtered_facts.append(fact_text)
            if len(filtered_facts) >= TOP_K:
                break
        
        if not filtered_facts or similarities[best_indices[0]] < THRESHOLD:
            return None
        
        return filtered_facts


    def _format_direct(self, facts):
        """Format timetable and calendar facts directly to ensure accuracy and brevity."""
        lines = []
        academic_facts = [f for f in facts if "[ACADEMIC]" in f]
        calendar_facts = [f for f in facts if "[CALENDAR]" in f or "[HOLIDAY]" in f]

        if academic_facts:
            for f in academic_facts:
                match = re.search(r'\[ACADEMIC\]:\s*.+?\):(.+)', f)
                if match:
                    schedule = match.group(1).strip().rstrip('.')
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
                else:
                    clean = re.sub(r'\[KLE Tech University Knowledge\]\s*\[ACADEMIC\]:\s*', '', f).strip()
                    lines.append(clean)
        
        elif calendar_facts:
            for f in calendar_facts:
                clean = re.sub(r'\[KLE Tech University Knowledge\]\s*(\[[A-Z]+\]:)?\s*', '', f).strip()
                lines.append(clean)
        
        # Unique lines only, preserving order
        seen = set()
        return "\n".join([x for x in lines if not (x in seen or seen.add(x))])

    def generate(self, query, facts):
        # Direct bypass for Timetables, Holidays, and Calendars
        q_lower = query.lower()
        direct_keywords = ["timetable", "schedule", "class", "monday", "tuesday", "wednesday", 
                           "thursday", "friday", "saturday", "holiday", "calendar", "vacation"]
        
        if any(k in q_lower for k in direct_keywords):
            return self._format_direct(facts)

        # DIRECT BYPASS for ALL factual queries — Qwen is too small to handle them accurately
        # This covers: fees, placements, location, ranking, calendar events
        tag_keywords = {
            "fee": "[FEE]", "fees": "[FEE]", "kcet": "[FEE]", "comedk": "[FEE]", 
            "cost": "[FEE]", "tuition": "[FEE]",
            "placement": "[PLACEMENT]", "placed": "[PLACEMENT]", "package": "[PLACEMENT]",
            "recruiter": "[PLACEMENT]", "recruiters": "[PLACEMENT]", "hiring": "[PLACEMENT]",
            "placement officer": "[PLACEMENT]", "placement cell": "[PLACEMENT]",
            "rank": "[RANKING]", "nirf": "[RANKING]", "ranking": "[RANKING]",
            "location": "[LOCATION]", "address": "[LOCATION]", "campus": "[LOCATION]",
            "reach": "[LOCATION]", "airport": "[LOCATION]",
            "minor exam": "[CALENDAR]", "minor 1": "[CALENDAR]", "minor 2": "[CALENDAR]",
            "registration": "[CALENDAR]", "counselling": "[CALENDAR]",
            "formative feedback": "[CALENDAR]", "summative feedback": "[CALENDAR]",
            "pleiades": "[CALENDAR]", "fest": "[CALENDAR]",
            "working days": "[CALENDAR]", "term commencement": "[CALENDAR]",
            "course drop": "[CALENDAR]", "course withdrawal": "[CALENDAR]",
            "attendance report": "[CALENDAR]", "class committee": "[CALENDAR]",
        }
        
        matched_tag = None
        for keyword, tag in tag_keywords.items():
            if keyword in q_lower:
                matched_tag = tag
                break
        
        if matched_tag:
            tagged_facts = [f for f in facts if matched_tag in f]
            if tagged_facts:
                # SUB-FILTERING: Narrow results based on specific keywords
                if matched_tag == "[FEE]":
                    if "kcet" in q_lower or "government" in q_lower or "cet" in q_lower:
                        specific = [f for f in tagged_facts if "KCET" in f or "Government" in f]
                        if specific: tagged_facts = specific
                    elif "comedk" in q_lower or "management" in q_lower:
                        specific = [f for f in tagged_facts if "COMEDK" in f]
                        if specific: tagged_facts = specific
                
                elif matched_tag == "[PLACEMENT]":
                    # Year-based filtering for placement queries
                    found_year = False
                    for year in ["2024", "2023", "2022"]:
                        if year in q_lower:
                            # Strict match: Fact must contain the year AND preferably start with it
                            specific = [f for f in tagged_facts if year in f]
                            # Avoid "2023 fact stealing 2024"
                            if year == "2024":
                                specific = [f for f in specific if not f.startswith("In 2023")]
                            if year == "2023":
                                specific = [f for f in specific if f.startswith("In 2023")]
                            
                            if specific: 
                                tagged_facts = specific
                                found_year = True
                            break
                    
                    if not found_year:
                        # Keyword-based narrowing only if no year was requested
                        if "officer" in q_lower or "contact" in q_lower or "head" in q_lower:
                            specific = [f for f in tagged_facts if "Placement Cell" in f or "Placement Officer" in f or "headed" in f]
                            if specific: tagged_facts = specific
                        elif "recruiter" in q_lower or "companies" in q_lower or "company" in q_lower:
                            specific = [f for f in tagged_facts if "Major recruiters" in f]
                            if specific: tagged_facts = specific
                        elif "cell" in q_lower or "help" in q_lower or "resume" in q_lower or "workshop" in q_lower:
                            specific = [f for f in tagged_facts if "Placement Cell facilitates" in f]
                            if specific: tagged_facts = specific
                        elif "average" in q_lower or "median" in q_lower:
                            specific = [f for f in tagged_facts if "average" in f.lower() or "median" in f.lower()]
                            if specific: tagged_facts = specific
                        elif "highest" in q_lower:
                            specific = [f for f in tagged_facts if "highest package ever" in f.lower()]
                            if specific: tagged_facts = specific
                        else:
                            tagged_facts = tagged_facts[:2]
                
                elif matched_tag == "[CALENDAR]":
                    # Keyword-based narrowing for calendar queries
                    if "minor 2" in q_lower or "minor-2" in q_lower or "second minor" in q_lower:
                        specific = [f for f in tagged_facts if "Minor-2" in f]
                        if specific: tagged_facts = specific
                    elif "minor 1" in q_lower or "minor-1" in q_lower or "first minor" in q_lower:
                        specific = [f for f in tagged_facts if "Minor-1" in f]
                        if specific: tagged_facts = specific
                    elif "minor" in q_lower and "make" in q_lower:
                        specific = [f for f in tagged_facts if "Make-Up" in f]
                        if specific: tagged_facts = specific
                    elif "minor" in q_lower:
                        specific = [f for f in tagged_facts if "Minor-" in f]
                        if specific: tagged_facts = specific
                    elif "registration" in q_lower or "register" in q_lower:
                        specific = [f for f in tagged_facts if "Registration" in f or "registration" in f]
                        if specific: tagged_facts = specific
                    elif "counselling" in q_lower or "counseling" in q_lower:
                        specific = [f for f in tagged_facts if "Counselling" in f]
                        if specific: tagged_facts = specific
                    elif "pleiades" in q_lower or "fest" in q_lower:
                        specific = [f for f in tagged_facts if "Pleiades" in f]
                        if specific: tagged_facts = specific
                    elif "formative" in q_lower:
                        specific = [f for f in tagged_facts if "Formative" in f]
                        if specific: tagged_facts = specific
                    elif "summative" in q_lower:
                        specific = [f for f in tagged_facts if "Summative" in f]
                        if specific: tagged_facts = specific
                    elif "feedback" in q_lower:
                        specific = [f for f in tagged_facts if "Feedback" in f]
                        if specific: tagged_facts = specific
                    elif "attendance report" in q_lower or "attendance" in q_lower:
                        specific = [f for f in tagged_facts if "attendance report" in f.lower()]
                        if specific: tagged_facts = specific
                    elif "class committee" in q_lower or "ccm" in q_lower:
                        specific = [f for f in tagged_facts if "Class Committee" in f]
                        if specific: tagged_facts = specific
                    elif "drop" in q_lower:
                        specific = [f for f in tagged_facts if "dropping" in f]
                        if specific: tagged_facts = specific
                    elif "withdrawal" in q_lower or "withdraw" in q_lower:
                        specific = [f for f in tagged_facts if "withdrawal" in f]
                        if specific: tagged_facts = specific
                    elif "working day" in q_lower or "how many days" in q_lower:
                        specific = [f for f in tagged_facts if "working days" in f.lower() or "90" in f]
                        if specific: tagged_facts = specific
                    elif "semester start" in q_lower or "term start" in q_lower or "when does semester" in q_lower:
                        specific = [f for f in tagged_facts if "commences" in f]
                        if specific: tagged_facts = specific
                    elif "semester end" in q_lower or "term end" in q_lower:
                        specific = [f for f in tagged_facts if "term ends" in f.lower()]
                        if specific: tagged_facts = specific
                    else:
                        # Generic calendar — show first 3 most relevant
                        tagged_facts = tagged_facts[:3]
                
                # Clean tags and return directly
                lines = []
                for f in tagged_facts:
                    clean = re.sub(r'\[KLE Tech University Knowledge\]\s*(\[[A-Z]+\]:)?\s*', '', f).strip()
                    lines.append(clean)
                return "\n".join(lines)


        # For general queries, use Qwen with strict constraints
        context = "\n".join([f"- {f}" for f in facts])
        prompt = f"""You are a helpful assistant for KLE Tech University.
Answer using ONLY the facts below. Output ONLY the direct answer in 1-2 sentences. Do not explain.

Facts:
{context}

Question: {query}
Answer:"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the answer — split on last occurrence of "Answer:"
        split_marker = "Answer:"
        if split_marker in full_text:
            answer = full_text.split(split_marker)[-1].strip()
        else:
            answer = full_text.strip()
        
        # TRUNCATION: Cut off AI rambling (Explanation:, Note:, etc.)
        for stop_phrase in ["Explanation:", "Explanation of", "Note:", "I have listed", 
                            "This approach", "The final answer", "Based on this"]:
            if stop_phrase in answer:
                answer = answer[:answer.index(stop_phrase)].strip()
        
        # Also limit to first 3 sentences max
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if len(sentences) > 3:
            answer = ' '.join(sentences[:3])
        
        return answer.strip() if answer.strip() else "I don't have enough information to answer that."

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
                elif facts == ["NO_DATA_FOR_YEAR"]:
                    # Extract the year they asked about
                    year_match = re.search(r'\b(20\d{2})\b', user_input)
                    year_str = year_match.group(1) if year_match else "that year"
                    print(f"{CLR_GREEN}Bot:{CLR_RESET} Sorry, placement data for {year_str} is not available in our database. We currently have data for 2022, 2023, and 2024.\n")
                else:
                    response = self.generate(user_input, facts)
                    print(f"{CLR_GREEN}Bot:{CLR_RESET} {response}\n")
                    
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    bot = KLETechChatbot()
    bot.chat()
