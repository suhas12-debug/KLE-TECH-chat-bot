import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import json
import random
import sys
import time

from vocab_builder import load_vocab, VocabBuilder, PAD_token, SOS_token, EOS_token, UNK_token
from transformer_model import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask

# --- ANSI Color Codes ---
CLR_BLUE = "\033[1;34m"
CLR_CYAN = "\033[1;36m"
CLR_GREEN = "\033[0;32m"
CLR_YELLOW = "\033[1;33m"
CLR_RESET = "\033[0m"

def typing_print(prefix, text):
    """Prints text with a typing effect."""
    sys.stdout.write(prefix)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01) # Speed of typing
    sys.stdout.write("\n\n")
    sys.stdout.flush()

def print_logo():
    logo = f"""
{CLR_BLUE}  _  _______      _____ _____  _____ _    _ 
 | |/ /  __ \    / ____|  __ \|_   _| |  | |
 | ' /| |__) |  | |  __| |__) | | | | |  | |
 |  < |  ___/   | | |_ |  _  /  | | | |  | |
 | . \| |       | |__| | | \ \ _| |_| |__| |
 |_|\_\_|        \_____|_|  \_\_____|\____/ {CLR_RESET}
    
 {CLR_YELLOW}>>> Custom Seq2Seq Transformer Chatbot{CLR_RESET}
    """
    print(logo)

# Hyperparameters must match the trained model
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 256
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Similarity threshold — if the user's question doesn't match any known
# question well enough, we return a fallback instead of a wrong answer.
SIMILARITY_THRESHOLD = 0.3

def load_known_questions(jsonl_path="kle_tech_dataset.jsonl"):
    """Load all user questions from the dataset for similarity matching."""
    questions = []
    if not os.path.exists(jsonl_path):
        return questions
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                q = data.get("user", "")
                if q:
                    questions.append(q.lower())
            except json.JSONDecodeError:
                continue
    return questions

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_matcher(known_questions):
    """Creates a TF-IDF matcher for the given known questions."""
    # We use a custom token pattern to treat digits (like '6th', '7th') as distinct important tokens
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words='english', ngram_range=(1, 2))
    if known_questions:
        tfidf_matrix = vectorizer.fit_transform(known_questions)
    else:
        tfidf_matrix = None
    return vectorizer, tfidf_matrix

def find_best_match(query, known_questions, vectorizer, tfidf_matrix):
    """Find the best matching known question using TF-IDF Cosine Similarity."""
    if not known_questions or tfidf_matrix is None:
        return 0.0, ""
        
    query_vec = vectorizer.transform([query.lower()])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = cosine_similarities.argmax()
    best_score = cosine_similarities[best_idx]
    
    return best_score, known_questions[best_idx]

def load_system(vocab_path="vocab.json", model_path="kle_tech_bot.pth"):
    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print("Please run `train.py` first to generate the vocabulary and model weights.")
        return None, None, []
        
    print("Loading vocabulary...")
    vocab = load_vocab(vocab_path)
    
    print("Initializing model...")
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        src_vocab_size=vocab.num_words,
        tgt_vocab_size=vocab.num_words,
        dim_feedforward=FFN_HID_DIM
    ).to(DEVICE)
    
    print("Loading trained weights...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    print("Loading known questions for matching...")
    known_questions = load_known_questions()
    vectorizer, tfidf_matrix = create_matcher(known_questions)
    print(f"  Loaded {len(known_questions)} known questions.\n")
    
    return vocab, model, known_questions, vectorizer, tfidf_matrix

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """Greedy decode: generate one token at a time, picking the most probable."""
    memory = model.encode(src, src_mask)
    memory = memory.to(DEVICE)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
                    
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        logits = model.generator(out[:, -1])
        
        _, next_word = torch.max(logits, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_token:
            break
    
    return ys

def format_response(text):
    """
    Post-process the raw token output to restore proper formatting:
    fix punctuation spacing, capitalize sentences, restore numbers like 46.38, etc.
    """
    # Fix spaces around punctuation
    for p in [".", ",", "!", "?", ")", ":"]:
        text = text.replace(f" {p}", p)
    text = text.replace("( ", "(")
    
    # Fix decimal numbers like "46 . 38" → "46.38" or "5 . 3" → "5.3"
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    
    # Fix comma-separated numbers like "1 , 25 , 000" → "1,25,000"
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
    
    # Capitalize the first letter of each sentence
    sentences = text.split('. ')
    sentences = [s.strip().capitalize() if s else s for s in sentences]
    text = '. '.join(sentences)
    
    # Capitalize known proper nouns and abbreviations
    proper_nouns = {
        'kle': 'KLE', 'tech': 'Tech', 'hubballi': 'Hubballi', 'karnataka': 'Karnataka',
        'vidyanagar': 'Vidyanagar', 'amazon': 'Amazon', 'google': 'Google',
        'india': 'India', 'aws': 'AWS', 'twilio': 'Twilio', 'bosch': 'Bosch',
        'tcs': 'TCS', 'deloitte': 'Deloitte', 'siemens': 'Siemens',
        'cognizant': 'Cognizant', 'accenture': 'Accenture', 'capgemini': 'Capgemini',
        'inr': 'INR', 'lpa': 'LPA', 'ece': 'ECE', 'cse': 'CSE',
        'bvbcet': 'BVBCET', 'kcet': 'KCET', 'comedk': 'COMEDK',
        'vlsi': 'VLSI', 'cmos': 'CMOS', 'arm': 'ARM', 'pse': 'PSE',
        'bba': 'BBA', 'bca': 'BCA', 'mba': 'MBA',
        'mercedes': 'Mercedes', 'benz': 'Benz', 'ppos': 'PPOs',
        'rtos': 'RTOS', 'cipe': 'CIPE', 'bgsw': 'BGSW',
        'prof': 'Prof', 'dr': 'Dr',
        'texas': 'Texas', 'instruments': 'Instruments',
        'tata': 'Tata', 'elxsi': 'Elxsi',
        'verilog': 'Verilog',
    }
    
    words = text.split()
    for i, word in enumerate(words):
        # Strip punctuation to check the core word
        clean = word.strip('.,!?():;')
        if clean.lower() in proper_nouns:
            replacement = proper_nouns[clean.lower()]
            words[i] = word.replace(clean, replacement)
    text = ' '.join(words)
    
    return text

def generate_response(model, vocab, question, max_len=150):
    """Generate a response from the transformer model."""
    src_indices = vocab.sentence_to_indices(question)
    src = torch.tensor([SOS_token] + src_indices + [EOS_token]).unsqueeze(1).to(DEVICE)
    src_mask = torch.zeros((src.shape[0], src.shape[0])).type(torch.bool).to(DEVICE)

    with torch.no_grad():
        tgt_tokens = greedy_decode(
            model, src, src_mask, max_len=max_len, start_symbol=SOS_token
        )
        tgt_tokens = tgt_tokens.flatten()
        
    generated_words = []
    for tok in tgt_tokens[1:]:  # Skip SOS
        tok = tok.item()
        if tok == EOS_token:
            break
        word = vocab.index2word.get(tok, "<UNK>")
        generated_words.append(word)
    
    raw_text = " ".join(generated_words)
    formatted = format_response(raw_text)
    
    return formatted

FALLBACK_RESPONSES = [
    "I'm sorry, I don't have specific information about that topic yet. I can help you with questions about KLE Tech's courses, placements, fees, and campus details.",
    "That's a great question, but I don't have data on that right now. Try asking me about KLE Tech's programs, placement statistics, or fee structure!",
    "I'm not sure about that one. My knowledge covers KLE Tech's academic programs, placement records, fee details, and campus info. Feel free to ask about any of those!",
]

def main():
    vocab, model, known_questions, vectorizer, tfidf_matrix = load_system()
    if model is None:
        return
        
    print_logo()
    print(f"{CLR_YELLOW}Initializing University Knowledge Grid...{CLR_RESET}")
    print(f"{CLR_GREEN}Bot Ready!{CLR_RESET} (Type '{CLR_YELLOW}quit{CLR_RESET}' or '{CLR_YELLOW}exit{CLR_RESET}' to stop)\n")
    
    while True:
        try:
            print(f"{CLR_CYAN}You:{CLR_RESET} ", end="")
            user_input = input()
            
            if user_input.lower().strip() in ["quit", "exit", "q"]:
                print(f"{CLR_BLUE}Bot:{CLR_RESET} Goodbye! 👋\n")
                break
                
            if not user_input.strip():
                continue
            
            # Step 1: Matching
            best_score, best_match = find_best_match(user_input, known_questions, vectorizer, tfidf_matrix)
            
            sys.stdout.write(f"{CLR_BLUE}Bot is thinking")
            for _ in range(3):
                time.sleep(0.2)
                sys.stdout.write(".")
                sys.stdout.flush()
            sys.stdout.write("\r" + " " * 20 + "\r") # Clear thinking line
            
            if best_score < SIMILARITY_THRESHOLD:
                typing_print(f"{CLR_BLUE}Bot:{CLR_RESET} ", random.choice(FALLBACK_RESPONSES))
            else:
                response = generate_response(model, vocab, best_match)
                typing_print(f"{CLR_BLUE}Bot:{CLR_RESET} ", response)
            
        except KeyboardInterrupt:
            print(f"\n{CLR_BLUE}Bot:{CLR_RESET} Session terminated. Goodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
