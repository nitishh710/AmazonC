import re
from difflib import SequenceMatcher

def safe_str(val):
    if val is None or str(val).lower() == 'nan':
        return ""
    return str(val).lower().strip()

def get_tokens(text):
    return set(re.findall(r'\b\w+\b', text))

def get_numbers(text):
    return set(re.findall(r'\b\d+\b', text))

def get_trigrams(text):
    text = text.replace(" ", "")
    if len(text) < 3:
        return set([text]) if text else set()
    return set([text[i:i+3] for i in range(len(text)-2)])

def jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def string_similarity(s1, s2):
    if not s1 and not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()

def extract_features(row1, row2):
    n1, n2 = safe_str(row1.get('business_name')), safe_str(row2.get('business_name'))
    a1, a2 = safe_str(row1.get('business_address')), safe_str(row2.get('business_address'))
    c1, c2 = safe_str(row1.get('country')), safe_str(row2.get('country'))

    # Name features
    name_exact = 1.0 if n1 and n2 and n1 == n2 else 0.0
    name_jacc = jaccard(get_tokens(n1), get_tokens(n2))
    name_sim = string_similarity(n1, n2)
    name_trigram = jaccard(get_trigrams(n1), get_trigrams(n2)) # NEW: Typo resilience

    # Address features
    addr_missing = 1.0 if not a1 or not a2 else 0.0 # NEW: Missingness flag
    addr_exact = 1.0 if a1 and a2 and a1 == a2 else 0.0
    addr_jacc = jaccard(get_tokens(a1), get_tokens(a2))
    addr_sim = string_similarity(a1, a2)
    addr_num_jacc = jaccard(get_numbers(a1), get_numbers(a2))

    # Country feature
    if not c1 or not c2:
        country_match = 0.5 
    else:
        country_match = 1.0 if c1 == c2 else 0.0

    return [
        name_exact, name_jacc, name_sim, name_trigram,
        addr_missing, addr_exact, addr_jacc, addr_sim, addr_num_jacc, 
        country_match
    ]