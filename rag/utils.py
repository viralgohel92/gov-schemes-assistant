import re
from rag.llm import UserProfile

NUM_MAP = {"one":1,"two":2,"three":3,"four":4,"five":5,
           "six":6,"seven":7,"eight":8,"nine":9,"ten":10}

MISSING = {"not available","","n/a","none","na","not found"}

def parse_limit(question: str):
    """
    Returns an explicit count only when the user clearly asks for N results.
    e.g. "give me 3 schemes", "show five schemes".
    Bare numbers in profile data (age 22, income 1.5 lakh) are ignored.
    """
    q = question.lower()
    QUANTITY_CONTEXT = r'(?:give|show|list|find|get|top|first|send|fetch|tell)\s+(?:me\s+)?'
    SCHEME_CONTEXT   = r'\s+(?:scheme|yojana|result|plan)s?'
    for word, num in NUM_MAP.items():
        if re.search(rf'{QUANTITY_CONTEXT}{word}', q) or re.search(rf'\b{word}{SCHEME_CONTEXT}', q):
            return num
    m = re.search(rf'{QUANTITY_CONTEXT}(\d+)', q)
    if m:
        return int(m.group(1))
    m = re.search(rf'(\d+){SCHEME_CONTEXT}', q)
    if m:
        return int(m.group(1))
    return None

def is_missing(val: str) -> bool:
    return not val or val.strip().lower() in MISSING

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def profile_to_text(profile: UserProfile) -> str:
    lines = [
        f"- Age: {getattr(profile, 'age', None)}" if getattr(profile, 'age', None) else "",
        f"- Annual Income: {getattr(profile, 'income', None)}" if getattr(profile, 'income', None) else "",
        f"- Occupation: {getattr(profile, 'occupation', None)}" if getattr(profile, 'occupation', None) else "",
        f"- State: {getattr(profile, 'state', None)}" if getattr(profile, 'state', None) else "",
        f"- Gender: {getattr(profile, 'gender', None)}" if getattr(profile, 'gender', None) else "",
        f"- Caste/Category: {getattr(profile, 'caste_category', None)}" if getattr(profile, 'caste_category', None) else "",
        f"- Other: {getattr(profile, 'extra', None)}" if getattr(profile, 'extra', None) else "",
    ]
    return "\n".join(l for l in lines if l)

def scheme_name_similarity(name_a: str, name_b: str) -> float:
    """
    Calculates similarity between two scheme names.
    Gives higher weight to unique identifying words and less to common ones like 'yojana'.
    """
    if not name_a or not name_b:
        return 0.0
        
    COMMON_WORDS = {"yojana", "yojna", "scheme", "schemes", "gujarat", "sahay", "mukhyamantri", "pradhan", "mantri", "bharat", "level"}
    
    def get_meaningful_words(name):
        return [w.lower() for w in re.findall(r'\b\w+\b', name) if len(w) > 2]

    words_a = get_meaningful_words(name_a)
    words_b = get_meaningful_words(name_b)
    
    if not words_a:
        return 0.0
        
    matches = 0
    total_weight = 0
    
    for word in words_a:
        weight = 0.5 if word in COMMON_WORDS else 1.0
        total_weight += weight
        if word in words_b:
            matches += weight
            
    return matches / total_weight if total_weight > 0 else 0.0

