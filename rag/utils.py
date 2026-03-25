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
        f"- Age: {profile.age}" if profile.age else "",
        f"- Annual Income: {profile.income}" if profile.income else "",
        f"- Occupation: {profile.occupation}" if profile.occupation else "",
        f"- State: {profile.state}" if profile.state else "",
        f"- Gender: {profile.gender}" if profile.gender else "",
        f"- Caste/Category: {profile.caste_category}" if profile.caste_category else "",
        f"- Other: {profile.extra}" if profile.extra else "",
    ]
    return "\n".join(l for l in lines if l)

def scheme_name_similarity(name_a: str, name_b: str) -> float:
    a_words = [w.lower() for w in name_a.split() if len(w) > 3]
    if not a_words:
        return 0.0
    b_lower = name_b.lower()
    return sum(1 for w in a_words if w in b_lower) / len(a_words)
