import re
from langchain_core.messages import HumanMessage, AIMessage
from rag.llm import get_llm, UserProfile

def is_direct_scheme_name_query(question: str) -> bool:
    q = question.strip()
    words = q.split()
    if len(words) < 5:
        return False
    conversational_starters = ["give me", "show me", "tell me", "find me", "what is", "which", "how", "can i", "do i", "am i", "is there", "are there"]
    q_lower = q.lower()
    if any(q_lower.startswith(s) for s in conversational_starters):
        return False
    profile_keywords = ["age","income","salary","lakh","occupation","caste","obc","sc/st","ews"]
    if any(kw in q_lower for kw in profile_keywords):
        return False
    capitalized_words = sum(1 for w in words if w[0].isupper())
    if capitalized_words >= len(words) * 0.5:
        return True
    question_words = ["eligible","eligib","qualify","apply","benefit","what","which","how","who","where","when","why","can","could","should","would","please","list","find","search"]
    if not any(qw in q_lower for qw in question_words) and len(words) >= 6:
        return True
    return False

def detect_intent(question: str, chat_history: list, awaiting_profile: bool) -> str:
    q = question.lower().strip()

    # \u2500\u2500 PRIORITY 1: Greeting (before everything else) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    greeting_words = ["hello", "hi", "hey", "namaste", "namaskar", "kem cho",
                      "good morning", "good afternoon", "good evening", "greetings",
                      "helo", "hii", "haai", "jai shri krishna", "jai jinendra"]
    if any(re.search(rf'\b{g}\b', q) for g in greeting_words) and len(q.split()) <= 6:
        return "greeting"

    # \u2500\u2500 PRIORITY 2: Scheme count question (before everything else) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    count_hints = ["how many scheme", "total scheme", "number of scheme",
                   "kitni yojana", "ketli yojana", "how many yojana",
                   "count of scheme", "total yojana", "how many government scheme",
                   "kitne scheme", "scheme count", "schemes are there"]
    if any(h in q for h in count_hints):
        return "scheme_count"

    if is_direct_scheme_name_query(question):
        return "full_detail"

    words = q.split()
    if len(words) <= 5 and not awaiting_profile:
        # Prevent "tell me about EBC scholarship" from being incorrectly marked as names_only
        if not any(w in q for w in ["about", "detail", "details", "explain", "what"]):
            if any(w in q for w in ["scheme", "yojana", "scholarship", "loan", "plan", "program", "subsidy", "welfare"]):
                return "names_only"

    if awaiting_profile:
        profile_keywords = [
            "age","income","occupation","student","farmer","salary","lakh","rupee",
            "sc","st","obc","general","ews","sebc","male","female","gender","woman","man",
            "unemployed","bpl","disabled","nt","dnt","minority","caste","category",
            "year","old","gujarat","maharashtra","rajasthan","delhi","karnataka",
            "self","employed","business","service","retired","widow","married"
        ]
        q = question.lower()
        if any(kw in q for kw in profile_keywords):
            return "eligibility_check"
        if re.search(r'\b\d+\b', q):
            return "eligibility_check"

    q = question.lower()
    eligibility_trigger_phrases = [
        "eligible for", "am i eligible", "which scheme can i apply",
        "qualify for", "i can apply", "can i apply", "suitable for me",
        "match my profile", "based on my profile", "for my profile",
        "new schemes i am eligible", "other schemes i am eligible",
        "find eligible", "show eligible",
        "which scheme am i eligible", "what schemes am i eligible",
        "schemes i qualify", "i am eligible", "i qualify for",
        "check my eligibility", "check eligibility",
    ]
    if any(phrase in q for phrase in eligibility_trigger_phrases):
        return "eligibility_for_shown"

    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    intent_prompt = f"""
You are an intent classifier for a government scheme chatbot.

Conversation so far:
{history_text}

New user message: "{question}"

Classify the intent into EXACTLY one of these:

- names_only        → user wants a LIST of scheme names only. Examples: "housing scheme", "loan schemes", "show me farmer schemes", "what are education schemes", "list schemes for women", "agriculture schemes"
- full_detail       → user wants COMPLETE details of one or more specific schemes. Examples: "tell me about PM Awas Yojana", "full details of that scheme", "give me details of the first scheme"
- specific_field    → user wants ONE specific field like eligibility, benefits, link, or documents. Examples: "what are the benefits?", "what documents do I need?", "eligibility for this scheme"
- eligibility_check → user provides their PERSONAL DETAILS to find matching schemes. Examples: "age 22, income 1.5 lakh, OBC, Gujarat", "I am a student, 20 years old, female, SC category"
- eligibility_for_shown → user asks WHICH SCHEMES THEY ARE ELIGIBLE FOR (without providing new profile details). ONLY classify this when the user explicitly says things like "which scheme am I eligible for?", "am I eligible?", "which one can I apply for?", "check my eligibility"
- conversational    → greeting, thanks, unrelated question. Examples: "hello", "thank you", "what is this chatbot"

IMPORTANT RULES:
1. If user types a scheme CATEGORY or TOPIC like "housing scheme", "loan scheme", "farmer scheme" → classify as names_only (NOT eligibility_for_shown)
2. Only use eligibility_for_shown when the user EXPLICITLY asks about their own eligibility
3. eligibility_check requires the user to share personal data (age, income, caste, occupation etc.)

Reply with ONLY the intent label, nothing else.
"""
    return get_llm().invoke(intent_prompt).content.strip().lower()

def detect_field(question: str) -> str:
    r = get_llm().invoke(f"""The user asked: "{question}"
Which ONE field? Reply ONLY with the field name from:
scheme_name, description, category, benefits, eligibility, documents_required, application_process, state, official_link""")
    return r.content.strip().lower()

def is_fresh_search_request(question: str) -> bool:
    q = question.lower()
    strong_fresh_hints = [
        "other", "different", "another", "more schemes", "find more",
        "else", "apart from", "besides", "instead", "show me more",
        "any more", "give me more", "find other", "suggest other",
        "recommend other", "new schemes",
    ]
    if any(h in q for h in strong_fresh_hints):
        return True
    if re.search(r'\b(search|look for)\b', q) and re.search(r'\b(more|other|different|another|additional)\b', q):
        return True
    return False

def extract_gender_from_question(question: str) -> str | None:
    q = question.lower()
    female_hints = ["woman", "women", "female", "girl", "mahila", "lady", "ladies"]
    male_hints   = ["man", "men", "male", "boy", "gents", "gentleman", "purush"]
    if any(h in q for h in female_hints):
        return "Female"
    if any(h in q for h in male_hints):
        return "Male"
    return None

def merge_gender_into_profile(profile: UserProfile, gender: str | None) -> UserProfile:
    if gender and not profile.gender:
        return profile.model_copy(update={"gender": gender})
    return profile

def is_followup_on_previous(question: str, chat_history: list, last_schemes: list = None) -> bool:
    """
    Returns True ONLY when the user is clearly asking about a scheme that was
    already shown \u2014 NOT when they are searching for a completely new topic.

    Rule: a followup requires an EXPLICIT reference signal:
      1. An ordinal word/number ("first", "2nd", "give me details of 3")
      2. A short pronoun sentence ("tell me about it", "what is that") ≤ 5 words
      3. The user typed a significant portion of an EXACT previously shown scheme name

    Anything else = fresh search = return False.
    This prevents "disabled people schemes" from being treated as a followup
    to "farmer schemes" just because both contain the word "scheme".
    """
    if not chat_history:
        return False
    if is_fresh_search_request(question):
        return False

    q = question.lower().strip()

    # \u2500\u2500 Signal 1: explicit ordinal words \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ORDINALS = [
        "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
        "1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th",
    ]
    if any(re.search(rf'\b{o}\b', q) for o in ORDINALS):
        return True

    # Plain digit reference: "give me detail of 3" / "number 2"
    if re.search(r'\b(number|no\.?|#)\s*\d{1,2}\b', q):
        return True

    # \u2500\u2500 Signal 2: short pronoun-only sentence \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    # "tell me about it" / "what is that" / "show me this"
    PRONOUNS = [r'\bit\b', r'\bthis one\b', r'\bthat one\b', r'\bthe above\b']
    if any(re.search(p, q) for p in PRONOUNS) and len(q.split()) <= 6:
        return True

    # \u2500\u2500 Signal 3: user typed a significant chunk of an exact scheme name \u2500\u2500\u2500\u2500\u2500
    # Require the question to contain ≥ 3 consecutive significant words from
    # a previously shown scheme name, OR ≥ 60% of the scheme's significant words.
    # This is strict enough to avoid false positives on generic words.
    if last_schemes:
        STOP = {
            "the","a","an","of","for","in","and","or","to","is","me","my","by","give","show","tell",
            "about","what","scheme","schemes","details","detail","full","please","get","find","i",
            "want","its","government","india","national","pradhan","mantri","yojana","rajya",
            "gujarat","welfare","under","from","with","also","only","more","any","all","new","this",
        }
        q_words = [w for w in re.findall(r'\b\w+\b', q) if len(w) > 2 and w not in STOP]
        if q_words:
            for s in last_schemes:
                name = (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")).lower()
                name_words = [w for w in re.findall(r'\b\w+\b', name) if len(w) > 2 and w not in STOP]
                if not name_words: continue
                matched = [w for w in q_words if w in name_words]
                # Match ≥ 50% of the scheme's significant name words OR if the name is an exact substring
                if (len(matched) / len(name_words) >= 0.5 and len(matched) >= 2) or name in q:
                    return True

    return False

def rewrite_question(question: str, chat_history: list) -> str:
    """
    Rewrites a follow-up question into a standalone search query.
    Only calls the LLM when there is a genuine back-reference signal.
    Fresh topic queries are returned as-is \u2014 no LLM, no context pollution.
    """
    if not chat_history:
        return question

    q = question.lower().strip()

    # If the question contains a clear back-reference signal, rewrite with history
    BACKREF_SIGNALS = [
        r'\bit\b', r'\bthis one\b', r'\bthat one\b', r'\bthe above\b',
        r'\bprevious\b', r'\bsame scheme\b',
        r'\b1st\b', r'\b2nd\b', r'\b3rd\b', r'\b4th\b', r'\b5th\b',
        r'\bfirst\b', r'\bsecond\b', r'\bthird\b',
    ]
    has_backref = any(re.search(p, q) for p in BACKREF_SIGNALS)

    if not has_backref:
        # No back-reference → fresh topic → return as-is
        return question

    # Has back-reference → rewrite using conversation history
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-4:]
    ])
    r = get_llm().invoke(
        f"Rewrite as standalone search query.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Follow-up: {question}\n\nStandalone query:"
    )
    return r.content.strip()

def resolve_scheme_reference(question: str, question_en: str, schemes: list) -> list:
    if not schemes:
        return schemes

    def get_name(s) -> str:
        return (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")).strip()

    q_lower = question_en.lower().strip()

    # 1. Exact or partial exact match (English matches)
    for s in schemes:
        name_lower = get_name(s).lower()
        if name_lower and (name_lower == q_lower or name_lower in q_lower or q_lower in name_lower):
            return [s]

    ORDINALS = {
        "first": 0,  "1st": 0,  "one": 0,
        "second": 1, "2nd": 1,  "two": 1,
        "third": 2,  "3rd": 2,  "three": 2,
        "fourth": 3, "4th": 3,  "four": 3,
        "fifth": 4,  "5th": 4,  "five": 4,
        "sixth": 5,  "6th": 5,  "six": 5,
        "seventh": 6,"7th": 6,  "seven": 6,
        "eighth": 7, "8th": 7,  "eight": 7,
        "ninth": 8,  "9th": 8,  "nine": 8,
        "tenth": 9,  "10th": 9, "ten": 9,
    }
    for word, idx in ORDINALS.items():
        if re.search(rf'\b{re.escape(word)}\b', q_lower) and idx < len(schemes):
            return [schemes[idx]]

    # Also check native question for digits (like Gujarati ૩)
    m = re.search(r'\b(\d{1,2})\b', question.lower())
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(schemes):
            return [schemes[idx]]
    
    m_en = re.search(r'\b(\d{1,2})\b', q_lower)
    if m_en:
        idx = int(m_en.group(1)) - 1
        if 0 <= idx < len(schemes):
            return [schemes[idx]]

    # 2. Robust LLM mapping fallback for cross-language/transliteration
    names_map = {f"{i+1}": get_name(s) for i, s in enumerate(schemes)}
    map_str = "\n".join(f"{k}. {v}" for k, v in names_map.items())
    
    prompt = f"""You are a matching system. Identify which exact scheme(s) from the provided list the user is referring to.

List of Schemes (English):
{map_str}

User Query (Native/Mix): "{question}"

Instructions:
- Identifying which scheme number(s) the user is interested in.
- Match translated Hindi/Gujarati names to English names.
- If one matches, return its number (e.g. 3).
- If multiple, return separated by comma (e.g. 1, 3).
- If NONE match, return 0.
- Reply ONLY with digits/commas, no other text."""

    try:
        raw = get_llm().invoke(prompt).content.strip()
        nums = [int(n.strip()) for n in raw.replace(".", ",").split(",") if n.strip().isdigit()]
        valid = [schemes[n-1] for n in nums if 1 <= n <= len(schemes)]
        if valid:
            return valid
    except Exception as e:
        print(f"[resolve_scheme] LLM fallback error: {e}")

    # 3. Fuzzy match fallback
    def name_score(name: str) -> float:
        name_lower = name.lower()
        STOP = {"the","a","an","of","for","in","and","or","to","is","me","my",
                "give","show","tell","about","what","scheme","details","detail",
                "full","please","get","find","i","want","its","this","that"}
        
        q_words = [w for w in re.findall(r'\b\w+\b', q_lower) 
                   if (len(w) > 2 or w.isdigit() or w in ("sc", "st", "nt")) and w not in STOP]
        if not q_words:
            return 0.0
        matched = sum(1 for w in q_words if w in name_lower)
        return matched / len(q_words)

    scored = [(name_score(get_name(s)), s) for s in schemes]
    best_score, best_scheme = max(scored, key=lambda x: x[0])

    if best_score >= 0.30:
        return [best_scheme]

    return None
