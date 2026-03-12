from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from pydantic import BaseModel, Field
from typing import List, Optional
import re, json

# -------------------------------------------------
# Language Support
# -------------------------------------------------

SUPPORTED_LANGUAGES = {"english": "en", "hindi": "hi", "gujarati": "gu"}

LANG_STRINGS = {
    "en": {
        "profile_request": """Sure! Please share your details so I can check eligibility:\n\n  \u2022 Age\n  \u2022 Annual Income  (e.g. 1.5 lakh, 50,000)\n  \u2022 Occupation     (e.g. student, farmer, self-employed)\n  \u2022 State          (e.g. Gujarat)\n  \u2022 Gender         (Male / Female)\n  \u2022 Caste/Category (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\nExample:\n  age: 22, income: 1.5 lakh, occupation: student, state: Gujarat, caste: OBC, Gender: Male""",
        "no_schemes_found": "No matching schemes found. Try providing more details like age, income, occupation, state, and caste/category.",
        "no_additional_schemes": "No additional matching schemes found for your profile.",
        "ask_schemes_first": "Please first ask for some schemes, then I can check your eligibility for them.",
    },
    "hi": {
        "profile_request": """ज़रूर! पात्रता जाँचने के लिए कृपया अपनी जानकारी दें:\n\n  \u2022 आयु (उम्र)\n  \u2022 वार्षिक आय  (जैसे 1.5 लाख, 50,000)\n  \u2022 पेशा        (जैसे छात्र, किसान, स्व-रोजगार)\n  \u2022 राज्य       (जैसे गुजरात)\n  \u2022 लिंग        (पुरुष / महिला)\n  \u2022 जाति/श्रेणी (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\nउदाहरण:\n  आयु: 22, आय: 1.5 लाख, पेशा: छात्र, राज्य: गुजरात, जाति: OBC, लिंग: पुरुष""",
        "no_schemes_found": "कोई मिलती-जुलती योजना नहीं मिली। कृपया अपनी आयु, आय, पेशा, राज्य और जाति की जानकारी दें।",
        "no_additional_schemes": "आपके प्रोफ़ाइल के लिए कोई अतिरिक्त योजना नहीं मिली।",
        "ask_schemes_first": "कृपया पहले कोई योजना खोजें, फिर मैं उनके लिए आपकी पात्रता जाँच सकता हूँ।",
    },
    "gu": {
        "profile_request": """ચોકકસ! પાત્રતા તપાસવા માટે કૃપા કરીને આ માહિતી આપો:\n\n  \u2022 ઉંમર\n  \u2022 વાર્ષિક આવક  (દા.ત. 1.5 લાખ, 50,000)\n  \u2022 વ્યવસાય      (દા.ત. વિદ્યાર્થી, ખેડૂત, સ્વ-રોજગાર)\n  \u2022 રાજ્ય        (દા.ત. ગુજરાત)\n  \u2022 જાતિ         (પુરુષ / સ્ત્રી)\n  \u2022 જ્ણાતિ/વર્ગ  (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\nઉદાહરણ:\n  ઉંમર: 22, આવક: 1.5 લાખ, વ્યવસાય: વિદ્યાર્થી, રાજ્ય: ગુજરાત, જ્ણાતિ: OBC, જાતિ: પુરુષ""",
        "no_schemes_found": "કોઈ મળતી યોજના મળી નહીં. કૃપા કરીને ઉંમર, આવક, વ્યવસાય, રાજ્ય અને જ્ણાતિ આપો.",
        "no_additional_schemes": "તમારી પ્રોફાઇલ માટે કોઈ વધારાની યોજના મળી નહીં.",
        "ask_schemes_first": "કૃપા કરીને પહેલાં કોઈ યોજના શોધો, પછી હું તમારી પાત્રતા તપાસીશ.",
    },
}

def detect_language(text: str) -> str:
    """Detect language: gu, hi, or en using Unicode ranges then romanized hints."""
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gu"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    lower = text.lower()
    gu_hints = ["kem cho", "tamne", "mane", "yojana", "aavak", "ummar", "vyvsa", "khedu", "rajya", "patrata"]
    hi_hints = ["mujhe", "kaunsi", "kya hai", "kaun si", "kaise", "bataiye", "sarkari", "patrata", "labh"]
    if any(w in lower for w in gu_hints):
        return "gu"
    if any(w in lower for w in hi_hints):
        return "hi"
    return "en"

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate user input to English for internal processing."""
    if source_lang == "en":
        return text
    lang_name = {"hi": "Hindi", "gu": "Gujarati"}[source_lang]
    r = llm.invoke(
        f"Translate this {lang_name} text to English.\n"
        f"Keep unchanged: scheme names, SC/ST/OBC/EWS/SEBC, state names, numbers, rupee amounts.\n"
        f"Return ONLY the English translation.\n\n{lang_name}: {text}\n\nEnglish:"
    )
    return r.content.strip()

def translate_response(text: str, target_lang: str) -> str:
    """Translate agent response from English to target language."""
    if target_lang == "en":
        return text
    lang_name = {"hi": "Hindi", "gu": "Gujarati"}[target_lang]
    r = llm.invoke(
        f"Translate this English text to {lang_name}.\n"
        f"Keep unchanged: scheme names, official links, SC/ST/OBC/EWS/SEBC, state names, ₹ amounts, numbers.\n"
        f"Return ONLY the {lang_name} translation.\n\nEnglish: {text}\n\n{lang_name}:"
    )
    return r.content.strip()

def get_string(key: str, lang: str) -> str:
    """Get a static UI string in the given language."""
    return LANG_STRINGS.get(lang, LANG_STRINGS["en"]).get(key, LANG_STRINGS["en"].get(key, ""))

# -------------------------------------------------
# Schemas
# -------------------------------------------------

class SchemeOutput(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    description: str = Field(description="Brief description of the scheme")
    category: str = Field(description="Category of the scheme")
    benefits: str = Field(description="Benefits provided under the scheme")
    eligibility: str = Field(description="Eligibility criteria for the scheme")
    documents_required: str = Field(description="Documents required to apply")
    application_process: str = Field(description="Steps to apply for the scheme")
    state: str = Field(description="State where the scheme is applicable")
    official_link: str = Field(description="Official website or link for the scheme")

class SchemesListOutput(BaseModel):
    schemes: List[SchemeOutput] = Field(description="List of all government schemes found in the context")

class UserProfile(BaseModel):
    age: Optional[int] = Field(None)
    income: Optional[str] = Field(None)
    occupation: Optional[str] = Field(None)
    state: Optional[str] = Field(None)
    gender: Optional[str] = Field(None)
    caste_category: Optional[str] = Field(None, description="Social/caste category: SC, ST, OBC, General, EWS, NT, DNT, SEBC, Minority etc.")
    extra: Optional[str] = Field(None, description="Any other relevant info e.g. disability, BPL, marital status")

# -------------------------------------------------
# Embedding + Vector DB
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

# -------------------------------------------------
# LLMs
# -------------------------------------------------

llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2, streaming=False)
structured_llm = llm.with_structured_output(SchemesListOutput)
profile_llm = llm.with_structured_output(UserProfile)

# -------------------------------------------------
# Memory
# -------------------------------------------------

store: dict = {}

def get_session(session_id: str) -> dict:
    if session_id not in store:
        store[session_id] = {
            "history": [],
            "last_schemes": [],
            "last_limit": None,
            "user_profile": None,
            "awaiting_profile": False,
            "lang": "en",              # detected language for this session
        }
    return store[session_id]

def save_to_history(session_id: str, question: str, answer: str):
    s = get_session(session_id)
    s["history"].append(HumanMessage(content=question))
    s["history"].append(AIMessage(content=answer))

# -------------------------------------------------
# Helpers
# -------------------------------------------------

NUM_MAP = {"one":1,"two":2,"three":3,"four":4,"five":5,
           "six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def parse_limit(question: str):
    q = question.lower()
    for word, num in NUM_MAP.items():
        if word in q:
            return num
    m = re.search(r'\b(\d+)\b', q)
    return int(m.group(1)) if m else None

MISSING = {"not available","","n/a","none","na","not found"}

def is_missing(val: str) -> bool:
    return not val or val.strip().lower() in MISSING

def apply_visit_site_fallback(d: dict) -> dict:
    link = d.get("official_link", "")
    for f in ["description","benefits","eligibility","documents_required","application_process"]:
        if is_missing(d.get(f,"")):
            d[f] = f"Not available. 👉 Visit: {link}" if link and not is_missing(link) else "Not available."
    return d

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

# -------------------------------------------------
# Intent detection
# -------------------------------------------------

def is_direct_scheme_name_query(question: str) -> bool:
    """
    Returns True if the user typed a long scheme name directly (not a natural question).
    Heuristic: long text (>6 words), no question words, no profile keywords.
    """
    q = question.strip()
    words = q.split()
    if len(words) < 5:
        return False
    # If it looks like a natural language question/sentence, skip
    conversational_starters = ["give me", "show me", "tell me", "find me", "what is", "which", "how", "can i", "do i", "am i", "is there", "are there"]
    q_lower = q.lower()
    if any(q_lower.startswith(s) for s in conversational_starters):
        return False
    # If it contains strong profile keywords, it's profile input
    profile_keywords = ["age","income","salary","lakh","occupation","caste","obc","sc/st","ews"]
    if any(kw in q_lower for kw in profile_keywords):
        return False
    # Long title-case or ALL-CAPS-words heavy text → likely a scheme name
    capitalized_words = sum(1 for w in words if w[0].isupper())
    if capitalized_words >= len(words) * 0.5:
        return True
    # Long enough text with no verb-like question words is likely a scheme name
    question_words = ["eligible","eligib","qualify","apply","benefit","what","which","how","who","where","when","why","can","could","should","would","please","list","find","search"]
    if not any(qw in q_lower for qw in question_words) and len(words) >= 6:
        return True
    return False

def detect_intent(question: str, chat_history: list, awaiting_profile: bool) -> str:
    # Direct scheme name lookup — highest priority
    if is_direct_scheme_name_query(question):
        return "full_detail"

    # If we're waiting for profile and message has profile-like content → treat as profile
    if awaiting_profile:
        profile_keywords = ["age","income","occupation","student","farmer","salary","lakh",
                            "sc","st","obc","general","ews","sebc","male","female","gender",
                            "unemployed","bpl","disabled","nt","dnt","minority","caste","category"]
        q = question.lower()
        if sum(1 for kw in profile_keywords if kw in q) >= 1:
            return "eligibility_check"

    # Hard keyword pre-check — catch eligibility intent before LLM
    q = question.lower()
    eligibility_trigger_phrases = [
        "eligible for", "am i eligible", "which scheme can i", "qualify for",
        "schemes for me", "i can apply", "can i apply", "suitable for me",
        "match my profile", "based on my profile", "for my profile",
        "new schemes i am eligible", "other schemes i am eligible",
        "find eligible", "show eligible", "eligible schemes",
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
- eligibility_check     → user provides personal details (age, income, occupation, state, caste/category etc.) to find matching schemes
- eligibility_for_shown → user asks about eligibility for schemes, shown or new (e.g. "which one am I eligible for?", "give me new schemes I am eligible for", "find schemes I qualify for", "show me eligible schemes", "find other eligible schemes", "which schemes can I apply for")
- names_only            → user wants only a list of scheme names WITHOUT eligibility filtering
- full_detail           → user wants complete details of scheme(s)
- specific_field        → user wants one specific field (eligibility, benefits, link, documents, etc.)
- conversational        → greeting, thanks, clarification, or non-scheme question

Reply with ONLY the intent label, nothing else.
"""
    return llm.invoke(intent_prompt).content.strip().lower()

def detect_field(question: str) -> str:
    r = llm.invoke(f"""The user asked: "{question}"
Which ONE field? Reply ONLY with the field name from:
scheme_name, description, category, benefits, eligibility, documents_required, application_process, state, official_link""")
    return r.content.strip().lower()

def is_fresh_search_request(question: str) -> bool:
    """Detect when user explicitly wants NEW/DIFFERENT results, not cached ones."""
    q = question.lower()
    fresh_hints = [
        "other", "different", "new", "another", "more schemes",
        "else", "apart from", "besides", "instead", "find more",
        "search", "look for", "locally", "show me more", "any more",
        "give me more", "find other", "suggest other", "recommend other"
    ]
    return any(h in q for h in fresh_hints)

def extract_gender_from_question(question: str) -> str | None:
    """Detect gender hint from natural language e.g. 'women scheme', 'male farmer'."""
    q = question.lower()
    female_hints = ["woman", "women", "female", "girl", "mahila", "lady", "ladies"]
    male_hints   = ["man", "men", "male", "boy", "gents", "gentleman", "purush"]
    if any(h in q for h in female_hints):
        return "Female"
    if any(h in q for h in male_hints):
        return "Male"
    return None

def merge_gender_into_profile(profile: "UserProfile", gender: str | None) -> "UserProfile":
    """Return a copy of profile with gender filled in if not already set."""
    if gender and not profile.gender:
        return profile.model_copy(update={"gender": gender})
    return profile

def is_followup_on_previous(question: str, chat_history: list, last_schemes: list = None) -> bool:
    if not chat_history: return False
    # If user wants fresh/new results, never treat as followup
    if is_fresh_search_request(question): return False

    q = question.lower()

    # Ordinal / number hints → always a followup reference
    ordinal_hints = [
        "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
        "1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th",
        "that","it","this","above","those","same","previous","last","shown","these"
    ]
    if any(h in q for h in ordinal_hints):
        return True

    # Plain number reference like "give me detail of 3"
    if re.search(r'\b\d{1,2}\b', q):
        return True

    # Check if any significant word from question partially matches a scheme name
    if last_schemes:
        STOP = {"the","a","an","of","for","in","and","or","to","is","me","my",
                "give","show","tell","about","what","scheme","details","detail",
                "full","please","get","find","i","want","its","this","that"}
        q_words = [w for w in re.findall(r'\b\w+\b', q) if len(w) > 3 and w not in STOP]
        for s in last_schemes:
            name = (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name","")).lower()
            if any(w in name for w in q_words):
                return True

    return False

def rewrite_question(question: str, chat_history: list) -> str:
    if not chat_history: return question
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    r = llm.invoke(f"Rewrite as standalone search query.\n\nConversation:\n{history_text}\n\nFollow-up: {question}\n\nStandalone query:")
    return r.content.strip()

def resolve_scheme_reference(question: str, schemes: list) -> list:
    """
    Resolve which scheme(s) the user is referring to.
    Handles:
      - Ordinal words: first, second, third... tenth
      - Ordinal numbers: 1st, 2nd, 3rd... 10th
      - Plain numbers: "scheme 3", "number 5", "3rd one"
      - Partial name match: user types part of a scheme name
      - Works with both SchemeOutput objects and plain dicts
    Returns a list with the single matched scheme, or all schemes if no match found.
    """
    if not schemes:
        return schemes

    def get_name(s) -> str:
        return (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")).strip()

    q = question.lower().strip()

    # ── 1. Ordinal word / number match ──────────────────────────────────────
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
        if re.search(rf'\b{re.escape(word)}\b', q) and idx < len(schemes):
            return [schemes[idx]]

    # Plain digit: "scheme 3", "number 3", "3rd", "#3"
    m = re.search(r'\b(\d{1,2})\b', q)
    if m:
        idx = int(m.group(1)) - 1          # convert 1-based to 0-based
        if 0 <= idx < len(schemes):
            return [schemes[idx]]

    # ── 2. Partial name match ────────────────────────────────────────────────
    # Score each scheme name by how many words from the question appear in it
    def name_score(name: str) -> float:
        name_lower = name.lower()
        # Split question into significant words (length > 2, skip common filler)
        STOP = {"the","a","an","of","for","in","and","or","to","is","me","my",
                "give","show","tell","about","what","scheme","details","detail",
                "full","please","get","find","i","want","its","this","that"}
        q_words = [w for w in re.findall(r'\b\w+\b', q) if len(w) > 2 and w not in STOP]
        if not q_words:
            return 0.0
        matched = sum(1 for w in q_words if w in name_lower)
        return matched / len(q_words)

    scored = [(name_score(get_name(s)), s) for s in schemes]
    best_score, best_scheme = max(scored, key=lambda x: x[0])

    # Only return the match if at least 30% of significant question words matched
    if best_score >= 0.30:
        return [best_scheme]

    # No match found → return all schemes unchanged
    return schemes

# -------------------------------------------------
# Extraction system prompt
# -------------------------------------------------

EXTRACTION_SYSTEM = """You are an AI assistant for Indian government schemes.

Map document fields as follows:
  "Scheme name"         → scheme_name
  "Description"         → description
  "category"            → category
  "benefits"            → benefits
  "eligibility"         → eligibility
  "application_process" → application_process
  "required_documents"  → documents_required
  "state"               → state
  "Link"                → official_link

Rules: Extract EVERY scheme. Copy values EXACTLY. Only use 'Not Available' if truly absent."""

# -------------------------------------------------
# Fetch schemes from vector DB
# -------------------------------------------------

def extract_specific_scheme_name(question: str, last_schemes: list) -> str | None:
    """
    Check if user is asking about a specific scheme by name.
    First checks against previously shown schemes, then looks for long quoted/named phrases.
    """
    q_lower = question.lower()

    # Match against previously shown scheme names
    for scheme in last_schemes:
        name = scheme.scheme_name if hasattr(scheme, "scheme_name") else scheme.get("scheme_name", "")
        if name and name.lower() in q_lower:
            return name

    # Detect trigger phrases like "details of X", "about X", "tell me about X"
    import re as _re
    patterns = [
        r"(?:full details of|details of|tell me about|info(?:rmation)? (?:about|on)|explain|describe|what is)\s+(.+?)(?:\s*\??\s*$)",
        r"^(.{10,}?)\s+(?:scheme)?\s*(?:give me|show me|details|full details|information)",
    ]
    for pat in patterns:
        m = _re.search(pat, question, _re.IGNORECASE)
        if m:
            candidate = m.group(1).strip(" .?")
            if len(candidate) > 8:
                return candidate

    # If the whole question looks like a bare scheme name (from is_direct_scheme_name_query),
    # use the entire question as the scheme name to search for
    if is_direct_scheme_name_query(question):
        return question.strip()

    return None


def scheme_name_similarity(name_a: str, name_b: str) -> float:
    """Return fraction of significant words from name_a that appear in name_b."""
    a_words = [w.lower() for w in name_a.split() if len(w) > 3]
    if not a_words:
        return 0.0
    b_lower = name_b.lower()
    return sum(1 for w in a_words if w in b_lower) / len(a_words)


def fetch_schemes(question: str, chat_history: list, k: int = 5, last_schemes: list = None) -> List[SchemeOutput]:
    standalone = rewrite_question(question, chat_history)

    # Detect if user is asking about ONE specific scheme
    specific_name = extract_specific_scheme_name(question, last_schemes or [])

    if specific_name:
        # Search with higher k so we have better chance of finding the right doc
        docs = vector_db.as_retriever(search_kwargs={"k": 5}).invoke(specific_name)
    else:
        docs = vector_db.as_retriever(search_kwargs={"k": k}).invoke(standalone)

    context = format_docs(docs)

    system = EXTRACTION_SYSTEM
    if specific_name:
        system += f'\n\nCRITICAL: The user is asking about ONLY this ONE specific scheme: "{specific_name}". Extract ONLY that scheme and nothing else. If the exact scheme is not in the context, return the closest matching one.'

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("placeholder", "{chat_history}"),
        ("human", "Extract ALL schemes from context. Copy values exactly.\n\nContext:\n{context}\n\nQuestion: {question}")
    ])
    result: SchemesListOutput = (prompt | structured_llm).invoke({
        "context": context, "question": question, "chat_history": chat_history,
    })

    # If specific scheme requested, strictly return only the best-matching one
    if specific_name:
        if not result.schemes:
            return []
        # Score each extracted scheme against the requested name
        scored = sorted(
            result.schemes,
            key=lambda s: scheme_name_similarity(specific_name, s.scheme_name),
            reverse=True
        )
        best = scored[0]
        # Only return it if it's a reasonable match (at least 30% word overlap)
        if scheme_name_similarity(specific_name, best.scheme_name) >= 0.3:
            return [best]
        # Low confidence match — still return the single best rather than all
        return [best]

    return result.schemes

# -------------------------------------------------
# Extract user profile
# -------------------------------------------------

def normalize_income(income_str: str) -> str:
    """Convert shorthand income to clear rupee amount.
    e.g. '1.5L' → '₹1,50,000 (1.5 lakh per year)'
         '50k'  → '₹50,000 (50k per year)'
    """
    if not income_str:
        return income_str
    s = income_str.strip().lower().replace(",", "")
    lakh_match = re.search(r"(\d+\.?\d*)\s*(l|lakh|lac)", s)
    k_match    = re.search(r"(\d+\.?\d*)\s*k", s)
    plain_match = re.match(r"^\d+$", s.strip())
    if lakh_match:
        val = float(lakh_match.group(1))
        rupees = int(val * 100000)
        return f"\u20b9{rupees:,} ({val} lakh per year)"
    elif k_match:
        val = float(k_match.group(1))
        rupees = int(val * 1000)
        return f"\u20b9{rupees:,} ({val}k per year)"
    elif plain_match:
        rupees = int(s.strip())
        return f"\u20b9{rupees:,} per year"
    return income_str

def extract_user_profile(question: str) -> UserProfile:
    try:
        profile = profile_llm.invoke(f"""Extract user profile from this message for government scheme eligibility.
Message: "{question}"
Extract: age, income, occupation, state, gender, caste_category (SC/ST/OBC/General/EWS/SEBC/NT/DNT/Minority), extra info.
Leave null if not mentioned.""")
        # Normalize income so LLM can compare amounts correctly
        if profile.income:
            profile.income = normalize_income(profile.income)
        return profile
    except Exception:
        return UserProfile()

# -------------------------------------------------
# Check eligibility against SPECIFIC shown schemes
# -------------------------------------------------

# ── Python-based caste eligibility checker (no LLM) ──────────────────────

# User caste → all equivalent terms that match in eligibility text
CASTE_MATCH_MAP = {
    "SC":       [r"\bsc\b", r"scheduled caste", r"harijan", r"dalit"],
    "ST":       [r"\bst\b", r"scheduled tribe", r"adivasi", r"tribal"],
    "OBC":      [r"\bobc\b", r"\bsebc\b", r"\bebc\b", r"other backward",
                 r"educationally backward", r"socially and educationally backward",
                 r"backward class"],
    "SEBC":     [r"\bobc\b", r"\bsebc\b", r"\bebc\b", r"other backward",
                 r"educationally backward", r"socially and educationally backward",
                 r"backward class"],
    "EBC":      [r"\bobc\b", r"\bsebc\b", r"\bebc\b", r"educationally backward",
                 r"backward class"],
    "EWS":      [r"\bews\b", r"economically weaker", r"\bgeneral\b"],
    "General":  [r"\bgeneral\b", r"\bunreserved\b", r"non-reserved", r"non reserved",
                 r"\bopen\b"],
    "NT":       [r"\bnt\b", r"\bdnt\b", r"nomadic", r"de-notified"],
    "DNT":      [r"\bnt\b", r"\bdnt\b", r"nomadic", r"de-notified"],
    "Minority": [r"\bminority\b", r"\bmuslim\b", r"\bchristian\b", r"\bsikh\b",
                 r"\bjain\b", r"\bbuddhist\b"],
}

OPEN_TO_ALL_TERMS = [
    r"\bunreserved\b", r"non-reserved", r"non reserved", r"\bgeneral\b",
    r"open to all", r"all categories", r"all castes", r"all citizens",
    r"all residents", r"irrespective of caste", r"any category",
    r"all community", r"regardless of caste",
]

def _re_search(pattern: str, text: str) -> bool:
    """Case-insensitive regex search helper."""
    return bool(re.search(pattern, text, re.IGNORECASE))

def python_caste_check(eligibility_text: str, user_caste: str) -> tuple:
    """
    Pure Python caste eligibility check using word-boundary regex — no LLM.
    Returns (is_eligible: bool, reason: str)
    """
    if not eligibility_text or eligibility_text.strip().lower() in (
        "not available", "not found", "", "n/a"
    ):
        return True, "No caste restriction specified — open to all categories"

    text = eligibility_text

    # 1. Check if scheme is open to all
    for pattern in OPEN_TO_ALL_TERMS:
        if _re_search(pattern, text):
            return True, f"Scheme is open to all categories — {user_caste} eligible"

    # 2. Get match patterns for user's caste
    user_caste_clean = (user_caste or "General").strip()
    # Normalise common variations
    caste_key = user_caste_clean.upper()
    if caste_key in ("OBC", "SEBC", "EBC"):
        caste_key = caste_key  # keep as-is, all map correctly
    match_patterns = CASTE_MATCH_MAP.get(
        caste_key,
        [rf"\b{re.escape(user_caste_clean.lower())}\b"]
    )

    # 3. Check if user's caste pattern appears in eligibility
    for pattern in match_patterns:
        if _re_search(pattern, text):
            return True, f"{user_caste_clean} matches scheme eligibility criteria"

    # 4. Check if eligibility restricts to a DIFFERENT caste group
    all_other_patterns = []
    for caste, patterns in CASTE_MATCH_MAP.items():
        if caste.upper() != caste_key:
            all_other_patterns.extend(patterns)

    found_other = [p for p in all_other_patterns if _re_search(p, text)]
    if found_other:
        # Strip regex syntax for display
        display = re.sub(r'\\b|\\', '', found_other[0]).strip()
        return False, f"Scheme restricted to {display} category — {user_caste_clean} not eligible"

    # 5. No caste info found → assume open
    return True, "No specific caste restriction — assumed open to all"


GENDER_FEMALE_PATTERNS = [
    r"\bwomen\b", r"\bwoman\b", r"\bfemale\b", r"\bgirl\b",
    r"\bmahila\b", r"\blady\b", r"\bladies\b",
]
GENDER_MALE_PATTERNS = [
    r"\bmen\b", r"\bman\b", r"\bmale\b", r"\bboy\b",
    r"\bpurush\b", r"\bgents\b",
]

def python_gender_check(eligibility_text: str, user_gender: str | None) -> tuple:
    """
    Pure Python gender eligibility check.
    Returns (is_eligible: bool, reason: str)
    """
    if not user_gender:
        return True, "No gender specified — skipping gender check"

    if not eligibility_text or eligibility_text.strip().lower() in MISSING:
        return True, "No gender restriction in eligibility"

    text = eligibility_text
    gender_lower = user_gender.strip().lower()
    is_female = gender_lower in ("female", "woman", "women", "girl", "mahila")
    is_male   = gender_lower in ("male", "man", "men", "boy", "gents", "purush")

    has_female_restriction = any(_re_search(p, text) for p in GENDER_FEMALE_PATTERNS)
    has_male_restriction   = any(_re_search(p, text) for p in GENDER_MALE_PATTERNS)

    # Scheme has no gender restriction at all → open to all
    if not has_female_restriction and not has_male_restriction:
        return True, "No gender restriction — open to all"

    # Scheme is for women only
    if has_female_restriction and not has_male_restriction:
        if is_female:
            return True, "Scheme is for women — user is Female ✓"
        return False, "Scheme is for women only — user is not Female"

    # Scheme is for men only
    if has_male_restriction and not has_female_restriction:
        if is_male:
            return True, "Scheme is for men — user is Male ✓"
        return False, "Scheme is for men only — user is not Male"

    # Both genders mentioned → open to all
    return True, "Scheme is open to all genders"


def check_eligibility_for_schemes(profile: UserProfile, schemes: List[SchemeOutput]) -> List[dict]:
    """
    Check eligibility:
      Step 1 — Python caste check (word-boundary regex, 100% reliable)
      Step 2 — Python gender check (word-boundary regex, 100% reliable)
      Step 3 — LLM checks ONLY age / income / state / occupation
    """
    user_caste  = (profile.caste_category or "General").strip()
    user_gender = (profile.gender or "").strip() or None
    results     = []

    # Caste keyword patterns for stripping — use word boundaries to avoid "sc" in "assistance"
    CASTE_STRIP_PATTERNS = [
        r"\bcaste\b", r"\bcategory\b", r"\b(sc|st|obc|sebc|ebc|ews|nt|dnt)\b",
        r"unreserved", r"reserved", r"backward", r"scheduled",
        r"\bminority\b", r"nomadic", r"tribal",
    ]

    for scheme in schemes:
        # ── Step 1: Caste check ──────────────────────────────────────────────
        caste_ok, caste_reason = python_caste_check(scheme.eligibility, user_caste)
        if not caste_ok:
            results.append({
                "scheme_name": scheme.scheme_name,
                "state": scheme.state,
                "official_link": scheme.official_link,
                "is_eligible": False,
                "reason": caste_reason,
            })
            continue

        # ── Step 2: Gender check ─────────────────────────────────────────────
        gender_ok, gender_reason = python_gender_check(scheme.eligibility, user_gender)
        if not gender_ok:
            results.append({
                "scheme_name": scheme.scheme_name,
                "state": scheme.state,
                "official_link": scheme.official_link,
                "is_eligible": False,
                "reason": gender_reason,
            })
            continue

        # ── Step 3: LLM checks ONLY age / income / state / occupation ────────
        profile_text = profile_to_text(profile)

        # Strip caste AND gender lines from eligibility to avoid LLM re-checking them
        eligibility_lines = scheme.eligibility.split(".") if scheme.eligibility else []
        non_caste_gender_lines = []
        for line in eligibility_lines:
            has_caste = any(_re_search(p, line) for p in CASTE_STRIP_PATTERNS)
            has_gender = any(_re_search(p, line) for p in GENDER_FEMALE_PATTERNS + GENDER_MALE_PATTERNS)
            if not has_caste and not has_gender:
                non_caste_gender_lines.append(line)

        clean_eligibility = ". ".join(non_caste_gender_lines).strip() \
            or "No specific age/income/state restriction mentioned."

        prompt = f"""You are checking NON-CASTE, NON-GENDER eligibility criteria only.
Caste and gender have already been verified separately — DO NOT re-evaluate them.

USER PROFILE:
{profile_text}

SCHEME: {scheme.scheme_name}
STATE: {scheme.state}
ELIGIBILITY (caste & gender lines removed): {clean_eligibility}

Check ONLY: age limit, income limit, occupation/profession, state of residence.
If the scheme has no such restrictions, answer PASS.

Reply with ONLY one of:
PASS: <brief reason about age/income/state/occupation>
FAIL: <specific criterion that failed — age/income/state/occupation only>"""

        r = llm.invoke(prompt)
        answer = r.content.strip()

        if answer.upper().startswith("PASS"):
            detail = answer.split(":", 1)[-1].strip() if ":" in answer else "Meets all criteria"
            # Build a clean combined reason
            parts = []
            if user_gender and gender_reason and "No gender" not in gender_reason:
                parts.append(gender_reason)
            if caste_reason and "No specific" not in caste_reason and "assumed" not in caste_reason:
                parts.append(caste_reason)
            parts.append(detail)
            results.append({
                "scheme_name": scheme.scheme_name,
                "state": scheme.state,
                "official_link": scheme.official_link,
                "is_eligible": True,
                "reason": ". ".join(p for p in parts if p),
            })
        else:
            detail = answer.split(":", 1)[-1].strip() if ":" in answer else answer
            results.append({
                "scheme_name": scheme.scheme_name,
                "state": scheme.state,
                "official_link": scheme.official_link,
                "is_eligible": False,
                "reason": detail,
            })

    return results


# -------------------------------------------------
# Fetch eligible schemes from full DB
# -------------------------------------------------

def fetch_eligible_schemes(profile: UserProfile, k: int = 10) -> List[dict]:
    # Build a targeted search query that includes gender so vector DB returns relevant docs
    parts = []
    if profile.gender:
        parts.append(profile.gender.lower())   # "female" / "male" → pulls women/men schemes
    if profile.occupation:
        parts.append(profile.occupation)
    if profile.state:
        parts.append(profile.state)
    if profile.caste_category:
        parts.append(profile.caste_category)
    if profile.age:
        parts.append(f"age {profile.age}")
    if profile.income:
        parts.append(f"income {profile.income}")
    if not parts:
        parts = ["government scheme"]

    query = " ".join(parts) + " eligibility scheme"
    docs = vector_db.as_retriever(search_kwargs={"k": k}).invoke(query)
    context = format_docs(docs)
    profile_text = profile_to_text(profile)

    # Build gender instruction for LLM
    gender_rule = ""
    if profile.gender:
        g = profile.gender.strip().lower()
        is_female = g in ("female", "woman", "women", "girl", "mahila")
        is_male   = g in ("male", "man", "men", "boy")
        if is_female:
            gender_rule = """
GENDER RULES:
- Include schemes that are specifically for women/females/mahila.
- Include schemes open to all genders (no gender restriction).
- EXCLUDE schemes that are explicitly for men/males only."""
        elif is_male:
            gender_rule = """
GENDER RULES:
- Include schemes that are specifically for men/males.
- Include schemes open to all genders (no gender restriction).
- EXCLUDE schemes that are explicitly for women/females only."""
    else:
        gender_rule = "\nGENDER: No gender specified — include all schemes regardless of gender."

    r = llm.invoke(f"""You are an Indian government scheme eligibility expert.

CRITICAL CASTE/CATEGORY RULES:
1. "Unreserved", "General", "Open", "Non-reserved", "All categories" = OPEN TO EVERYONE. Do NOT reject SC/ST/OBC/EWS users.
2. "SC only" = only SC eligible. "ST only" = only ST eligible.
3. "OBC" or "SEBC" or "EBC" = OBC/SEBC users ARE eligible.
4. "Minority" = only minority community eligible.
5. No caste restriction mentioned = open to all categories.
6. NEVER reject an OBC/SC/ST applicant from a "General/Unreserved/Open" scheme.
{gender_rule}

USER PROFILE:
{profile_text}

AVAILABLE SCHEMES:
{context}

Return ONLY schemes where this user clearly qualifies based on ALL criteria (caste, gender, age, income, occupation, state).

Return ONLY this JSON array (no markdown, no extra text):
[
  {{
    "scheme_name": "...",
    "category": "...",
    "state": "...",
    "official_link": "...",
    "why_eligible": "Specific reason mentioning matched criteria (gender, caste, age etc.)"
  }}
]""")

    raw = re.sub(r"```json|```", "", r.content.strip()).strip()
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []

# -------------------------------------------------
# Conversational reply
# -------------------------------------------------

def conversational_reply(question: str, chat_history: list, lang: str = "en") -> str:
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    lang_instruction = {
        "hi": "Always reply in Hindi (Devanagari script).",
        "gu": "Always reply in Gujarati (Gujarati script).",
        "en": "Reply in English.",
    }.get(lang, "Reply in English.")
    r = llm.invoke(f"""You are a helpful Indian government scheme assistant.
{lang_instruction}
Help users find schemes and check eligibility.

Conversation:
{history_text}

User: {question}
AI:""")
    return r.content.strip()

# -------------------------------------------------
# Main ask function
# -------------------------------------------------

def ask_agent(question: str, session_id: str = "user_1"):
    session = get_session(session_id)
    chat_history = session["history"]
    awaiting_profile = session.get("awaiting_profile", False)

    # ── Language detection ───────────────────────────────────────────────────
    lang = detect_language(question)
    # Persist language — once user speaks Hindi/Gujarati keep it unless they switch
    if lang != "en" or session.get("lang", "en") == "en":
        session["lang"] = lang
    lang = session["lang"]

    # Translate question to English for all internal processing
    question_en = translate_to_english(question, lang)

    # Helper: translate a reply string back to user's language
    def reply_in_lang(text: str) -> str:
        return translate_response(text, lang)

    # Shortcut to get localised static strings
    def ls(key: str) -> str:
        return get_string(key, lang)

    PROFILE_REQUEST = ls("profile_request")

    awaiting_profile = session.get("awaiting_profile", False)
    intent = detect_intent(question_en, chat_history, awaiting_profile)

    # ── User provided their profile ─────────────────────────────────────────
    if intent == "eligibility_check":
        profile = extract_user_profile(question_en)
        gender_hint = extract_gender_from_question(question_en)
        profile = merge_gender_into_profile(profile, gender_hint)
        session["user_profile"] = profile

        if awaiting_profile and session.get("last_schemes"):
            session["awaiting_profile"] = False
            print(ls("searching_shown") if "searching_shown" in LANG_STRINGS["en"] else "🔍 Checking...")
            results = check_eligibility_for_schemes(profile, session["last_schemes"])
            save_to_history(session_id, question, f"Checked eligibility for {len(session['last_schemes'])} schemes.")
            return {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results, "lang": lang}

        session["awaiting_profile"] = False
        print("🔍 Checking eligibility across all schemes...")
        eligible = fetch_eligible_schemes(profile, k=10)

        if not eligible:
            reply = reply_in_lang(ls("no_schemes_found"))
            save_to_history(session_id, question, reply)
            return {"type": "conversational", "reply": reply, "lang": lang}

        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        session["last_schemes"] = eligible
        return {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}

    # ── User asks eligibility for previously shown schemes ───────────────────
    if intent == "eligibility_for_shown":
        last_schemes = session.get("last_schemes", [])
        gender_hint = extract_gender_from_question(question_en)

        if is_fresh_search_request(question_en):
            if not session.get("user_profile"):
                session["awaiting_profile"] = True
                save_to_history(session_id, question, PROFILE_REQUEST)
                return {"type": "conversational", "reply": PROFILE_REQUEST, "lang": lang}
            profile = merge_gender_into_profile(session["user_profile"], gender_hint)
            session["user_profile"] = profile
            print("🔍 Searching all schemes for your eligibility...")
            eligible = fetch_eligible_schemes(profile, k=10)
            if not eligible:
                reply = reply_in_lang(ls("no_additional_schemes"))
                save_to_history(session_id, question, reply)
                return {"type": "conversational", "reply": reply, "lang": lang}
            save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
            session["last_schemes"] = eligible
            return {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}

        if not last_schemes:
            reply = reply_in_lang(ls("ask_schemes_first"))
            save_to_history(session_id, question, reply)
            return {"type": "conversational", "reply": reply, "lang": lang}

        if session.get("user_profile"):
            profile = merge_gender_into_profile(session["user_profile"], gender_hint)
            session["user_profile"] = profile
            print("🔍 Checking eligibility for shown schemes...")
            results = check_eligibility_for_schemes(profile, last_schemes)
            save_to_history(session_id, question, f"Checked eligibility for {len(last_schemes)} schemes.")
            return {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results, "lang": lang}

        if gender_hint:
            session["user_profile"] = UserProfile(gender=gender_hint)
        session["awaiting_profile"] = True
        save_to_history(session_id, question, PROFILE_REQUEST)
        return {"type": "conversational", "reply": PROFILE_REQUEST, "lang": lang}

    # ── Normal scheme queries ────────────────────────────────────────────────
    session["awaiting_profile"] = False
    limit = parse_limit(question_en)
    followup = is_followup_on_previous(question_en, chat_history, session["last_schemes"])
    fresh = is_fresh_search_request(question_en)

    if followup and not fresh and session["last_schemes"]:
        schemes = resolve_scheme_reference(question_en, session["last_schemes"])
        converted = []
        for s in schemes:
            if isinstance(s, dict):
                fetched = fetch_schemes(s.get("scheme_name", ""), [], k=3, last_schemes=[])
                if fetched:
                    converted.append(fetched[0])
                else:
                    converted.append(SchemeOutput(
                        scheme_name=s.get("scheme_name", ""),
                        description="", category=s.get("category", ""),
                        benefits="", eligibility="",
                        documents_required="", application_process="",
                        state=s.get("state", ""),
                        official_link=s.get("official_link", "")
                    ))
            else:
                converted.append(s)
        schemes = converted
    else:
        prev_names = []
        if fresh:
            for s in session["last_schemes"]:
                name = s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")
                if name:
                    prev_names.append(name)
        fetch_k = max(limit or 3, 5) + len(prev_names)
        schemes = fetch_schemes(question_en, chat_history, k=fetch_k, last_schemes=session["last_schemes"])
        if fresh and prev_names:
            schemes = [s for s in schemes if s.scheme_name not in prev_names]
        session["last_schemes"] = schemes
        session["last_limit"] = limit

    if intent == "names_only":
        selected = schemes[:limit] if limit else schemes
        names_text = "\n".join(f"{i+1}. {s.scheme_name}" for i, s in enumerate(selected))
        reply = reply_in_lang(names_text)
        save_to_history(session_id, question, reply)
        return {"type": "names_only", "reply": reply, "lang": lang}

    if intent == "specific_field":
        field = detect_field(question_en)
        lines = [f"• {s.scheme_name}:\n  {apply_visit_site_fallback(s.model_dump()).get(field, 'Not Available')}"
                 for s in schemes]
        reply_en = "\n\n".join(lines)
        reply = reply_in_lang(reply_en)
        save_to_history(session_id, question, reply)
        return {"type": "specific_field", "field": field, "reply": reply, "lang": lang}

    if intent == "conversational":
        reply = conversational_reply(question, chat_history, lang)
        save_to_history(session_id, question, reply)
        return {"type": "conversational", "reply": reply, "lang": lang}

    # full_detail — scheme card fields stay in English (original DB language)
    # but we add lang so UI can label fields in the right language
    selected = schemes[:limit] if limit else schemes
    save_to_history(session_id, question, f"Showed details for: {', '.join(s.scheme_name for s in selected)}")
    return {
        "type": "full_detail",
        "schemes": [apply_visit_site_fallback(s.model_dump()) for s in selected],
        "lang": lang,
    }