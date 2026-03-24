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
import re, json, time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        "no_additional_schemes": "તમારી પ્રોફાઇલ માટે કોઈ વધારાની યોજना મળી નહીં.",
        "ask_schemes_first": "કૃપા કરીને પહેલાં કોઈ યોજना શોધો, પછી હું તમારી પાત્રતા તપાસીશ.",
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
    r = get_llm().invoke(
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
    r = get_llm().invoke(
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
# ✅ LAZY LOADERS — models load only when first used
#    This removes 20-30s blocking delay at import time.
# -------------------------------------------------

_embedding_model = None
_vector_db = None
_llm = None
_structured_llm = None
_profile_llm = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("⏳ Loading embedding model...")
        _embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embedding model loaded.")
    return _embedding_model

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = Chroma(persist_directory="vector_db", embedding_function=get_embedding_model())
    return _vector_db

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2, streaming=False)
    return _llm

def get_structured_llm():
    global _structured_llm
    if _structured_llm is None:
        _structured_llm = get_llm().with_structured_output(SchemesListOutput)
    return _structured_llm

def get_profile_llm():
    global _profile_llm
    if _profile_llm is None:
        _profile_llm = get_llm().with_structured_output(UserProfile)
    return _profile_llm

def warmup():
    """
    Pre-load the embedding model and LLM so the first real request is fast.
    Call this from app.py in a background thread right after Flask starts.
    """
    print("🔥 Warming up models...")
    get_embedding_model()
    get_vector_db()
    get_llm()
    get_structured_llm()
    get_profile_llm()
    print("✅ Warmup complete — all models ready.")

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
            "lang": "en",
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

MISSING = {"not available","","n/a","none","na","not found"}

def is_missing(val: str) -> bool:
    return not val or val.strip().lower() in MISSING

def _fetch_page_text_requests(url: str, timeout: int = 8) -> str:
    """Try fetching page text with requests (works for static HTML sites)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r'<(script|style)[^>]*>.*?</(script|style)>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<[^>]+>', ' ', html)
        html = re.sub(r'&nbsp;', ' ', html)
        html = re.sub(r'&[a-zA-Z]+;', '', html)
        html = re.sub(r'\s{2,}', ' ', html)
        return html.strip()[:8000]
    except Exception as e:
        print(f"[requests_fetch] Failed: {e}")
        return ""


def _fetch_page_text_playwright(url: str, timeout: int = 20000) -> str:
    """Fetch JS-rendered page text using Playwright (for sites like myscheme.gov.in)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(4000)   # wait for JS to render content
            text = page.evaluate("document.body.innerText")
            browser.close()
            cleaned = re.sub(r'\s{2,}', ' ', text or "")
            return cleaned.strip()[:8000]
    except Exception as e:
        print(f"[playwright_fetch] Failed: {e}")
        return ""


def _fetch_page_text(url: str) -> str:
    """
    Try requests first (fast). If it returns too little content (JS-rendered site),
    fall back to Playwright automatically.
    """
    text = _fetch_page_text_requests(url)
    if len(text) < 300:
        print(f"[fetch] requests got too little content, trying Playwright for {url}")
        text = _fetch_page_text_playwright(url)
    return text


def enrich_scheme_from_web(url: str, scheme_name: str, missing_fields: list) -> dict:
    """
    Fetch the official scheme page and use the LLM to extract only the missing fields.
    Returns a dict of extracted values. Falls back gracefully on any error.
    """
    print(f"🌐 Fetching live data for '{scheme_name}' → {url}")
    page_text = _fetch_page_text(url)
    if not page_text or len(page_text) < 100:
        print(f"[enrich] Could not get usable content from {url}")
        return {}

    fields_desc = {
        "description":         "A brief overall description of what this scheme is about",
        "benefits":            "Benefits provided to beneficiaries (financial amount, subsidies, services, etc.)",
        "eligibility":         "Who is eligible — age, income, caste, occupation, state restrictions",
        "documents_required":  "List of documents needed to apply",
        "application_process": "Step-by-step instructions on how to apply",
    }
    fields_to_extract = {f: fields_desc[f] for f in missing_fields if f in fields_desc}
    fields_json = json.dumps(fields_to_extract, indent=2)

    prompt = f"""You are extracting information about the government scheme "{scheme_name}" from a webpage.

Webpage content:
\"\"\"
{page_text}
\"\"\"

Extract ONLY the following fields from the content above:
{fields_json}

Rules:
- Copy relevant text exactly as found on the page.
- For application_process, number each step clearly: "Step 1: ... Step 2: ..."
- For documents_required, list each document on a new line or separated by semicolons.
- If a field is not present in the content at all, set its value to "Not available".
- Do NOT invent or hallucinate any information.
- Reply ONLY with a valid JSON object using the exact field names as keys. No markdown, no explanation.

JSON:"""

    try:
        response = get_llm().invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        print(f"✅ Enriched {list(result.keys())} for '{scheme_name}'")
        return result
    except Exception as e:
        print(f"[enrich_scheme] Parse error for '{scheme_name}': {e}")
        return {}


def apply_visit_site_fallback(d: dict) -> dict:
    """
    For any missing fields, fetch live data from the official link.
    Falls back to a clickable link message if fetch fails or page is unreachable.
    """
    link = d.get("official_link", "")
    missing_fields = [
        f for f in ["description", "benefits", "eligibility", "documents_required", "application_process"]
        if is_missing(d.get(f, ""))
    ]
    if not missing_fields:
        return d  # all fields present, nothing to do

    if link and not is_missing(link):
        enriched = enrich_scheme_from_web(link, d.get("scheme_name", ""), missing_fields)
        for f in missing_fields:
            val = enriched.get(f, "")
            d[f] = val if val and not is_missing(val) else f"Not available. 👉 Visit: {link}"
    else:
        for f in missing_fields:
            d[f] = "Not available."
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

    # ── PRIORITY 1: Greeting (before everything else) ─────────────────────────
    greeting_words = ["hello", "hi", "hey", "namaste", "namaskar", "kem cho",
                      "good morning", "good afternoon", "good evening", "greetings",
                      "helo", "hii", "haai", "jai shri krishna", "jai jinendra"]
    import re
    if any(re.search(rf'\b{g}\b', q) for g in greeting_words) and len(q.split()) <= 6:
        return "greeting"

    # ── PRIORITY 2: Scheme count question (before everything else) ────────────
    count_hints = ["how many scheme", "total scheme", "number of scheme",
                   "kitni yojana", "ketli yojana", "how many yojana",
                   "count of scheme", "total yojana", "how many government scheme",
                   "kitne scheme", "scheme count", "schemes are there"]
    if any(h in q for h in count_hints):
        return "scheme_count"

    if is_direct_scheme_name_query(question):
        return "full_detail"

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

def merge_gender_into_profile(profile: "UserProfile", gender: str | None) -> "UserProfile":
    if gender and not profile.gender:
        return profile.model_copy(update={"gender": gender})
    return profile

def is_followup_on_previous(question: str, chat_history: list, last_schemes: list = None) -> bool:
    """
    Returns True ONLY when the user is clearly asking about a scheme that was
    already shown — NOT when they are searching for a completely new topic.

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

    # ── Signal 1: explicit ordinal words ─────────────────────────────────────
    ORDINALS = [
        "first","second","third","fourth","fifth","sixth","seventh","eighth","ninth","tenth",
        "1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th",
    ]
    if any(re.search(rf'\b{o}\b', q) for o in ORDINALS):
        return True

    # Plain digit reference: "give me detail of 3" / "number 2"
    if re.search(r'\b(number|no\.?|#)\s*\d{1,2}\b', q):
        return True

    # ── Signal 2: short pronoun-only sentence ────────────────────────────────
    # "tell me about it" / "what is that" / "show me this"
    PRONOUNS = [r'\bit\b', r'\bthis one\b', r'\bthat one\b', r'\bthe above\b']
    if any(re.search(p, q) for p in PRONOUNS) and len(q.split()) <= 6:
        return True

    # ── Signal 3: user typed a significant chunk of an exact scheme name ─────
    # Require the question to contain ≥ 3 consecutive significant words from
    # a previously shown scheme name, OR ≥ 60% of the scheme's significant words.
    # This is strict enough to avoid false positives on generic words.
    if last_schemes:
        STOP = {
            "the","a","an","of","for","in","and","or","to","is","me","my","by",
            "give","show","tell","about","what","scheme","schemes","details","detail",
            "full","please","get","find","i","want","its","government","india",
            "national","pradhan","mantri","yojana","rajya","gujarat","welfare",
            "under","from","with","also","only","more","any","all","new",
        }
        q_words = [w for w in re.findall(r'\b\w+\b', q) if len(w) > 3 and w not in STOP]
        if q_words:
            for s in last_schemes:
                name = (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")).lower()
                name_words = [w for w in re.findall(r'\b\w+\b', name) if len(w) > 3 and w not in STOP]
                if not name_words:
                    continue
                matched = [w for w in q_words if w in name_words]
                # Must match ≥ 60% of the scheme's significant name words
                if len(matched) / len(name_words) >= 0.6 and len(matched) >= 2:
                    return True

    return False

def rewrite_question(question: str, chat_history: list) -> str:
    """
    Rewrites a follow-up question into a standalone search query.
    Only calls the LLM when there is a genuine back-reference signal.
    Fresh topic queries are returned as-is — no LLM, no context pollution.
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

def resolve_scheme_reference(question: str, schemes: list) -> list:
    if not schemes:
        return schemes

    def get_name(s) -> str:
        return (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")).strip()

    q = question.lower().strip()

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

    m = re.search(r'\b(\d{1,2})\b', q)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(schemes):
            return [schemes[idx]]

    def name_score(name: str) -> float:
        name_lower = name.lower()
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

    if best_score >= 0.30:
        return [best_scheme]

    return schemes

# -------------------------------------------------
# Extraction system prompt
# -------------------------------------------------

EXTRACTION_SYSTEM = """You are an AI assistant for Gujarat government schemes.

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
    q_lower = question.lower()
    for scheme in last_schemes:
        name = scheme.scheme_name if hasattr(scheme, "scheme_name") else scheme.get("scheme_name", "")
        if name and name.lower() in q_lower:
            return name

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

    if is_direct_scheme_name_query(question):
        return question.strip()

    return None


def scheme_name_similarity(name_a: str, name_b: str) -> float:
    a_words = [w.lower() for w in name_a.split() if len(w) > 3]
    if not a_words:
        return 0.0
    b_lower = name_b.lower()
    return sum(1 for w in a_words if w in b_lower) / len(a_words)


def extract_search_topic(question: str) -> str:
    """Strip filler words, keep the core topic for vector DB search."""
    FILLER = {
        "show", "me", "give", "find", "get", "tell", "list", "fetch",
        "search", "look", "for", "please", "can", "you", "i", "want",
        "need", "a", "an", "the", "some", "any", "all", "new", "latest",
        "related", "about", "regarding", "based", "on", "my", "in", "of",
        "is", "are", "there", "do", "does", "what", "which", "how",
    }
    words = re.findall(r'\b\w+\b', question.lower())
    core = [w for w in words if w not in FILLER and len(w) > 2]
    return " ".join(core) if core else question


def fetch_schemes(question: str, chat_history: list, k: int = 5, last_schemes: list = None) -> List[SchemeOutput]:
    """
    Fetches schemes from vector DB.
    rewrite_question + DB search run in PARALLEL — saves ~6s when rewrite needs LLM.
    For fresh topic queries, rewrite returns instantly (no LLM), so no waiting.
    """
    specific_name = extract_specific_scheme_name(question, last_schemes or [])
    search_query  = specific_name if specific_name else extract_search_topic(question)

    # Run rewrite + initial DB search in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        rewrite_future = ex.submit(rewrite_question, question, chat_history)
        db_future      = ex.submit(
            lambda q=search_query, _k=5 if specific_name else k:
                get_vector_db().as_retriever(search_kwargs={"k": _k}).invoke(q)
        )

    standalone = rewrite_future.result()
    docs       = db_future.result()

    # If rewrite actually changed the query, re-run DB search with the better query
    if not specific_name and standalone.lower().strip() != question.lower().strip():
        docs = get_vector_db().as_retriever(search_kwargs={"k": k}).invoke(standalone)

    context = format_docs(docs)

    system = EXTRACTION_SYSTEM
    if specific_name:
        system += (
            f'\n\nCRITICAL: The user is asking about ONLY this ONE specific scheme: '
            f'"{specific_name}". Extract ONLY that scheme. If not in context, return closest match.'
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("placeholder", "{chat_history}"),
        ("human", "Extract ALL schemes from context. Copy values exactly.\n\nContext:\n{context}\n\nQuestion: {question}")
    ])
    result: SchemesListOutput = (prompt | get_structured_llm()).invoke({
        "context": context, "question": question, "chat_history": chat_history,
    })

    if specific_name:
        if not result.schemes:
            return []
        scored = sorted(
            result.schemes,
            key=lambda s: scheme_name_similarity(specific_name, s.scheme_name),
            reverse=True
        )
        best = scored[0]
        if scheme_name_similarity(specific_name, best.scheme_name) >= 0.3:
            return [best]
        return [best]

    return result.schemes

# -------------------------------------------------
# Extract user profile
# -------------------------------------------------

def normalize_income(income_str: str) -> str:
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
        profile = get_profile_llm().invoke(f"""Extract user profile from this message for government scheme eligibility.
Message: "{question}"
Extract: age, income, occupation, state, gender, caste_category (SC/ST/OBC/General/EWS/SEBC/NT/DNT/Minority), extra info.
Leave null if not mentioned.""")
        if profile.income:
            profile.income = normalize_income(profile.income)
        return profile
    except Exception:
        return UserProfile()

# -------------------------------------------------
# Check eligibility against SPECIFIC shown schemes
# -------------------------------------------------

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
    return bool(re.search(pattern, text, re.IGNORECASE))

def python_caste_check(eligibility_text: str, user_caste: str) -> tuple:
    if not eligibility_text or eligibility_text.strip().lower() in (
        "not available", "not found", "", "n/a"
    ):
        return True, "No caste restriction specified — open to all categories"

    text = eligibility_text

    for pattern in OPEN_TO_ALL_TERMS:
        if _re_search(pattern, text):
            return True, f"Scheme is open to all categories — {user_caste} eligible"

    user_caste_clean = (user_caste or "General").strip()
    caste_key = user_caste_clean.upper()
    if caste_key in ("OBC", "SEBC", "EBC"):
        caste_key = caste_key
    match_patterns = CASTE_MATCH_MAP.get(
        caste_key,
        [rf"\b{re.escape(user_caste_clean.lower())}\b"]
    )

    for pattern in match_patterns:
        if _re_search(pattern, text):
            return True, f"{user_caste_clean} matches scheme eligibility criteria"

    all_other_patterns = []
    for caste, patterns in CASTE_MATCH_MAP.items():
        if caste.upper() != caste_key:
            all_other_patterns.extend(patterns)

    found_other = [p for p in all_other_patterns if _re_search(p, text)]
    if found_other:
        display = re.sub(r'\\b|\\', '', found_other[0]).strip()
        return False, f"Scheme restricted to {display} category — {user_caste_clean} not eligible"

    return True, "No specific caste restriction — assumed open to all"


GENDER_FEMALE_PATTERNS = [
    r"\bwomen\b", r"\bwoman\b", r"\bfemale\b", r"\bgirl\b",
    r"\bmahila\b", r"\blady\b", r"\bladies\b", r"\bstree\b"
]
GENDER_MALE_PATTERNS = [
    r"\bmen\b", r"\bman\b", r"\bmale\b", r"\bboy\b",
    r"\bpurush\b", r"\bgents\b",
]

def python_gender_check(eligibility_text: str, user_gender: str | None) -> tuple:
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

    if not has_female_restriction and not has_male_restriction:
        return True, "No gender restriction — open to all"

    if has_female_restriction and not has_male_restriction:
        if is_female:
            return True, "Scheme is for women — user is Female ✓"
        return False, "Scheme is for women only — user is not Female"

    if has_male_restriction and not has_female_restriction:
        if is_male:
            return True, "Scheme is for men — user is Male ✓"
        return False, "Scheme is for men only — user is not Male"

    return True, "Scheme is open to all genders"


def check_eligibility_for_schemes(profile: UserProfile, schemes: List[SchemeOutput]) -> List[dict]:
    user_caste  = (profile.caste_category or "General").strip()
    user_gender = (profile.gender or "").strip() or None

    CASTE_STRIP_PATTERNS = [
        r"\bcaste\b", r"\bcategory\b", r"\b(sc|st|obc|sebc|ebc|ews|nt|dnt)\b",
        r"unreserved", r"reserved", r"backward", r"scheduled",
        r"\bminority\b", r"nomadic", r"tribal",
    ]

    needs_llm    = []
    fast_results = {}

    for scheme in schemes:
        caste_ok, caste_reason = python_caste_check(scheme.eligibility, user_caste)
        if not caste_ok:
            fast_results[scheme.scheme_name] = {
                "scheme_name": scheme.scheme_name, "category": scheme.category,
                "state": scheme.state, "official_link": scheme.official_link,
                "is_eligible": False, "reason": caste_reason,
            }
            continue

        gender_ok, gender_reason = python_gender_check(scheme.eligibility, user_gender)
        if not gender_ok:
            fast_results[scheme.scheme_name] = {
                "scheme_name": scheme.scheme_name, "category": scheme.category,
                "state": scheme.state, "official_link": scheme.official_link,
                "is_eligible": False, "reason": gender_reason,
            }
            continue

        needs_llm.append((scheme, caste_reason, gender_reason))

    profile_text = profile_to_text(profile)

    def _llm_check_one(args):
        scheme, caste_reason, gender_reason = args
        eligibility_lines = scheme.eligibility.split(".") if scheme.eligibility else []
        non_caste_gender_lines = []
        for line in eligibility_lines:
            has_caste  = any(_re_search(p, line) for p in CASTE_STRIP_PATTERNS)
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

        r = get_llm().invoke(prompt)
        answer = r.content.strip()

        if answer.upper().startswith("PASS"):
            detail = answer.split(":", 1)[-1].strip() if ":" in answer else "Meets all criteria"
            parts = []
            if user_gender and gender_reason and "No gender" not in gender_reason:
                parts.append(gender_reason)
            if caste_reason and "No specific" not in caste_reason and "assumed" not in caste_reason:
                parts.append(caste_reason)
            parts.append(detail)
            return scheme.scheme_name, {
                "scheme_name": scheme.scheme_name, "category": scheme.category,
                "state": scheme.state, "official_link": scheme.official_link,
                "is_eligible": True,
                "reason": ". ".join(p for p in parts if p),
            }
        else:
            detail = answer.split(":", 1)[-1].strip() if ":" in answer else answer
            return scheme.scheme_name, {
                "scheme_name": scheme.scheme_name, "category": scheme.category,
                "state": scheme.state, "official_link": scheme.official_link,
                "is_eligible": False, "reason": detail,
            }

    llm_results = {}
    if needs_llm:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_llm_check_one, args): args for args in needs_llm}
            for future in as_completed(futures):
                try:
                    name, result = future.result()
                    llm_results[name] = result
                except Exception as e:
                    scheme = futures[future][0]
                    print(f"[eligibility] LLM check failed for {scheme.scheme_name}: {e}")
                    llm_results[scheme.scheme_name] = {
                        "scheme_name": scheme.scheme_name, "category": scheme.category,
                        "state": scheme.state, "official_link": scheme.official_link,
                        "is_eligible": True, "reason": "Could not verify — assuming eligible",
                    }

    results = []
    for scheme in schemes:
        if scheme.scheme_name in fast_results:
            results.append(fast_results[scheme.scheme_name])
        elif scheme.scheme_name in llm_results:
            results.append(llm_results[scheme.scheme_name])

    return results


# -------------------------------------------------
# Fetch eligible schemes from full DB
# -------------------------------------------------

def fetch_eligible_schemes(profile: UserProfile, k: int = 10) -> List[dict]:
    parts = []
    if profile.gender:
        parts.append(profile.gender.lower())
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
        parts = ["government scheme eligibility"]

    def _extract_and_check(fetch_k: int, query: str) -> List[dict]:
        docs = get_vector_db().as_retriever(search_kwargs={"k": fetch_k}).invoke(query)
        if not docs:
            print(f"[fetch_eligible] No docs returned for query: {query}")
            return []

        print(f"[fetch_eligible] Got {len(docs)} docs, extracting schemes...")
        context = format_docs(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTION_SYSTEM),
            ("human", "Extract ALL schemes from context. Copy values exactly.\n\nContext:\n{context}\n\nQuestion: find eligible schemes")
        ])
        try:
            result: SchemesListOutput = (prompt | get_structured_llm()).invoke({"context": context})
            candidate_schemes = result.schemes
        except Exception as e:
            print(f"[fetch_eligible] Extraction failed: {e}")
            return []

        print(f"[fetch_eligible] Extracted {len(candidate_schemes)} candidate schemes, checking eligibility...")

        if not candidate_schemes:
            return []

        strict_results = check_eligibility_for_schemes(profile, candidate_schemes)
        eligible = [
            {
                "scheme_name":   r["scheme_name"],
                "category":      r.get("category", ""),
                "state":         r.get("state", ""),
                "official_link": r.get("official_link", ""),
                "why_eligible":  r.get("reason", "Meets all eligibility criteria"),
            }
            for r in strict_results if r.get("is_eligible")
        ]
        print(f"[fetch_eligible] {len(eligible)} schemes passed strict check.")
        return eligible

    query = " ".join(parts) + " eligibility scheme"
    eligible = _extract_and_check(fetch_k=15, query=query)

    if not eligible:
        print("[fetch_eligible] First pass empty, trying broader search...")
        broad_query = (profile.state or "Gujarat") + " government scheme eligibility"
        eligible = _extract_and_check(fetch_k=20, query=broad_query)

    return eligible

# -------------------------------------------------
# Conversational reply
# -------------------------------------------------

def get_total_scheme_count() -> int:
    """Count total schemes stored in ChromaDB."""
    try:
        return get_vector_db()._collection.count()
    except Exception:
        return 0


def conversational_reply_stream(question: str, chat_history: list, lang: str = "en", intent: str = "conversational"):
    # ── Greeting ──────────────────────────────────────────────────────────────
    if intent == "greeting":
        greetings = {
            "en": "Hello! 👋 Welcome to Yojana AI — your Gujarat Government Scheme Assistant.\nI can help you find schemes, check eligibility, and get application details. How can I help you today?",
            "hi": "नमस्ते! 👋 योजना AI में आपका स्वागत है।\nमैं आपको सरकारी योजनाएं खोजने, पात्रता जाँचने और आवेदन प्रक्रिया जानने में मदद कर सकता हूँ। आज मैं आपकी कैसे मदद करूँ?",
            "gu": "નમસ્તે! 👋 યોજना AI માં આपनું સ્વાગत छे.\nहुं आपने सरकारी योजनाओ शोधवा, पात्रता चकासवा अने अरजी प्रक्रिया जाणवामां मदद करी शकुं छुं. आज हुं आपनी केवी रीते मदद करी शकुं?",
        }
        yield greetings.get(lang, greetings["en"])
        return

    # ── Scheme count ──────────────────────────────────────────────────────────
    if intent == "scheme_count":
        count = get_total_scheme_count()
        counts = {
            "en": f"There are currently **{count} government schemes** available in our Gujarat scheme database. You can ask me to find schemes by category, occupation, or check which ones you're eligible for!",
            "hi": f"हमारे गुजरात योजना डेटाबेस में वर्तमान में **{count} सरकारी योजनाएं** उपलब्ध हैं। आप मुझसे श्रेणी, पेशे के अनुसार योजनाएं खोजने या पात्रता जाँचने के लिए कह सकते हैं!",
            "gu": f"અमारा ગુजরात योजना डेटाबेस में अभी **{count} सरकारी योजनाएं** छे. आप मने श्रेणी के पेशा मुजब योजनाओ शोधवा के पात्रता चकासवा कही शको छो!",
        }
        yield counts.get(lang, counts["en"])
        return

    # ── General conversational ────────────────────────────────────────────────
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    lang_instruction = {
        "hi": "Always reply in Hindi (Devanagari script).",
        "gu": "Always reply in Gujarati (Gujarati script).",
        "en": "Reply in English.",
    }.get(lang, "Reply in English.")
    prompt = f"""You are a helpful Gujarat government scheme assistant.
{lang_instruction}
Help users find schemes and check eligibility.

Conversation:
{history_text}

User: {question}
AI:"""
    for chunk in get_llm().stream(prompt):
        yield chunk.content

# -------------------------------------------------
# Main ask function
# -------------------------------------------------

def ask_agent(question: str, session_id: str = "user_1", ui_lang: str = None):
    session = get_session(session_id)
    chat_history = session["history"]
    awaiting_profile = session.get("awaiting_profile", False)

    # ── Language detection ──────────────────────────────────────────────────
    detected = detect_language(question)
    if detected != "en":
        lang = detected
    elif ui_lang and ui_lang in ("en", "hi", "gu"):
        lang = ui_lang
    else:
        lang = session.get("lang", "en")
    session["lang"] = lang

    def reply_in_lang(text: str) -> str:
        return translate_response(text, lang)

    def ls(key: str) -> str:
        return get_string(key, lang)

    PROFILE_REQUEST = ls("profile_request")
    awaiting_profile = session.get("awaiting_profile", False)

    # ── ✅ PARALLEL: translate + detect intent at the same time ─────────────
    # For English: translate is a no-op so both run truly in parallel.
    # For non-English: translate first (1 LLM call), then intent on English text.
    if lang == "en":
        with ThreadPoolExecutor(max_workers=2) as executor:
            t_future = executor.submit(translate_to_english, question, lang)
            i_future = executor.submit(detect_intent, question, chat_history, awaiting_profile)
        question_en = t_future.result()
        intent = i_future.result()
    else:
        question_en = translate_to_english(question, lang)
        intent = detect_intent(question_en, chat_history, awaiting_profile)

    # ── User provided their profile ─────────────────────────────────────────
    if intent == "eligibility_check":
        profile = extract_user_profile(question_en)
        gender_hint = extract_gender_from_question(question_en)
        profile = merge_gender_into_profile(profile, gender_hint)
        session["user_profile"] = profile
        session["awaiting_profile"] = False

        if awaiting_profile and session.get("last_schemes"):
            last = session["last_schemes"]
            scheme_objects = []
            for s in last:
                if isinstance(s, SchemeOutput):
                    scheme_objects.append(s)
                else:
                    name = s.get("scheme_name", "")
                    if name:
                        fetched = fetch_schemes(name, [], k=3, last_schemes=[])
                        if fetched:
                            scheme_objects.append(fetched[0])
                        else:
                            scheme_objects.append(SchemeOutput(
                                scheme_name=name,
                                description="", category=s.get("category", ""),
                                benefits="", eligibility="",
                                documents_required="", application_process="",
                                state=s.get("state", ""),
                                official_link=s.get("official_link", "")
                            ))
            if scheme_objects:
                print("🔍 Checking eligibility for shown schemes...")
                results = check_eligibility_for_schemes(profile, scheme_objects)
                save_to_history(session_id, question, f"Checked eligibility for {len(scheme_objects)} schemes.")
                yield {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results, "lang": lang}
                return

        # No prior shown schemes OR scheme_objects came out empty → search full DB
        print("🔍 No prior schemes found — searching full DB for eligibility...")
        try:
            eligible = fetch_eligible_schemes(profile, k=4)
        except Exception as e:
            print(f"[eligibility_check] fetch_eligible_schemes error: {e}")
            reply = reply_in_lang("Sorry, something went wrong while checking eligibility. Please try again.")
            save_to_history(session_id, question, reply)
            yield {"type": "conversational", "reply": reply, "lang": lang}
            return

        if not eligible:
            reply = reply_in_lang(ls("no_schemes_found"))
            save_to_history(session_id, question, reply)
            yield {"type": "conversational", "reply": reply, "lang": lang}
            return

        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        session["last_schemes"] = eligible
        yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}
        return

    # ── User asks eligibility for previously shown schemes ───────────────────
    if intent == "eligibility_for_shown":
        last_schemes = session.get("last_schemes", [])

        # ✅ GUARD: If no schemes have been shown yet, treat this as a scheme search first.
        # The user should see schemes before being asked for their profile.
        if not last_schemes:
            print("ℹ️  eligibility_for_shown but no schemes shown yet — fetching schemes first...")
            schemes = fetch_schemes(question_en, chat_history, k=5, last_schemes=[])
            session["last_schemes"] = schemes
            if schemes:
                selected = schemes[:5]
                save_to_history(session_id, question, f"Showed schemes: {', '.join(s.scheme_name for s in selected)}")
                dicts = [s.model_dump() for s in selected]
                with ThreadPoolExecutor(max_workers=min(len(dicts), 5)) as ex:
                    enriched = list(ex.map(apply_visit_site_fallback, dicts))
                yield {
                    "type": "full_detail",
                    "schemes": enriched,
                    "lang": lang,
                }
                return
            else:
                reply = reply_in_lang(ls("no_schemes_found"))
                save_to_history(session_id, question, reply)
                yield {"type": "conversational", "reply": reply, "lang": lang}
                return

        gender_hint = extract_gender_from_question(question_en)

        if gender_hint and not session.get("user_profile"):
            session["user_profile"] = UserProfile(gender=gender_hint)
        elif gender_hint and session.get("user_profile"):
            session["user_profile"] = merge_gender_into_profile(session["user_profile"], gender_hint)

        if not session.get("user_profile") or not any([
            session["user_profile"].age,
            session["user_profile"].income,
            session["user_profile"].occupation,
            session["user_profile"].caste_category,
        ]):
            session["awaiting_profile"] = True
            save_to_history(session_id, question, PROFILE_REQUEST)
            yield {"type": "conversational", "reply": PROFILE_REQUEST, "lang": lang}
            return

        profile = session["user_profile"]

        if is_fresh_search_request(question_en):
            profile = merge_gender_into_profile(profile, gender_hint)
            session["user_profile"] = profile
            print("🔍 Searching all schemes for your eligibility...")
            eligible = fetch_eligible_schemes(profile, k=4)
            if not eligible:
                reply = reply_in_lang(ls("no_additional_schemes"))
                save_to_history(session_id, question, reply)
                yield {"type": "conversational", "reply": reply, "lang": lang}
                return
            save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
            session["last_schemes"] = eligible
            yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}
            return

        if last_schemes:
            print("🔍 Checking eligibility for shown schemes...")
            results = check_eligibility_for_schemes(profile, last_schemes)
            save_to_history(session_id, question, f"Checked eligibility for {len(last_schemes)} schemes.")
            yield {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results, "lang": lang}
            return

        print("🔍 Searching all schemes for your eligibility...")
        eligible = fetch_eligible_schemes(profile, k=4)
        if not eligible:
            reply = reply_in_lang(ls("no_schemes_found"))
            save_to_history(session_id, question, reply)
            yield {"type": "conversational", "reply": reply, "lang": lang}
            return
        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        session["last_schemes"] = eligible
        yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}
        return

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
        names_text += "\n\n💡 Ask me for full details of any scheme above."
        reply = reply_in_lang(names_text)
        save_to_history(session_id, question, reply)

        # Stream names as text tokens (ChatGPT-style)
        import time
        yield {"type": "conversational_start", "lang": lang}
        # Split into small chunks for typing effect
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield {"type": "chunk", "text": chunk}
            time.sleep(0.02)  # Simulate actual typing speed
        yield {"type": "conversational_end"}
        return

    if intent == "specific_field":
        field = detect_field(question_en)
        lines = [f"• {s.scheme_name}:\n  {apply_visit_site_fallback(s.model_dump()).get(field, 'Not Available')}"
                 for s in schemes]
        reply_en = "\n\n".join(lines)
        reply = reply_in_lang(reply_en)
        save_to_history(session_id, question, reply)
        yield {"type": "specific_field", "field": field, "reply": reply, "lang": lang}
        return

    if intent in ("conversational", "greeting", "scheme_count"):
        yield {"type": "conversational_start", "lang": lang}
        full_reply = ""
        for chunk in conversational_reply_stream(question, chat_history, lang, intent=intent):
            full_reply += chunk
            yield {"type": "chunk", "text": chunk}
        save_to_history(session_id, question, full_reply)
        yield {"type": "conversational_end"}
        return

    # full_detail — Stream TEXT -> then render CARDS
    import time
    selected = schemes[:limit] if limit else schemes
    save_to_history(session_id, question, f"Showed details for: {', '.join(s.scheme_name for s in selected)}")
    dicts = [s.model_dump() for s in selected]

    preview_parts = []
    for i, d in enumerate(dicts):
        benefits_short = str(d.get("benefits", ""))[:150]
        preview_parts.append(f"**{i+1}. {d['scheme_name']}**\nBenefits: {benefits_short}...\n")
    
    preview_text = "Here are the details:\n\n" + "\n".join(preview_parts) + "\n*Loading cards...*"
    preview_text = reply_in_lang(preview_text)

    # Run enrichment in background WHILE streaming text
    with ThreadPoolExecutor(max_workers=min(len(dicts), 5)) as ex:
        futures = {ex.submit(apply_visit_site_fallback, d): i for i, d in enumerate(dicts)}
        
        # Stream the typing effect text
        yield {"type": "conversational_start", "lang": lang}
        words = preview_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield {"type": "chunk", "text": chunk}
            time.sleep(0.03)  # simulates typing and masks the web scraping latency
        
        # Ensure we don't accidentally send a conversational_end yet, we just pause
        results = [None] * len(dicts)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = dicts[idx]

    # Convert the streamed text bubble into cards!
    yield {"type": "convert_to_cards", "schemes": results, "lang": lang}
    return