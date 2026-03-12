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
            "last_schemes": [],       # List[SchemeOutput] — last shown schemes
            "last_limit": None,
            "user_profile": None,     # UserProfile — cached after first provided
            "awaiting_profile": False, # True = we asked user for profile, waiting for reply
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

def is_followup_on_previous(question: str, chat_history: list) -> bool:
    if not chat_history: return False
    # If user wants fresh/new results, never treat as followup
    if is_fresh_search_request(question): return False
    hints = ["first","second","third","fourth","fifth","1st","2nd","3rd","4th","5th",
             "that","it","this","above","those","same","previous","last","shown","these"]
    return any(h in question.lower() for h in hints)

def rewrite_question(question: str, chat_history: list) -> str:
    if not chat_history: return question
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    r = llm.invoke(f"Rewrite as standalone search query.\n\nConversation:\n{history_text}\n\nFollow-up: {question}\n\nStandalone query:")
    return r.content.strip()

def resolve_scheme_reference(question: str, schemes: list) -> list:
    q = question.lower()
    ordinals = {"first":0,"1st":0,"second":1,"2nd":1,"third":2,"3rd":2,"fourth":3,"4th":3,"fifth":4,"5th":4}
    for word, idx in ordinals.items():
        if word in q and idx < len(schemes):
            return [schemes[idx]]
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
    "SC":       ["sc", "scheduled caste", "harijan", "dalit"],
    "ST":       ["st", "scheduled tribe", "adivasi", "tribal"],
    "OBC":      ["obc", "sebc", "ebc", "other backward", "educationally backward",
                 "socially and educationally backward", "backward class"],
    "SEBC":     ["obc", "sebc", "ebc", "other backward", "educationally backward",
                 "socially and educationally backward", "backward class"],
    "EBC":      ["obc", "sebc", "ebc", "educationally backward", "backward class"],
    "EWS":      ["ews", "economically weaker", "general"],
    "General":  ["general", "unreserved", "non-reserved", "non reserved", "open"],
    "NT":       ["nt", "dnt", "nomadic", "de-notified"],
    "DNT":      ["nt", "dnt", "nomadic", "de-notified"],
    "Minority": ["minority", "muslim", "christian", "sikh", "jain", "buddhist"],
}

OPEN_TO_ALL_TERMS = [
    "unreserved", "non-reserved", "non reserved", "general",
    "open to all", "all categories", "all castes", "all citizens",
    "all residents", "irrespective of caste", "any category",
    "all community", "regardless of caste",
]

def python_caste_check(eligibility_text: str, user_caste: str) -> tuple:
    """
    Pure Python caste eligibility check — no LLM.
    Returns (is_eligible: bool, reason: str)
    """
    if not eligibility_text or eligibility_text.strip().lower() in (
        "not available", "not found", "", "n/a"
    ):
        return True, "No caste restriction specified — open to all categories"

    text_lower = eligibility_text.lower()

    # 1. Check if scheme is open to all
    for term in OPEN_TO_ALL_TERMS:
        if term in text_lower:
            return True, f"Scheme is open to all categories ('{term}' found) — {user_caste} eligible"

    # 2. Get match terms for user's caste
    user_caste_clean = (user_caste or "").strip().upper()
    match_terms = CASTE_MATCH_MAP.get(user_caste_clean, [user_caste.lower()])

    # 3. Check if any of user's caste terms appear in eligibility
    for term in match_terms:
        if term in text_lower:
            return True, f"{user_caste} ({term}) matches scheme eligibility criteria"

    # 4. Check if eligibility mentions any OTHER specific caste group (not user's)
    all_restricted_terms = []
    for caste, terms in CASTE_MATCH_MAP.items():
        if caste != user_caste_clean:
            all_restricted_terms.extend(terms)

    found_other = [t for t in all_restricted_terms if t in text_lower]
    if found_other:
        return False, f"Scheme restricted to specific category ({found_other[0]}) — {user_caste} not eligible"

    # 5. No caste info found → assume open
    return True, "No specific caste restriction detected — assumed open to all"


def check_eligibility_for_schemes(profile: UserProfile, schemes: List[SchemeOutput]) -> List[dict]:
    """
    Check eligibility using Python for caste (reliable),
    and LLM only for non-caste criteria (age, income, state).
    """
    user_caste = (profile.caste_category or "General").strip()
    results = []

    for scheme in schemes:
        # Step 1: Python caste check (100% reliable)
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

        # Step 2: LLM checks ONLY age/income/state — caste is already handled by Python
        profile_text = profile_to_text(profile)

        # Strip any caste-related sentences from eligibility to prevent LLM re-checking caste
        import re as _re
        caste_keywords = ["caste", "category", "sc", "st", "obc", "sebc", "ebc",
                          "unreserved", "reserved", "general", "backward", "scheduled",
                          "minority", "ews", "nt", "dnt", "nomadic", "tribal"]
        eligibility_lines = scheme.eligibility.split(".") if scheme.eligibility else []
        non_caste_lines = [
            line for line in eligibility_lines
            if not any(kw in line.lower() for kw in caste_keywords)
        ]
        non_caste_eligibility = ". ".join(non_caste_lines).strip() or "No specific age/income/state restriction mentioned."

        prompt = f"""You are checking NON-CASTE eligibility criteria only.
Caste/category has already been verified — DO NOT mention or evaluate caste at all.

USER PROFILE:
{profile_text}

SCHEME: {scheme.scheme_name}
STATE: {scheme.state}
ELIGIBILITY (caste lines removed): {non_caste_eligibility}

Check ONLY: age limit, income limit, occupation/profession, state of residence.
If no such restrictions are mentioned, answer PASS.

Reply with ONLY one of:
PASS: <brief reason about age/income/state>
FAIL: <specific criterion that failed (age/income/state only)>"""

        r = llm.invoke(prompt)
        answer = r.content.strip()

        if answer.upper().startswith("PASS"):
            detail = answer.split(":", 1)[-1].strip() if ":" in answer else "Meets all criteria"
            reason = f"{caste_reason.split('—')[0].strip()}. {detail}"
            results.append({
                "scheme_name": scheme.scheme_name,
                "state": scheme.state,
                "official_link": scheme.official_link,
                "is_eligible": True,
                "reason": reason,
            })
        else:
            detail = answer.split(":", 1)[-1].strip() if ":" in answer else answer
            results.append({
                "scheme_name": scheme.scheme_name,
                "state": scheme.state,
                "official_link": scheme.official_link,
                "is_eligible": False,
                "reason": f"Caste eligible, but: {detail}",
            })

    return results


# -------------------------------------------------
# Fetch eligible schemes from full DB
# -------------------------------------------------

def fetch_eligible_schemes(profile: UserProfile, k: int = 10) -> List[dict]:
    parts = [p for p in [profile.occupation, profile.state, profile.caste_category,
             f"age {profile.age}" if profile.age else "",
             f"income {profile.income}" if profile.income else ""] if p]
    if not parts: parts = ["government scheme"]

    docs = vector_db.as_retriever(search_kwargs={"k": k}).invoke(" ".join(parts) + " eligibility")
    context = format_docs(docs)
    profile_text = profile_to_text(profile)

    r = llm.invoke(f"""You are an Indian government scheme eligibility expert.


CRITICAL CASTE/CATEGORY RULES — READ CAREFULLY:
1. "Unreserved", "General", "Open", "Non-reserved", "All categories" = scheme is OPEN TO EVERYONE including SC/ST/OBC/EWS. Do NOT reject based on caste for these.
2. "SC only" or "Scheduled Caste" = only SC applicants eligible.
3. "ST only" or "Scheduled Tribe" = only ST applicants eligible.
4. "OBC" or "SEBC" or "EBC" or "Educationally Backward Class" = OBC/SEBC applicants ARE eligible.
5. "Minority" = only minority community applicants eligible.
6. If NO caste/category restriction is mentioned = open to all categories.
7. NEVER reject an OBC/SC/ST applicant from a "General/Unreserved/Open" scheme — those terms mean open to all.

USER PROFILE:
{profile_text}

AVAILABLE SCHEMES:
{context}

Return ONLY schemes where this user clearly qualifies.
Apply the caste rules above strictly — do NOT exclude users from open/unreserved/general schemes.

Return ONLY this JSON array:
[
  {{
    "scheme_name": "...",
    "category": "...",
    "state": "...",
    "official_link": "...",
    "why_eligible": "Specific reason mentioning matched criteria"
  }}
]

Return ONLY valid JSON, nothing else.""")
    raw = re.sub(r"```json|```", "", r.content.strip()).strip()
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []

# -------------------------------------------------
# Conversational reply
# -------------------------------------------------

def conversational_reply(question: str, chat_history: list) -> str:
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    r = llm.invoke(f"""You are a helpful Indian government scheme assistant.
Help users find schemes and check eligibility.

Conversation:
{history_text}

User: {question}
AI:""")
    return r.content.strip()

# -------------------------------------------------
# Profile request message
# -------------------------------------------------

PROFILE_REQUEST = """Sure! Please share your details so I can check eligibility:

  • Age
  • Annual Income  (e.g. 1.5 lakh, 50,000)
  • Occupation     (e.g. student, farmer, self-employed)
  • State          (e.g. Gujarat)
  • Gender         (optional)
  • Caste/Category (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)

Example:
  age: 22, income: 1.5 lakh, occupation: student, state: Gujarat, caste: OBC"""

# -------------------------------------------------
# Main ask function
# -------------------------------------------------

def ask_agent(question: str, session_id: str = "user_1"):
    session = get_session(session_id)
    chat_history = session["history"]
    awaiting_profile = session.get("awaiting_profile", False)

    intent = detect_intent(question, chat_history, awaiting_profile)

    # ── User provided their profile ─────────────────────────────────────────
    if intent == "eligibility_check":
        profile = extract_user_profile(question)
        session["user_profile"] = profile

        # Were we waiting to check SHOWN schemes specifically?
        if awaiting_profile and session.get("last_schemes"):
            session["awaiting_profile"] = False
            print("🔍 Checking eligibility for shown schemes, please wait...")
            results = check_eligibility_for_schemes(profile, session["last_schemes"])
            save_to_history(session_id, question, f"Checked eligibility for {len(session['last_schemes'])} schemes.")
            return {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results}

        # General eligibility → search full DB
        session["awaiting_profile"] = False
        print("🔍 Checking eligibility across all schemes, please wait...")
        eligible = fetch_eligible_schemes(profile, k=10)

        if not eligible:
            reply = "No matching schemes found. Try providing more details like age, income, occupation, state, and caste/category."
            save_to_history(session_id, question, reply)
            return {"type": "conversational", "reply": reply}

        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        return {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible}

    # ── User asks eligibility for previously shown schemes ───────────────────
    if intent == "eligibility_for_shown":
        last_schemes = session.get("last_schemes", [])

        # If user wants NEW/DIFFERENT schemes they are eligible for → full DB search
        if is_fresh_search_request(question):
            if not session.get("user_profile"):
                session["awaiting_profile"] = True
                save_to_history(session_id, question, PROFILE_REQUEST)
                return {"type": "conversational", "reply": PROFILE_REQUEST}
            profile = session["user_profile"]
            print("🔍 Searching all schemes for your eligibility, please wait...")
            eligible = fetch_eligible_schemes(profile, k=10)
            if not eligible:
                reply = "No additional matching schemes found for your profile."
                save_to_history(session_id, question, reply)
                return {"type": "conversational", "reply": reply}
            save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
            return {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible}

        if not last_schemes:
            reply = "Please first ask for some schemes, then I can check your eligibility for them."
            save_to_history(session_id, question, reply)
            return {"type": "conversational", "reply": reply}

        # Already have profile cached → check immediately
        if session.get("user_profile"):
            profile = session["user_profile"]
            print("🔍 Checking eligibility for shown schemes, please wait...")
            results = check_eligibility_for_schemes(profile, last_schemes)
            save_to_history(session_id, question, f"Checked eligibility for {len(last_schemes)} schemes.")
            return {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results}

        # No profile yet → ask for it, remember we want to check shown schemes
        session["awaiting_profile"] = True
        save_to_history(session_id, question, PROFILE_REQUEST)
        return {"type": "conversational", "reply": PROFILE_REQUEST}

    # ── Normal scheme queries ────────────────────────────────────────────────
    session["awaiting_profile"] = False
    limit = parse_limit(question)
    followup = is_followup_on_previous(question, chat_history)
    fresh = is_fresh_search_request(question)

    if followup and not fresh and session["last_schemes"]:
        schemes = resolve_scheme_reference(question, session["last_schemes"])
    else:
        # Fresh search — exclude previously shown scheme names so results are new
        prev_names = [s.scheme_name for s in session["last_schemes"]] if fresh else []
        fetch_k = max(limit or 3, 5) + len(prev_names)  # fetch extra to compensate
        schemes = fetch_schemes(question, chat_history, k=fetch_k, last_schemes=session["last_schemes"])
        # Filter out previously shown schemes if user wants new ones
        if fresh and prev_names:
            schemes = [s for s in schemes if s.scheme_name not in prev_names]
        session["last_schemes"] = schemes
        session["last_limit"] = limit

    if intent == "names_only":
        selected = schemes[:limit] if limit else schemes
        reply = "\n".join(f"{i+1}. {s.scheme_name}" for i, s in enumerate(selected))
        save_to_history(session_id, question, reply)
        return {"type": "names_only", "reply": reply}

    if intent == "specific_field":
        field = detect_field(question)
        lines = [f"• {s.scheme_name}:\n  {apply_visit_site_fallback(s.model_dump()).get(field,'Not Available')}"
                 for s in schemes]
        reply = "\n\n".join(lines)
        save_to_history(session_id, question, reply)
        return {"type": "specific_field", "field": field, "reply": reply}

    if intent == "conversational":
        reply = conversational_reply(question, chat_history)
        save_to_history(session_id, question, reply)
        return {"type": "conversational", "reply": reply}

    # full_detail
    selected = schemes[:limit] if limit else schemes
    save_to_history(session_id, question, f"Showed details for: {', '.join(s.scheme_name for s in selected)}")
    return {"type": "full_detail", "schemes": [apply_visit_site_fallback(s.model_dump()) for s in selected]}