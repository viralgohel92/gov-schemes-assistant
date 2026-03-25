import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import ChatPromptTemplate
from rag.llm import get_llm, get_structured_llm, get_profile_llm, get_vector_db, UserProfile, SchemeOutput, SchemesListOutput
from rag.utils import MISSING, profile_to_text, format_docs
from rag.retriever import EXTRACTION_SYSTEM

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
        return True, "No caste restriction specified \u2014 open to all categories"

    text = eligibility_text

    for pattern in OPEN_TO_ALL_TERMS:
        if _re_search(pattern, text):
            return True, f"Scheme is open to all categories \u2014 {user_caste} eligible"

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
        return False, f"Scheme restricted to {display} category \u2014 {user_caste_clean} not eligible"

    return True, "No specific caste restriction \u2014 assumed open to all"

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
        return True, "No gender specified \u2014 skipping gender check"

    if not eligibility_text or eligibility_text.strip().lower() in MISSING:
        return True, "No gender restriction in eligibility"

    text = eligibility_text
    gender_lower = user_gender.strip().lower()
    is_female = gender_lower in ("female", "woman", "women", "girl", "mahila")
    is_male   = gender_lower in ("male", "man", "men", "boy", "gents", "purush")

    has_female_restriction = any(_re_search(p, text) for p in GENDER_FEMALE_PATTERNS)
    has_male_restriction   = any(_re_search(p, text) for p in GENDER_MALE_PATTERNS)

    if not has_female_restriction and not has_male_restriction:
        return True, "No gender restriction \u2014 open to all"

    if has_female_restriction and not has_male_restriction:
        if is_female:
            return True, "Scheme is for women \u2014 user is Female ✓"
        return False, "Scheme is for women only \u2014 user is not Female"

    if has_male_restriction and not has_female_restriction:
        if is_male:
            return True, "Scheme is for men \u2014 user is Male ✓"
        return False, "Scheme is for men only \u2014 user is not Male"

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
Caste and gender have already been verified separately \u2014 DO NOT re-evaluate them.

USER PROFILE:
{profile_text}

SCHEME: {scheme.scheme_name}
STATE: {scheme.state}
ELIGIBILITY (caste & gender lines removed): {clean_eligibility}

Check ONLY: age limit, income limit, occupation/profession, state of residence.
If the scheme has no such restrictions, answer PASS.

Reply with ONLY one of:
PASS: <brief reason about age/income/state/occupation>
FAIL: <specific criterion that failed \u2014 age/income/state/occupation only>"""

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
                        "is_eligible": True, "reason": "Could not verify \u2014 assuming eligible",
                    }

    results = []
    for scheme in schemes:
        if scheme.scheme_name in fast_results:
            results.append(fast_results[scheme.scheme_name])
        elif scheme.scheme_name in llm_results:
            results.append(llm_results[scheme.scheme_name])

    return results

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
