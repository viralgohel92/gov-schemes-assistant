import re
import json
from rag.llm import get_llm, TranslatedScheme

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
        "profile_request": """     !                                             :\n\n  \u2022     (    )\n  \u2022             (     1.5    , 50,000)\n  \u2022             (          ,      ,    -      )\n  \u2022             (           )\n  \u2022             (      /      )\n  \u2022     /       (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\n      :\n     : 22,   : 1.5    ,     :      ,      :       ,     : OBC,     :      """,
        "no_schemes_found": "         -                                     ,   ,     ,                              ",
        "no_additional_schemes": "                                                   ",
        "ask_schemes_first": "                          ,                                             ",
    },
    "gu": {
        "profile_request": """      !                                            :\n\n  \u2022     \n  \u2022              (  . . 1.5    , 50,000)\n  \u2022              (  . .           ,      ,    -      )\n  \u2022              (  . .       )\n  \u2022              (      /       )\n  \u2022       /      (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\n      :\n      : 22,    : 1.5    ,        :           ,      :       ,       : OBC,     :      """,
        "no_schemes_found": "                       .                ,    ,        ,                     .",
        "no_additional_schemes": "                                              .",
        "ask_schemes_first": "                                ,                             .",
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

    clean = text.strip()
    buttons = {
        "                        ": "Schemes for farmers",
        "                    ": "Women welfare schemes",
        "                  ": "Education scholarships",
        "                 ": "Healthcare schemes",
        "          ": "housing scheme",
        "                               ": "Startup schemes for youth",
        "                  ": "Schemes in Gujarat",
        "                    ": "Skill development programs",
        
        "                    ": "Schemes for farmers",
        "                   ": "Women welfare schemes",
        "                  ": "Education scholarships",
        "                  ": "Healthcare schemes",
        "          ": "housing scheme",
        "                            ": "Startup schemes for youth",
        "                ": "Schemes in Gujarat",
        "                      ": "Skill development programs"
    }
    if clean in buttons:
        return buttons[clean]

    if not text or not text.strip():
        return text

    lang_name = {"hi": "Hindi", "gu": "Gujarati"}[source_lang]
    prompt = (
        f"Translate this {lang_name} query to English for a government scheme search engine.\n"
        f"Critical for Search Accuracy: For local terms (especially agricultural crops like 'Bajri', 'Kapas', or 'Jowar', products, or occupations like 'Khedut'), "
        f"provide BOTH the common transliterated name and the standard English translation. "
        f"Example: '     ' -> 'bajri (millet)', '    ' -> 'kapas (cotton)', '     ' -> 'magfali (groundnut)'. "
        f"Keep unchanged: scheme names, acronyms like SC/ST/OBC/EWS/SEBC, state names, numbers, rupee amounts.\n"
        f"Return ONLY the English translation.\n\n"
        f"{lang_name}: {text}\n"
        f"English:"
    )
    r = get_llm().invoke(prompt)
    translated = r.content.strip()
    return translated if translated else text

def translate_response(text: str, target_lang: str) -> str:
    """Translate agent response from English to target language."""
    if target_lang == "en":
        return text
    if not text or not text.strip():
        return text

    lang_name = {"hi": "Hindi (      , Devanagari script)", "gu": "Gujarati (       , Gujarati script)"}[target_lang]
    script_warn = {
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (      ). Do NOT use Gujarati script.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (       ). Do NOT use Hindi/Devanagari script.",
    }[target_lang]
    r = get_llm().invoke(
        f"Translate this English text to {lang_name}.\n"
        f"{script_warn}\n"
        f"Keep unchanged: scheme names, official links, SC/ST/OBC/EWS/SEBC, state names,   amounts, numbers.\n"
        f"Return ONLY the translation.\n\nEnglish: {text}\n\nTranslation:"
    )
    translated = r.content.strip()
    return translated if translated else text

def get_string(key: str, lang: str) -> str:
    """Get a static UI string in the given language."""
    return LANG_STRINGS.get(lang, LANG_STRINGS["en"]).get(key, LANG_STRINGS["en"].get(key, ""))

def translate_scheme_dict(d: dict, target_lang: str) -> dict:
    if target_lang == "en":
        return d
    lang_name = {"hi": "Hindi", "gu": "Gujarati"}.get(target_lang, "English")
    
    # We only care about translating text that users read.
    keys_to_translate = [k for k in ("scheme_name", "description", "benefits", 
                                     "eligibility", "documents_required", 
                                     "application_process", "category") 
                         if k in d and d[k]]
    
    if not keys_to_translate:
        return d

    to_translate = {k: d[k] for k in keys_to_translate}
    json_in = json.dumps(to_translate, ensure_ascii=False)
    
    script_warn = {
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (      ). Do NOT output Gujarati.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (       ). Do NOT output Hindi.",
    }.get(target_lang, "")
    prompt = f"""Translate text values to {lang_name} while keeping numbers, Rs amounts, links, and SC/ST/OBC acronyms exactly the same.
CRITICAL: Preserve all original line breaks and structural patterns like numbered lists (1. 2. 3.) or bullet points ( ).
CLEAN UI RULE: Remove redundant link-directing filler phrases such as "click here", "click", "here", "visit link", or native versions like "               ", "    ", "              ", "    " from the output to keep it cleaner.
{script_warn}
Return ONLY valid JSON that matches the required schema.

Input JSON to translate:
{json_in}
"""
    try:
        translated = get_llm().with_structured_output(TranslatedScheme).invoke(prompt)
        res = d.copy()
        res.update({k: v for k, v in translated.model_dump().items() if v})
        return res
    except Exception as e:
        print(f"[translate_scheme_dict] Translation failed: {e}")
        return d
