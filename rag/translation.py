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
        "profile_request": """ज़रूर! पात्रता जाँचने के लिए कृपया अपनी जानकारी दें:\n\n  \u2022 आयु (उम्र)\n  \u2022 वार्षिक आय  (जैसे 1.5 लाख, 50,000)\n  \u2022 पेशा        (जैसे छात्र, किसान, स्व-रोजगार)\n  \u2022 राज्य       (जैसे गुजरात)\n  \u2022 लिंग        (पुरुष / महिला)\n  \u2022 जाति/श्रेणी (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\nउदाहरण:\n  आयु: 22, आय: 1.5 लाख, पेशा: छात्र, राज्य: गुजरात, जाति: OBC, लिंग: पुरुष""",
        "no_schemes_found": "कोई मिलती-जुलती योजना नहीं मिली। कृपया अपनी आयु, आय, पेशा, राज्य और जाति की जानकारी दें।",
        "no_additional_schemes": "आपके प्रोफ़ाइल के लिए कोई अतिरिक्त योजना नहीं मिली।",
        "ask_schemes_first": "कृपया पहले कोई योजना खोजें, फिर मैं उनके लिए आपकी पात्रता जाँच सकता हूँ।",
    },
    "gu": {
        "profile_request": """ચોક્કસ! પાત્રતા તપાસવા માટે કૃપા કરીને આ માહિતી આપો:\n\n  \u2022 ઉંમર\n  \u2022 વાર્ષિક આવક  (દા.ત. 1.5 લાખ, 50,000)\n  \u2022 વ્યવસાય      (દા.ત. વિદ્યાર્થી, ખેડૂત, સ્વ-રોજગાર)\n  \u2022 રાજ્ય        (દા.ત. ગુજરાત)\n  \u2022 જાતિ         (પુરુષ / સ્ત્રી)\n  \u2022 જ્ઞાતિ/વર્ગ  (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\nઉદાહરણ:\n  ઉંમર: 22, આવક: 1.5 લાખ, વ્યવસાય: વિદ્યાર્થી, રાજ્ય: ગુજરાત, જ્ઞાતિ: OBC, જાતિ: પુરુષ""",
        "no_schemes_found": "કોઈ મળતી યોજના મળી નહીં. કૃપા કરીને ઉંમર, આવક, વ્યવસાય, રાજ્ય અને જ્ઞાતિ આપો.",
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

    clean = text.strip()
    buttons = {
        "किसानों के लिए योजनाएं 🌾": "Schemes for farmers",
        "महीला कल्याण योजनाएं": "Women welfare schemes",
        "शिक्षा छात्रवृत्ति": "Education scholarships",
        "स्वास्थ्य योजनाएं": "Healthcare schemes",
        "आवास योजना": "housing scheme",
        "युवाओं के लिए स्टार्टअप योजनाएं": "Startup schemes for youth",
        "गुजरात में योजनाएं": "Schemes in Gujarat",
        "कौशल विकास कार्यक्रम": "Skill development programs",
        
        "ખેડૂતો માટે યોજનાઓ 🌾": "Schemes for farmers",
        "મહિલા કલ્યાણ યોજનાઓ": "Women welfare schemes",
        "શિક્ષણ શિષ્યવૃત્તિ": "Education scholarships",
        "આરોગ્ય સેવા યોજનાઓ": "Healthcare schemes",
        "આવાસ યોજના": "housing scheme",
        "યુવાનો માટે સ્ટાર્ટઅપ યોજનાઓ": "Startup schemes for youth",
        "ગુજરાતમાં યોજનાઓ": "Schemes in Gujarat",
        "કૌશલ્ય વિકાસ કાર્યક્રમ": "Skill development programs"
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
        f"Example: 'બાજરી' -> 'bajri (millet)', 'કપાસ' -> 'kapas (cotton)', 'મગફળી' -> 'magfali (groundnut)'. "
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

    lang_name = {"hi": "Hindi (हिन्दी, Devanagari script)", "gu": "Gujarati (ગુજરાતી, Gujarati script)"}[target_lang]
    script_warn = {
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (हिन्दी). Do NOT use Gujarati script.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (ગુજરાતી). Do NOT use Hindi/Devanagari script.",
    }[target_lang]
    r = get_llm().invoke(
        f"Translate this English text to {lang_name}.\n"
        f"{script_warn}\n"
        f"Keep unchanged: scheme names, official links, SC/ST/OBC/EWS/SEBC, state names, ₹ amounts, numbers.\n"
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
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (हिन्दी). Do NOT output Gujarati.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (ગુજરાતી). Do NOT output Hindi.",
    }.get(target_lang, "")
    prompt = f"""Translate text values to {lang_name} while keeping numbers, Rs amounts, links, and SC/ST/OBC acronyms exactly the same.
CRITICAL: Preserve all original line breaks and structural patterns like numbered lists (1. 2. 3.) or bullet points (•).
CLEAN UI RULE: Remove redundant link-directing filler phrases such as "click here", "click", "here", "visit link", or native versions like "यहाँ क्लिक करें", "यहाँ", "અહીં ક્લિક કરો", "અહીં" from the output to keep it cleaner.
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
