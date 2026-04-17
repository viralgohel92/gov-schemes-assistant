import re
import json
from rag.llm import get_llm, TranslatedScheme, SuggestionListOutput

# -------------------------------------------------
# Language Support
# -------------------------------------------------

SUPPORTED_LANGUAGES = {"english": "en", "hindi": "hi", "gujarati": "gu"}

LANG_STRINGS = {
    "en": {
        "profile_request": """Sure! Please share your details so I can check eligibility:\n\n  • Age\n  • Annual Income  (e.g. 1.5 lakh, 50,000)\n  • Occupation     (e.g. student, farmer, self-employed)\n  • State          (e.g. Gujarat)\n  • Gender         (Male / Female)\n  • Caste/Category (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)\n\nExample:\n  age: 22, income: 1.5 lakh, occupation: student, state: Gujarat, caste: OBC, Gender: Male""",
        "no_schemes_found": "No matching schemes found. Try providing more details like age, income, occupation, state, and caste/category.",
        "no_additional_schemes": "No additional matching schemes found for your profile.",
        "ask_schemes_first": "Please first ask for some schemes, then I can check your eligibility for them.",
    },
    "hi": {
        "profile_request": """\u0928\u093f\u0936\u094d\u091a\u093f\u0924 \u0930\u0942\u092a \u0938\u0947! \u0915\u0943\u092a\u092f\u093e \u0905\u092a\u0928\u0940 \u092a\u093e\u0924\u094d\u0930\u0924\u093e \u091c\u093e\u0902\u091a\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0935\u093f\u0935\u0930\u0923 \u0938\u093e\u091d\u093e \u0915\u0930\u0947\u0902:

  \u2022 \u0906\u092f\u0941 (Age)
  \u2022 \u0935\u093e\u0930\u094d\u0937\u093f\u0915 \u0906\u092f (\u091c\u0948\u0938\u0947 1.5 \u0932\u093e\u0916, 50,000)
  \u2022 \u0935\u094d\u092f\u0935\u0938\u093e\u092f (\u091c\u0948\u0938\u0947 \u091b\u093e\u0924\u094d\u0930, \u0915\u093f\u0938\u093e\u0928, \u0938\u094d\u0935-\u0928\u093f\u092f\u094b\u091c\u093f\u0924)
  \u2022 \u0930\u093e\u091c\u094d\u092f (\u091c\u0948\u0938\u0947 \u0917\u0941\u091c\u0930\u093e\u0924)
  \u2022 \u0932\u093f\u0902\u0917 (\u092a\u0941\u0930\u0941\u0937 / \u092e\u0939\u093f\u0932\u093e)
  \u2022 \u091c\u093e\u0924\u093e/\u0936\u094d\u0930\u0947\u0923\u0940 (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)

\u0909\u092a\u0939\u093e\u0930:
  \u0906\u092f\u0941: 22, \u0906\u092f: 1.5 \u0932\u093e\u0916, \u0935\u094d\u092f\u0935\u0938\u093e\u092f: \u091b\u093e\u0924\u094d\u0930, \u0930\u093e\u091c\u094d\u092f: \u0917\u0941\u091c\u0930\u093e\u0924, \u091c\u093e\u0924\u093f: OBC, \u0932\u093f\u0902\u0917: \u092a\u0941\u0930\u0941\u0937""",
        "no_schemes_found": "\u0915\u094b\u0908 \u092e\u093f\u0932\u093e\u0928 \u0935\u093e\u0932\u0940 \u092f\u094b\u091c\u0928\u093e \u0922\u0942\u0902\u0922\u0928\u0947 \u092e\u0947\u0902 \u0935\u093f\u092b\u0932\u0964 \u0906\u092f\u0941, \u0906\u092f, \u0935\u094d\u092f\u0935\u0938\u093e\u092f, \u0930\u093e\u091c\u094d\u092f \u0914\u0930 \u091c\u093e\u0924\u093f \u091c\u0948\u0938\u0940 \u0905\u0927\u093f\u0915 \u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u092a\u094d\u0930\u0926\u093e\u0928 \u0915\u0930\u0928\u0947 \u0915\u093e \u092a\u094d\u0930\u092f\u093e\u0938 \u0915\u0930\u0947\u0902\u0964",
        "no_additional_schemes": "\u0906\u092a\u0915\u0940 \u092a\u094d\u0930\u094b\u092b\u093e\u0907\u0932 \u0915\u0947 \u0932\u093f\u090f \u0915\u094b\u0908 \u0905\u0924\u093f\u0930\u093f\u0915\u094d\u0924 \u092e\u093f\u0932\u093e\u0928 \u0935\u093e\u0932\u0940 \u092f\u094b\u091c\u0928\u093e \u0928\u0939\u0940\u0902 \u092e\u093f\u0932\u0940\u0964",
        "ask_schemes_first": "\u0915\u0943\u092a\u092f\u093e \u092a\u0939\u0932\u0947 \u0915\u0941\u091b \u092f\u094b\u091c\u0928\u093e\u0913\u0902 \u0915\u0947 \u092c\u093e\u0930\u0947 \u092e\u0947\u0902 \u092a\u0942\u091b\u0947\u0902, \u092b\u093f\u0930 \u092e\u0948\u0902 \u0909\u0928\u0915\u0947 \u0932\u093f\u090f \u0906\u092a\u0915\u0940 \u092a\u093e\u0924\u094d\u0930\u0924\u093e \u0915\u0940 \u091c\u093e\u0902\u091a \u0915\u0930 \u0938\u0915\u0924\u093e \u0939\u0942\u0901\u0964",
    },
    "gu": {
        "profile_request": """\u0a9a\u0acb\u0a95\u0acd\u0a95\u0ab8! \u0a95\u0ac3\u0aaa\u0abe \u0a95\u0ab0\u0ac0\u0aa8\u0ac7 \u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0aaa\u0abe\u0aa4\u0acd\u0ab0\u0aa4\u0abe \u0aa4\u0aaa\u0abe\u0ab8\u0ab5\u0abe \u0aae\u0abe\u0a9f\u0ac7 \u0ab5\u0abf\u0a97\u0aa4\u0acb \u0ab6\u0ac7\u0ab0 \u0a95\u0ab0\u0acb:

  \u2022 \u0a89\u0a82\u0aae\u0ab0 (Age)
  \u2022 \u0ab5\u0abe\u0ab0\u0acd\u0ab7\u0abf\u0a95 \u0a86\u0ab5\u0a95 (\u0aa6\u0abe.\u0aa4. 1.5 \u0ab2\u0abe\u0a96, 50,000)
  \u2022 \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf (\u0aa6\u0abe.\u0aa4. \u0ab5\u0abf\u0aa6\u0acd\u0aaf\u0abe\u0ab0\u0acd\u0aa5\u0ac0, \u0a96\u0ac7\u0aa1\u0ac2\u0aa4, \u0ab8\u0acd\u0ab5-\u0ab0\u0acb\u0a9c\u0a97\u0abe\u0ab0)
  \u2022 \u0ab0\u0abe\u0a9c\u0acd\u0aaf (\u0aa6\u0abe.\u0aa4. \u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4)
  \u2022 \u0ab2\u0ab2\u0abf\u0a82\u0a97 (\u0aaa\u0ac1\u0ab0\u0ac1\u0ab7 / \u0aae\u0ab9\u0abf\u0ab2\u0abe)
  \u2022 \u0a9c\u0acd\u0a9e\u0abe\u0aa4\u0abf/\u0ab6\u0acd\u0ab0\u0ac7\u0aa3\u0ac0 (SC / ST / OBC / General / EWS / SEBC / NT / DNT / Minority)

\u0a89\u0aa6\u0abe\u0ab9\u0ab0\u0aa3:
  \u0a89\u0a82\u0aae\u0ab0: 22, \u0a86\u0ab5\u0a95: 1.5 \u0ab2\u0abe\u0a96, \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf: \u0ab5\u0abf\u0aa6\u0acd\u0aaf\u0abe\u0ab0\u0acd\u0aa5\u0ac0, \u0ab0\u0abe\u0a9c\u0acd\u0aaf: \u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4, \u0a9c\u0acd\u0a9e\u0abe\u0aa4\u0abf: OBC, \u0ab2\u0abf\u0a82\u0a97: \u0aaa\u0ac1\u0ab0\u0ac1\u0ab7""",
        "no_schemes_found": "\u0a95\u0acb\u0a88 \u0aae\u0ac7\u0ab3 \u0a96\u0abe\u0aa4\u0ac0 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93 \u0aae\u0ab3\u0ac0 \u0aa8\u0aa5\u0ac0\u0964 \u0a89\u0a82\u0aae\u0ab0, \u0a86\u0ab5\u0a95, \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf, \u0ab0\u0abe\u0a9c\u0acd\u0aaf \u0a85\u0aa8\u0ac7 \u0a9c\u0acd\u0a9e\u0abe\u0aa4\u0abf \u0a9c\u0ac7\u0ab5\u0ac0 \u0ab5\u0abf\u0a97\u0aa4\u0acb \u0a86\u0aaa\u0ab5\u0abe\u0aa8\u0acb \u0aaa\u0acd\u0ab0\u0aaf\u0aa4\u0acd\u0aa8 \u0a95\u0ab0\u0acb\u0964",
        "no_additional_schemes": "\u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0aaa\u0acd\u0ab0\u0acb\u0aab\u0abe\u0a87\u0ab2 \u0aae\u0abe\u0a9f\u0ac7 \u0a95\u0acb\u0a88 \u0ab5\u0aa7\u0abe\u0ab0\u0abe\u0aa8\u0ac0 \u0aaf\u0acb\u0a9c\u0aa8\u0abe \u0aae\u0ab3\u0ac0 \u0aa8\u0aa5\u0ac0\u0964",
        "ask_schemes_first": "\u0a95\u0ac3\u0aaa\u0abe \u0a95\u0ab0\u0ac0\u0aa8\u0ac7 \u0aaa\u0ab9\u0ac7\u0ab2\u0abe \u0a95\u0ac7\u0a9f\u0ab2\u0ac0\u0a95 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93 \u0ab5\u0abf\u0ab6\u0ac7 \u0aaa\u0ac2\u0a9b\u0acb, \u0aaa\u0a9b\u0ac0 \u0ab9\u0ac1\u0a82 \u0aa4\u0ac7 \u0aae\u0abe\u0a9f\u0ac7 \u0aa4\u0aae\u0abe\u0ab0\u0ac0 \u0aaa\u0abe\u0aa4\u0acd\u0ab0\u0aa4\u0abe \u0aa4\u0aaa\u0abe\u0ab8\u0ac0 \u0ab6\u0a95\u0ac1\u0a82 \u0a9b\u0ac1\u0a82\u0964",
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
        "\u0a96\u0ac7\u0aa1\u0ac2\u0aa4\u0acb \u0aae\u0abe\u0a9f\u0ac7 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93 \u1f33e": "Schemes for farmers",
        "\u0aae\u0ab9\u0abf\u0ab2\u0abe \u0a95\u0ab2\u0acd\u0aaf\u0abe\u0aa3 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93": "Women welfare schemes",
        "\u0ab6\u0abf\u0a95\u0acd\u0ab7\u0aa3 \u0ab6\u0abf\u0ab7\u0acd\u0aaf\u0ab5\u0ac3\u0aa4\u0acd\u0aa4\u0abf": "Education scholarships",
        "\u0a86\u0ab0\u0acb\u0a97\u0acd\u0aaf \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93": "Healthcare schemes",
        "\u0a86\u0ab5\u0abe\u0ab8 \u0aaf\u0acb\u0a9c\u0aa8\u0abe": "housing scheme",
        "\u0aaf\u0ac1\u0ab5\u0abe\u0aa8\u0acb \u0aae\u0abe\u0a9f\u0ac7 \u0ab8\u0acd\u0a9f\u0abe\u0ab0\u0acd\u0a9f\u0a85\u0aaa \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93": "Startup schemes for youth",
        "\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0aae\u0abe\u0a82 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a93": "Schemes in Gujarat",
        "\u0a95\u0acc\u0ab6\u0ab2\u0acd\u0aaf \u0ab5\u0abf\u0a95\u0abe\u0ab8 \u0a95\u0abe\u0ab0\u0acd\u0aaf\u0a95\u0acd\u0ab0\u0aae": "Skill development programs",
        
        "\u0915\u093f\u0938\u093e\u0928\u094b\u0902 \u0915\u0947 \u0932\u093f\u090f \u092f\u094b\u091c\u0928\u093e\u090f\u0902 \u1f33e": "Schemes for farmers",
        "\u092e\u0939\u093f\u0932\u093e \u0915\u0932\u094d\u092f\u093e\u0923 \u092f\u094b\u091c\u0928\u093e\u090f\u0902": "Women welfare schemes",
        "\u0936\u093f\u0915\u094d\u0937\u093e \u091b\u093e\u0924\u094d\u0930\u0935\u0943\u0924\u094d\u0924\u093f": "Education scholarships",
        "\u0938\u094d\u0935\u093e\u0938\u094d\u0925\u094d\u092f \u092f\u094b\u091c\u0928\u093e\u090f\u0902": "Healthcare schemes",
        "\u0906\u0935\u093e\u0938 \u092f\u094b\u091c\u0928\u093e": "housing scheme",
        "\u092f\u0941\u0935\u093e\u0913\u0902 \u0915\u0947 \u0932\u093f\u090f \u0938\u094d\u091f\u093e\u0930\u094d\u091f\u0905\u092a \u092f\u094b\u091c\u0928\u093e\u090f\u0902": "Startup schemes for youth",
        "\u0917\u0941\u091c\u0930\u093e\u0924 \u092e\u0947\u0902 \u092f\u094b\u091c\u0928\u093e\u090f\u0902": "Schemes in Gujarat",
        "\u0915\u094c\u0936\u0932 \u0935\u093f\u0915\u093e\u0938 \u0915\u093e\u0930\u094d\u092f\u0915\u094d\u0930\u092e": "Skill development programs"
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

    lang_name = {"hi": "Hindi (\u0939\u093f\u0902\u0926\u0940, Devanagari script)", "gu": "Gujarati (\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0, Gujarati script)"}[target_lang]
    script_warn = {
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (\u0939\u093f\u0902\u0926\u0940). Do NOT use Gujarati script.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0). Do NOT use Hindi/Devanagari script.",
    }[target_lang]
    r = get_llm().invoke(
        f"Translate this English text to {lang_name}.\n"
        f"{script_warn}\n"
        f"Keep unchanged: scheme names, official links, SC/ST/OBC/EWS/SEBC, state names, amounts, numbers.\n"
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
    keys_to_translate = [k for k in ("scheme_name", "description", "benefits", 
                                     "eligibility", "documents_required", 
                                     "application_process", "category") 
                         if k in d and d[k]]
    if not keys_to_translate:
        return d

    to_translate = {k: d[k] for k in keys_to_translate}
    json_in = json.dumps(to_translate, ensure_ascii=False)
    
    script_warn = {
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (\u0939\u093f\u0902\u0926\u0940). Do NOT output Gujarati.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0). Do NOT output Hindi.",
    }.get(target_lang, "")
    prompt = f"""Translate text values to {lang_name} while keeping numbers, Rs amounts, links, and SC/ST/OBC acronyms exactly the same.
CRITICAL: Preserve all original line breaks and structural patterns like numbered lists (1. 2. 3.) or bullet points (\u2022).
CLEAN UI RULE: Remove redundant link-directing filler phrases such as "click here", "click", "here", "visit link", or native versions from the output.
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

def translate_suggestions_batch(suggestions: list, target_lang: str) -> list:
    """Translate a list of suggestion objects (name + category) in bulk."""
    if target_lang == "en":
        return suggestions
    
    if not suggestions:
        return []

    lang_name = {"hi": "Hindi", "gu": "Gujarati"}.get(target_lang, "English")
    
    # Format input for LLM
    items_to_translate = []
    for s in suggestions:
        items_to_translate.append({"name": s["name"], "category": s.get("category", "")})
    
    json_in = json.dumps(items_to_translate, ensure_ascii=False)
    
    script_warn = {
        "hi": "IMPORTANT: Use ONLY Hindi language with Devanagari script (\u0939\u093f\u0902\u0926\u0940). Do NOT output Gujarati.",
        "gu": "IMPORTANT: Use ONLY Gujarati language with Gujarati script (\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0). Do NOT output Hindi.",
    }.get(target_lang, "")

    prompt = f"""Translate these government scheme names and categories to {lang_name}.
IMPORTANT: Use the localized script ({'Devanagari for Hindi' if target_lang == 'hi' else 'Gujarati script for Gujarati'}).
Even if the scheme name is a proper noun, it should be TRANSLITERATED into the target script so it is readable in that language. 
Example (Hindi): "Mukhyamantri Mahila" -> "मुख्यमंत्री महिला"
Example (Gujarati): "Mukhyamantri Mahila" -> "મુખ્યમંત્રી મહિલા"

Keep SC/ST/OBC/EWS/SEBC exactly the same.
{script_warn}
Return ONLY a JSON object with a "suggestions" key containing the list of translated objects.

Input JSON:
{json_in}
"""
    try:
        translated_list = get_llm().with_structured_output(SuggestionListOutput).invoke(prompt)
        results = []
        for i, item in enumerate(translated_list.suggestions):
            # Include en_name for bilingual filtering
            results.append({
                "name": item.name,
                "category": item.category,
                "en_name": suggestions[i]["name"]
            })
        return results
    except Exception as e:
        print(f"[translate_suggestions_batch] Batch translation failed: {e}")
        # Fallback to original with en_name
        return [{"name": s["name"], "category": s.get("category"), "en_name": s["name"]} for s in suggestions]
