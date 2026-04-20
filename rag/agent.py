from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from rag.llm import SchemeOutput, MinimalSchemeOutput, UserProfile, warmup, get_vector_db, get_llm
from rag.utils import parse_limit
from rag.translation import detect_language, translate_to_english, translate_response, get_string, translate_scheme_dict
from rag.memory import get_session, save_to_history, save_session
from rag.intent import detect_intent, detect_field, is_fresh_search_request, extract_gender_from_question, merge_gender_into_profile, is_followup_on_previous, resolve_scheme_reference
from rag.retriever import fetch_schemes, fetch_random_schemes
from rag.eligibility import extract_user_profile, check_eligibility_for_schemes, fetch_eligible_schemes
from rag.web_enrichment import apply_visit_site_fallback

def _translate_scheme_names(scheme_dicts, lang):
    """Helper to translate scheme names in a list of scheme dicts (for pills/cards)."""
    if lang == "en" or not scheme_dicts:
        return scheme_dicts
    try:
        from rag.translation import translate_suggestions_batch
        # Note: translate_suggestions_batch returns dicts with 'name', 'category', 'en_name'
        to_translate = [{"name": s.get("scheme_name", ""), "category": s.get("category", "")} for s in scheme_dicts]
        translated = translate_suggestions_batch(to_translate, lang)
        for i, t in enumerate(translated):
            # Update name and category. Keep other fields (benefits, etc.) as is.
            scheme_dicts[i]["scheme_name"] = t["name"]
            if t.get("category"):
                scheme_dicts[i]["category"] = t["category"]
    except Exception as e:
        print(f"[_translate_scheme_names] Error: {e}")
    return scheme_dicts


def get_total_scheme_count() -> int:
    """Count total schemes stored in the Vector DB or Database."""
    try:
        from database.db import SessionLocal
        from database.models import Scheme  # Assuming such a model exists for metadata
        # Fallback to counting from PG directly if vector DB count fails
        from sqlalchemy import text
        db = SessionLocal()
        count = db.execute(text("SELECT count(*) FROM schemes")).scalar()
        db.close()
        return count
    except Exception:
        try:
            # Fallback for Chroma
            return get_vector_db()._collection.count()
        except:
            return 0

def conversational_reply_stream(question: str, chat_history: list, lang: str = "en", intent: str = "conversational"):
    # \u2500\u2500 Greeting 
    if intent == "greeting":
        greetings = {
            "en": "Hello! Welcome to Yojana AI \u2014 your Gujarat Government Scheme Assistant.\nI can help you find schemes, check eligibility, and get application details. How can I help you today?",
            "hi": "\u0927\u0928\u094d\u092f\u0935\u093e\u0926! \u092f\u094b\u091c\u0928\u093e AI \u092e\u0947\u0902 \u0906\u092a\u0915\u093e \u0938\u094d\u0935\u093e\u0917\u0924 \u0939\u0948 \u2014 \u0906\u092a\u0915\u0947 \u0917\u0941\u091c\u0930\u093e\u0924 \u0938\u0930\u0915\u093e\u0930 \u0915\u0940 \u092f\u094b\u091c\u0928\u093e \u0938\u0939\u093e\u092f\u0915\u0964\n\u092e\u0948\u0902 \u092f\u094b\u091c\u0928\u093e\u0913\u0902 \u0915\u094b \u0916\u094b\u091c\u0928\u0947, \u092a\u093e\u0924\u094d\u0930\u0924\u093e \u0915\u0940 \u091c\u093e\u0902\u091a \u0915\u0930\u0928\u0947 \u0914\u0930 \u0906\u0935\u0947\u0926\u0928 \u0935\u093f\u0935\u0930\u0923 \u092a\u094d\u0930\u093e\u092a\u094d\u0924 \u0915\u0930\u0928\u0947 \u092e\u0947\u0902 \u0906\u092a\u0915\u0940 \u0938\u0939\u093e\u092f\u0924\u093e \u0915\u0930 \u0938\u0915\u0924\u093e \u0939\u0942\u0901\u0964 \u0906\u091c \u092e\u0948\u0902 \u0906\u092a\u0915\u0940 \u0915\u0948\u0938\u0947 \u092e\u0920\u0926 \u0915\u0930 \u0938\u0915\u0924\u093e \u0939\u0942\u0901?",
            "gu": "\u0aa8\u0aae\u0ab8\u0acd\u0aa4\u0ac7! \u0aaf\u0acb\u0a9c\u0aa8\u0abe AI \u0aae\u0abe\u0a82 \u0aa4\u0aae\u0abe\u0ab0\u0ac1\u0a82 \u0ab8\u0acd\u0ab5\u0abe\u0a97\u0aa4 \u0a9b\u0ac7 \u2014 \u0aa4\u0aae\u0abe\u0ab0\u0abe \u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4 \u0ab8\u0ab0\u0a95\u0abe\u0ab0\u0aa8\u0abe \u0aaf\u0acb\u0a9c\u0aa8\u0abe \u0ab8\u0ab9\u0abe\u0aaf\u0a95\u0aed\n\u0ab9\u0ac1\u0a82 \u0aa4\u0aae\u0aa8\u0ac7 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a83 \u0ab6\u0acb\u0aa7\u0ab5\u0abe\u0aae\u0abe\u0a82, \u0aaa\u0abe\u0aa4\u0acd\u0ab0\u0aa4\u0abe \u0aa4\u0aaa\u0abe\u0ab8\u0ab5\u0abe\u0aae\u0abe\u0a82 \u0a85\u0aa8\u0ac7 \u0a85\u0ab0\u0a9c\u0ac0\u0aa8\u0ac0 \u0ab5\u0abf\u0a97\u0aa4\u0acb \u0aae\u0ac7\u0ab2\u0ab5\u0ab5\u0abe\u0aae\u0abe\u0a82 \u0aae\u0aa6\u0aa6 \u0a95\u0ab0\u0ac0 \u0ab6\u0a95\u0ac1\u0a82 \u0a9b\u0ac1\u0a82\u0aed \u0a86\u0a9c\u0ac7 \u0ab9\u0ac1\u0a82 \u0aa4\u0aae\u0aa8\u0ac7 \u0a95\u0ac7\u0ab5\u0ac0 \u0ab0\u0ac0\u0aa4\u0ac7 \u0aae\u0aa6\u0aa6 \u0a95\u0ab0\u0ac0 \u0ab6\u0a95\u0ac1\u0a82 \u0a9b\u0ac1\u0a82?",
        }
        yield greetings.get(lang, greetings["en"])
        return

    # \u2500\u2500 Scheme count 
    if intent == "scheme_count":
        count = get_total_scheme_count()
        counts = {
            "en": f"There are currently **{count} government schemes** available in our Gujarat scheme database. You can ask me to find schemes by category, occupation, or check which ones you're eligible for!",
            "hi": f"\u0935\u0930\u094d\u0924\u092e\u093e\u0928 \u092e\u0947\u0902 \u0939\u092e\u093e\u0930\u0947 \u0917\u0941\u091c\u0930\u093e\u0924 \u092f\u094b\u091c\u0928\u093e \u0921\u0947\u091f\u093e\u092c\u0947\u0938 \u092e\u0947\u0902 **{count} \u0938\u0930\u0915\u093e\u0930\u0940 \u092f\u094b\u091c\u0928\u093e\u0902\u090f\u0902** \u0909\u092a\u0932\u092c\u094d\u0927 \u0939\u0948\u0902\u0964 \u0906\u092a \u092e\u0941\u091d\u0938\u0947 \u0936\u094d\u0930\u0947\u0923\u0940, \u0935\u094d\u092f\u0935\u0938\u093e\u092f \u0915\u0947 \u0906\u0927\u093e\u0930 \u092a\u0930 \u092f\u094b\u091c\u0928\u093e\u0902\u090f\u0902 \u0916\u094b\u091c\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0915\u0939 \u0938\u0915\u0924\u0947 \u0939\u0948\u0902, \u092f\u093e \u091c\u093e\u0902\u091a \u0938\u0915\u0924\u0947 \u0939\u0948\u0902 \u0915\u093f \u0906\u092a \u0915\u093f\u0928 \u092f\u094b\u091c\u0928\u093e\u0913\u0902 \u0915\u0947 \u0932\u093f\u090f \u092a\u093e\u0924\u094d\u0930 \u0939\u0948\u0902!",
            "gu": f"\u0ab9\u0abe\u0ab2\u0aae\u0abe\u0a82 \u0a85\u0aae\u0abe\u0ab0\u0abe \u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4 \u0aaf\u0acb\u0a9c\u0aa8\u0abe \u0aa1\u0ac7\u0a9f\u0abe\u0aac\u0ac7\u0a9d\u0aae\u0abe\u0a82 **{count} \u0ab8\u0ab0\u0a95\u0abe\u0ab0\u0ac0 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a83\u0aed** \u0a89\u0aaa\u0ab2\u0aac\u0acd\u0aa7 \u0a9b\u0ac7\u0aed \u0aa4\u0aae\u0ac7 \u0aae\u0aa8\u0ac7 \u0ab6\u0acd\u0ab0\u0ac7\u0aa3\u0ac0, \u0ab5\u0acd\u0aaf\u0ab5\u0ab8\u0abe\u0aaf \u0aa6\u0acd\u0ab5\u0abe\u0ab0\u0abe \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a83 \u0ab6\u0acb\u0aa7\u0ab5\u0abe \u0aae\u0abe\u0a9f\u0ac7 \u0a95\u0ab9\u0ac0 \u0ab6\u0a95\u0acb \u0a9b\u0acb, \u0a85\u0aa5\u0ab5\u0abe \u0aa4\u0aaa\u0abe\u0ab8\u0ac0 \u0ab6\u0a95\u0acb \u0a9b\u0acb \u0a95\u0ac7 \u0aa4\u0aae\u0ac7 \u0a95\u0a88 \u0aaf\u0acb\u0a9c\u0aa8\u0abe\u0a83 \u0aae\u0abe\u0a9f\u0ac7 \u0aaa\u0abe\u0aa4\u0acd\u0ab0 \u0a9b\u0acb!"
        }
        yield counts.get(lang, counts["en"])
        return

    # \u2500\u2500 General conversational 
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    lang_instruction = {
        "hi": "Always reply in Hindi (Devanagari script).",
        "gu": "Always reply in Gujarati (Gujarati script).",
        "en": "Reply in English.",
    }.get(lang, "Reply in English.")
    prompt = f"""You are Yojana AI, the official Gujarat government scheme assistant. 
{lang_instruction}

Guidelines:
1. Provide accurate information about government schemes based on the database.
2. If the user asks for a scheme that is currently being updated or lacks full details, explain that details are being fetched from the official portal and will be available shortly.
3. NEVER make up eligibility criteria or benefits (hallucinate).
4. If you don't know the answer, politely say so.

Conversation:
{history_text}

User: {question}
AI:"""
    for chunk in get_llm().stream(prompt):
        yield chunk.content

# -------------------------------------------------
# Main ask function
# -------------------------------------------------

def ask_agent(question: str, session_id: str = "user_1", ui_lang: str = None, user_context: dict = None):
    session = get_session(session_id)
    chat_history = session["history"]
    awaiting_profile = session.get("awaiting_profile", False)

    #    Initialize/Update Profile from User Context (if logged in)           
    if user_context:
        current_profile = session.get("user_profile") or UserProfile()
        updated_data = current_profile.model_dump()
        for k, v in user_context.items():
            if v and not updated_data.get(k):
                updated_data[k] = v
        session["user_profile"] = UserProfile(**updated_data)
        if not getattr(session["user_profile"], 'state', None):
            session["user_profile"].state = "Gujarat"
        save_session(session_id, session)

    #    Language detection   
    detected = detect_language(question)
    
    # Priority 1: UI selection
    if ui_lang and ui_lang in ("en", "hi", "gu"):
        lang = ui_lang
    # Priority 2: User input language
    elif detected != "en":
        lang = detected
    else:
        lang = session.get("lang", "en")
    
    session["lang"] = lang
    save_session(session_id, session)
    print(f"[DEBUG] ask_agent called with question: {question[:100]}")

    def reply_in_lang(text: str) -> str:
        return translate_response(text, lang)

    def ls(key: str) -> str:
        return get_string(key, lang)

    def _process_and_translate_scheme(d: dict) -> dict:
        d = apply_visit_site_fallback(d)
        if lang != "en":
            d = translate_scheme_dict(d, lang)
        return d

    PROFILE_REQUEST = ls("profile_request")
    awaiting_profile = session.get("awaiting_profile", False)

    # \u2500\u2500 Sequential Step 1: Translation (MUST happen first)
    question_en = translate_to_english(question, detected)
    
    # \u2500\u2500 Step 3: Parallelized Tasks (Intent, Profile, Field, Gender)
    # Once we have question_en, we can run all other analysis in parallel to save ~4-6 seconds.
    # Parallelize: detect_intent, extract_user_profile, detect_field, extract_gender_from_question
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_intent = ex.submit(detect_intent, question_en, chat_history, awaiting_profile)
        f_profile = ex.submit(extract_user_profile, question_en)
        f_gender = ex.submit(extract_gender_from_question, question_en)
        
        # Wait for those that are needed immediately
        intent = f_intent.result()
        profile_update = f_profile.result()
        gender_hint = f_gender.result()

    # \u2500\u2500 User provided their profile 
    if intent == "eligibility_check":
        profile = merge_gender_into_profile(profile_update, gender_hint)
        session["user_profile"] = profile
        session["awaiting_profile"] = False
        save_session(session_id, session)

        if awaiting_profile and session.get("last_schemes"):
            last = session["last_schemes"]
            scheme_objects = []
            for s in last:
                # Check for eligibility content directly
                has_eligibility = False
                if hasattr(s, 'eligibility') and s.eligibility:
                    has_eligibility = True
                elif isinstance(s, dict) and s.get('eligibility'):
                    has_eligibility = True
                
                if has_eligibility:
                    if isinstance(s, dict): scheme_objects.append(SchemeOutput(**s))
                    else: scheme_objects.append(s)
                else:
                    name = s.scheme_name if hasattr(s, 'scheme_name') else s.get('scheme_name')
                    if name:
                        fetched = fetch_schemes(name, [], k=1, last_schemes=[])
                        if fetched: scheme_objects.append(fetched[0])
            
            if scheme_objects:
                print("  Checking eligibility for shown schemes...")
                results = check_eligibility_for_schemes(profile, scheme_objects)
                save_to_history(session_id, question, f"Checked eligibility for {len(scheme_objects)} schemes.")
                yield {"type": "conversational_end"}
                yield {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": _translate_scheme_names(results, lang), "lang": lang}
                return


        # No prior shown schemes OR scheme_objects came out empty   search full DB
        print("  No prior schemes found \u2014 searching full DB for eligibility...")
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

        # Stream eligibility results
        yield {"type": "eligibility_start", "profile": profile.model_dump(), "lang": lang}
        
        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        session["last_schemes"] = eligible
        
        dicts = [s.model_dump() for s in eligible]
        results = [None] * len(dicts)
        
        with ThreadPoolExecutor(max_workers=max(1, min(len(dicts), 5))) as ex:
            futures = {ex.submit(_process_and_translate_scheme, d): i for i, d in enumerate(dicts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res_dict = future.result()
                except Exception:
                    res_dict = dicts[idx]
                results[idx] = res_dict
                yield {"type": "scheme_card", "scheme": res_dict, "index": idx + 1, "lang": lang}
                
        yield {"type": "schemes_end", "schemes": results, "lang": lang}
        return

    # \u2500\u2500 User asks eligibility for previously shown schemes 
    if intent == "eligibility_for_shown":
        last_schemes = session.get("last_schemes", [])

        #   GUARD: If no schemes have been shown yet, treat this as a scheme search first.
        # The user should see schemes before being asked for their profile.
        if not last_schemes:
            print("    eligibility_for_shown but no schemes shown yet \u2014 fetching schemes first...")
            schemes = fetch_schemes(question_en, chat_history, k=5, last_schemes=[])
            session["last_schemes"] = schemes
            if schemes:
                selected = schemes[:5]
                save_to_history(session_id, question, f"Showed schemes: {', '.join(s.scheme_name for s in selected)}")
                
                yield {"type": "schemes_start", "lang": lang}
                
                dicts = [s.model_dump() for s in selected]
                results = [None] * len(dicts)
                with ThreadPoolExecutor(max_workers=min(len(dicts), 5)) as ex:
                    futures = {ex.submit(_process_and_translate_scheme, d): i for i, d in enumerate(dicts)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        try:
                            res_dict = future.result()
                        except Exception:
                            res_dict = dicts[idx]
                        results[idx] = res_dict
                        yield {"type": "scheme_card", "scheme": res_dict, "index": idx + 1, "lang": lang}
                
                yield {"type": "schemes_end", "schemes": results, "lang": lang}
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
            yield {"type": "conversational_end"}
            yield {"type": "conversational", "reply": PROFILE_REQUEST, "lang": lang}
            return

        profile = session["user_profile"]

        if is_fresh_search_request(question_en):
            profile = merge_gender_into_profile(profile, gender_hint)
            session["user_profile"] = profile
            print("  Searching all schemes for your eligibility...")
            eligible = fetch_eligible_schemes(profile, k=4)
            if not eligible:
                reply = reply_in_lang(ls("no_additional_schemes"))
                save_to_history(session_id, question, reply)
                yield {"type": "conversational", "reply": reply, "lang": lang}
                return
            save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
            session["last_schemes"] = eligible
            yield {"type": "conversational_end"}
            yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": _translate_scheme_names(eligible, lang), "lang": lang}
            return


        if last_schemes:
            print(f"  Found {len(last_schemes)} schemes in history. Converting to full detail...")
            
            full_schemes = []
            for s in last_schemes:
                # Be very robust: check if it has the required field for eligibility check
                has_eligibility = False
                if hasattr(s, 'eligibility') and s.eligibility:
                    has_eligibility = True
                elif isinstance(s, dict) and s.get('eligibility'):
                    has_eligibility = True
                
                if has_eligibility and (isinstance(s, SchemeOutput) or (isinstance(s, dict) and 'scheme_name' in s)):
                    # If it's a dict but has full details, convert it to SchemeOutput
                    if isinstance(s, dict):
                        full_schemes.append(SchemeOutput(**s))
                    else:
                        full_schemes.append(s)
                else:
                    # It's a MinimalSchemeOutput or a dict with just names -> Fetch full detail
                    name = s.scheme_name if hasattr(s, 'scheme_name') else s.get('scheme_name')
                    if name:
                        print(f"   -> Fetching full details for: {name}")
                        fetched = fetch_schemes(name, [], k=1, last_schemes=[])
                        if fetched:
                            full_schemes.append(fetched[0])
            
            print(f"  Proceeding with eligibility check for {len(full_schemes)} full schemes.")
            results = check_eligibility_for_schemes(profile, full_schemes)
            save_to_history(session_id, question, f"Checked eligibility for {len(full_schemes)} schemes.")
            yield {"type": "conversational_end"}
            yield {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": _translate_scheme_names(results, lang), "lang": lang}

            return

        print("  Searching all schemes for your eligibility...")
        eligible = fetch_eligible_schemes(profile, k=4)
        if not eligible:
            reply = reply_in_lang(ls("no_schemes_found"))
            save_to_history(session_id, question, reply)
            yield {"type": "conversational", "reply": reply, "lang": lang}
            return
        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        session["last_schemes"] = eligible
        yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": _translate_scheme_names(eligible, lang), "lang": lang}

        return

    # \u2500\u2500 Sequential Step: Common analysis
    session["awaiting_profile"] = False
    limit = parse_limit(question_en)
    followup = is_followup_on_previous(question_en, chat_history, session.get("last_schemes", []))
    fresh = is_fresh_search_request(question_en)

    # \u2500\u2500 Special Handling: Random Gujarat Schemes (Suggestion Chip)
    if question_en.lower().strip() in ["schemes in gujarat", "scheme in gujarat"] and not followup:
        print("  Suggestion chip 'Schemes in Gujarat' detected. Fetching 5 random schemes...")
        schemes = fetch_random_schemes(k=5)
        if schemes:
            session["last_schemes"] = schemes
            
            # Format as a list of names (similar to names_only)
            # Send as names_only format for clickable pills
            reply = ls("found_schemes")
            scheme_dicts = [s.model_dump() for s in schemes]
            yield {"type": "names_only", "reply": reply, "schemes": _translate_scheme_names(scheme_dicts, lang), "lang": lang}
            return



    schemes = None
    resolved = None

    # Step 1: If there are previously shown schemes, ALWAYS try to resolve against them first.
    # This is critical for Hindi/Gujarati where is_followup_on_previous may miss the match.
    # We also check if the question EXACTLY matches any last scheme name (pill click).
    exact_match = None
    if session.get("last_schemes"):
        for s in session["last_schemes"]:
            s_name = s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")
            if s_name and (question.lower().strip() == s_name.lower().strip() or question_en.lower().strip() == s_name.lower().strip()):
                exact_match = s
                break

    if (session["last_schemes"] and not fresh and intent != "names_only") or exact_match:
        if exact_match:
            resolved = [exact_match]
        else:
            resolved = resolve_scheme_reference(question, question_en, session["last_schemes"])
            
        if resolved is not None:
            # Successfully matched specific scheme(s) from the list
            converted = []
            for s in resolved:
                name = s.get("scheme_name", "") if isinstance(s, dict) else s.scheme_name
                # If we have an exact match OR a strong resolution, we WANT full details
                intent = "full_detail" 
                
                if isinstance(s, dict) or isinstance(s, MinimalSchemeOutput):
                    fetched = fetch_schemes(name, [], k=3, last_schemes=[], minimal_extraction=False)
                    if fetched:
                        converted.append(fetched[0])
                    else:
                        # Rebuild simple output if fetch failed
                        converted.append(SchemeOutput(
                            scheme_name=name,
                            description=s.get("description", "") if isinstance(s, dict) else "",
                            category=s.get("category", "") if isinstance(s, dict) else "",
                            benefits="", eligibility="",
                            documents_required="", application_process="",
                            state="", official_link=""
                        ))
                else:
                    converted.append(s)
            schemes = converted

    # Force "Names First" policy ONLY for new, non-specific searches
    if resolved is None and intent in ("full_detail", "specific_field") and not followup:
        # Check if it's a very specific long name query (from is_direct_scheme_name_query)
        # using a simple length check or keyword check
        from rag.intent import is_direct_scheme_name_query
        if not is_direct_scheme_name_query(question):
            print(f"  New search detected with intent {intent}. Forcing names_only first.")
            intent = "names_only"

    # Step 2: If resolution failed (None) or no last_schemes, do a fresh DB search 
    if schemes is None:
        prev_names = []
        if fresh:
            for s in session["last_schemes"]:
                name = s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")
                if name:
                    prev_names.append(name)
        base_k = 5
        fetch_k = max(limit or base_k, base_k) + len(prev_names)
        schemes = fetch_schemes(question_en, chat_history, k=fetch_k, last_schemes=session["last_schemes"], minimal_extraction=(intent == "names_only"))
        
        #    Prevent Hallucinations (Filtering junk responses)   
        invalid_keywords = {"scheme name", "not available", "not found", "n/a", "unknown", "scheme"}
        cleaned = []
        for s in schemes:
            s_name = (s.scheme_name if hasattr(s, "scheme_name") else s.get("scheme_name", "")).strip()
            if s_name and s_name.lower() not in invalid_keywords:
                cleaned.append(s)
        schemes = cleaned
        
        # Only filter out previously shown schemes if the user is doing a broad search (names_only)
        # or a fresh search for DIFFERENT schemes. Do NOT filter if they are asking for full_detail
        # of a scheme they just saw.
        if fresh and prev_names and intent not in ("full_detail", "specific_field"):
            schemes = [s for s in schemes if s.scheme_name not in prev_names]
        session["last_schemes"] = schemes
        session["last_limit"] = limit
        save_session(session_id, session)

    if intent == "names_only":
        # Filter out invalid or hallucinated names (LLM sometimes matches labels instead of values)
        invalid_names = {"scheme name", "not available", "not found", "none", "n/a", "unknown", "scheme"}
        selected = [s for s in schemes if s.scheme_name.lower().strip() not in invalid_names]
        selected = selected[:limit] if limit else selected
        
        if not selected:
            # Try to be more helpful: suggest categories based on keywords
            q_lower = question_en.lower()
            suggestions = []
            from rag.retriever import SYNONYMS
            for cat, syns in SYNONYMS.items():
                if cat in q_lower or any(s in q_lower for s in syns):
                    suggestions.append(cat.title())
            
            if suggestions:
                suggestion_text = ls("no_schemes_found") + "\n\n  " + \
                    f"I couldn't find a match for your exact query, but I have many schemes in the **{', '.join(suggestions)}** categories. Would you like to see those?"
                reply = reply_in_lang(suggestion_text)
            else:
                reply = reply_in_lang(ls("no_schemes_found"))
                
            save_to_history(session_id, question, reply)
            yield {"type": "conversational", "reply": reply, "lang": lang}
            return
            
        # Use a localized summary message instead of a raw numbered list
        reply = ls("found_schemes")

        # Stream names_only pills (YIELD START IMMEDIATELY)
        yield {"type": "names_only_start", "reply": reply, "lang": lang}
        
        save_to_history(session_id, question, reply)
        scheme_dicts = [s.model_dump() for s in selected]
        translated_schemes = _translate_scheme_names(scheme_dicts, lang)
        
        for i, s in enumerate(translated_schemes):
            yield {"type": "names_only_pill", "scheme": s, "index": i, "lang": lang}
        
        yield {"type": "names_only_end", "reply": reply, "schemes": translated_schemes, "lang": lang}
        return



    if intent == "specific_field":
        field = detect_field(question_en)
        lines = [f"  {s.scheme_name}:\n  {apply_visit_site_fallback(s.model_dump()).get(field, 'Not Available')}"
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

    # full_detail   Stream TEXT -> then render CARDS
    # full_detail   Stream individual CARDS one-by-one as they are ready
    # Filter out invalid or hallucinated names
    invalid_names = {"scheme name", "not available", "not found", "none", "n/a", "unknown", "scheme"}
    selected = [s for s in schemes if s.scheme_name.lower().strip() not in invalid_names]
    selected = selected[:limit] if limit else selected

    if not selected:
        reply = reply_in_lang(ls("no_schemes_found"))
        save_to_history(session_id, question, reply)
        yield {"type": "conversational", "reply": reply, "lang": lang}
        return

    save_to_history(session_id, question, f"Showed details for: {', '.join(s.scheme_name for s in selected)}")
    
    # Yield start event IMMEDIATELY before slow work
    yield {"type": "schemes_start", "lang": lang}
    
    dicts = [s.model_dump() for s in selected]

    # Run enrichment and translation in parallel and yield as they complete
    results = [None] * len(dicts)
    with ThreadPoolExecutor(max_workers=max(1, min(len(dicts), 5))) as ex:
        futures = {ex.submit(_process_and_translate_scheme, d): i for i, d in enumerate(dicts)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                res_dict = future.result()
            except Exception as e:
                print(f"[full_detail] Streaming error for scheme {idx}: {e}")
                res_dict = dicts[idx]
            
            results[idx] = res_dict
            # Yield individual card for true streaming
            yield {"type": "scheme_card", "scheme": res_dict, "index": idx + 1, "lang": lang}

    # Signal completion and provide full list for history saving
    yield {"type": "schemes_end", "schemes": results, "lang": lang}
    return