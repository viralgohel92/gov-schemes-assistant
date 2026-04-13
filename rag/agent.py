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
from rag.memory import get_session, save_to_history
from rag.intent import detect_intent, detect_field, is_fresh_search_request, extract_gender_from_question, merge_gender_into_profile, is_followup_on_previous, resolve_scheme_reference
from rag.retriever import fetch_schemes
from rag.eligibility import extract_user_profile, check_eligibility_for_schemes, fetch_eligible_schemes
from rag.web_enrichment import apply_visit_site_fallback

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
            "en": "Hello!   Welcome to Yojana AI \u2014 your Gujarat Government Scheme Assistant.\nI can help you find schemes, check eligibility, and get application details. How can I help you today?",
            "hi": "      !         AI                    \n                             ,                                                                                       ?",
            "gu": "      !         AI                    .\n                            ,                                                            .                                   ?",
        }
        yield greetings.get(lang, greetings["en"])
        return

    # \u2500\u2500 Scheme count 
    if intent == "scheme_count":
        count = get_total_scheme_count()
        counts = {
            "en": f"There are currently **{count} government schemes** available in our Gujarat scheme database. You can ask me to find schemes by category, occupation, or check which ones you're eligible for!",
            "hi": f"                                           **{count}               **                            ,                                                                  !",
            "gu": f"         r t yojan      b sam   h l **{count} sark r  y jan o** upalabdha ch . Tam  man   r   , vyavas ya dv r  y jan o   dhv  athav  tam  k v  p trat  dhar v  ch  t  thakvav m   p ch   ak  ch !",
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

    def reply_in_lang(text: str) -> str:
        return translate_response(text, lang)

    def ls(key: str) -> str:
        return get_string(key, lang)

    PROFILE_REQUEST = ls("profile_request")
    awaiting_profile = session.get("awaiting_profile", False)

    # \u2500\u2500 Sequential Step 1: Translation (MUST happen first)
    question_en = translate_to_english(question, detected)
    
    # 2. IMMEDIATE FEEDBACK: Tell the user we're working
    search_msg = { "en": "Searching...", "hi": "\u0916\u094b\u091c \u0930\u0939\u093e \u0939\u0942\u0901...", "gu": "\u0ab6\u0acb\u0aa7\u0ac0 \u0ab0\u0ab9\u0acd\u0aaf\u0acb \u0a9b\u0ac1\u0a82..." }.get(lang, "Searching...")
    # yield {"type": "conversational_start", "lang": lang}
    # yield {"type": "chunk", "text": f"*{search_msg}*"}

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
                yield {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results, "lang": lang}
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

        save_to_history(session_id, question, f"Found {len(eligible)} eligible schemes.")
        session["last_schemes"] = eligible
        yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}
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
            print("  Searching all schemes for your eligibility...")
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
            yield {"type": "eligibility_for_shown", "profile": profile.model_dump(), "schemes": results, "lang": lang}
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
        yield {"type": "eligibility_result", "profile": profile.model_dump(), "schemes": eligible, "lang": lang}
        return

    # \u2500\u2500 Normal scheme queries 
    session["awaiting_profile"] = False
    limit = parse_limit(question_en)

    followup = is_followup_on_previous(question_en, chat_history, session["last_schemes"])
    fresh = is_fresh_search_request(question_en)

    schemes = None

    # Step 1: If there are previously shown schemes, ALWAYS try to resolve against them first.
    # This is critical for Hindi/Gujarati where is_followup_on_previous may miss the match.
    if session["last_schemes"] and not fresh and intent != "names_only":
        resolved = resolve_scheme_reference(question, question_en, session["last_schemes"])
        if resolved is not None:
            # Successfully matched specific scheme(s) from the list
            converted = []
            for s in resolved:
                name = s.get("scheme_name", "") if isinstance(s, dict) else s.scheme_name
                if isinstance(s, dict) or isinstance(s, MinimalSchemeOutput):
                    fetched = fetch_schemes(name, [], k=3, last_schemes=[], minimal_extraction=(intent == "names_only"))
                    if fetched:
                        converted.append(fetched[0])
                    else:
                        converted.append(SchemeOutput(
                            scheme_name=name,
                            description="", category="",
                            benefits="", eligibility="",
                            documents_required="", application_process="",
                            state="", official_link=""
                        ))
                else:
                    converted.append(s)
            schemes = converted

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

    if intent == "names_only":
        # Filter out invalid or hallucinated names (LLM sometimes matches labels instead of values)
        invalid_names = {"scheme name", "not available", "not found", "none", "n/a", "unknown", "scheme"}
        selected = [s for s in schemes if s.scheme_name.lower().strip() not in invalid_names]
        selected = selected[:limit] if limit else selected
        
        if not selected:
            reply = reply_in_lang(ls("no_schemes_found"))
            save_to_history(session_id, question, reply)
            yield {"type": "conversational", "reply": reply, "lang": lang}
            return
            
        names_text = "\n".join(f"{i+1}. {s.scheme_name}" for i, s in enumerate(selected))
        names_text += "\n\n  Ask me for full details of any scheme above."
        reply = reply_in_lang(names_text)
        save_to_history(session_id, question, reply)

        # Stream names as text tokens (ChatGPT-style)
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
    dicts = [s.model_dump() for s in selected]

    preview_parts = []
    for i, d in enumerate(dicts):
        benefits_short = str(d.get("benefits", ""))[:150]
        preview_parts.append(f"**{i+1}. {d['scheme_name']}**\nBenefits: {benefits_short}...\n")
    
    preview_text = "Here are the details:\n\n" + "\n".join(preview_parts) + "\n*Loading cards...*"
    preview_text = reply_in_lang(preview_text)
    
    def _process_and_translate_scheme(d: dict) -> dict:
        d = apply_visit_site_fallback(d)
        if lang != "en":
            d = translate_scheme_dict(d, lang)
        return d

    # Run enrichment and translation in background WHILE streaming text
    with ThreadPoolExecutor(max_workers=max(1, min(len(dicts), 5))) as ex:
        futures = {ex.submit(_process_and_translate_scheme, d): i for i, d in enumerate(dicts)}
        
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