import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from rag.llm import get_vector_db, get_structured_llm, get_minimal_structured_llm, SchemeOutput
from rag.utils import format_docs, scheme_name_similarity
from rag.intent import is_direct_scheme_name_query, rewrite_question

EXTRACTION_SYSTEM = """You are a precise data extractor for government schemes. 
Your ONLY task is to extract exact scheme names from the provided context.

CRITICAL RULES:
1. ONLY extract schemes that are explicitly listed as a value after a "Scheme name:" label or similar clear identifier.
2. NEVER extract the labels themselves (e.g., do NOT extract "Scheme name", "Description", "Benefits", "Eligibility" as scheme names).
3. If no specific scheme is found in the context that matches the user's intent, return an empty list.
4. Do NOT invent or hallucinate scheme names.
5. Do NOT return the user's question or topic as a scheme name.
6. Return the results in the required JSON format.

RULES FOR FULL EXTRACTION:
- For 'application_process', ALWAYS refactor messy blocks into a clean, numbered list (1, 2, 3...) with one step per line.
- For 'benefits' and 'eligibility', use clear bullet points ( ) if there are multiple distinct points.
- Remove redundant filler phrases like "click here", "click", "visit link", or "here" from all fields.
- Copy values from the context but ensure they are STRUCTURED for readability.
- Keep scheme names, state names, and acronyms like SC/ST/OBC exactly as found.
- Only use 'Not Available' if truly absent.
"""


def extract_specific_scheme_name(question: str, last_schemes: list) -> str | None:
    q_lower = question.lower()
    for scheme in last_schemes:
        name = scheme.scheme_name if hasattr(scheme, "scheme_name") else scheme.get("scheme_name", "")
        if name and name.lower() in q_lower:
            return name

    patterns = [
        r"(?:full details of|details of|tell me about|info(?:rmation)? (?:about|on)|explain|describe|what is)\s+(.+?)(?:\s*\??\s*$)",
        r"^(.{10,}?)\s+(?:scheme)?\s*(?:give me|show me|details|full details|information)",
    ]
    for pat in patterns:
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip(" .?")
            if len(candidate) > 8:
                return candidate

    if is_direct_scheme_name_query(question):
        candidate = question.strip(" .?")
        if len(candidate) > 8:
            # Clean common fillers from beginning/end
            candidate = re.sub(r'^(the|a|an|show|detail|details|info of)\s+', '', candidate, flags=re.IGNORECASE)
            candidate = re.sub(r'\s+(scheme|yojana|yojna)$', '', candidate, flags=re.IGNORECASE)
            return candidate.strip()

    return None


# Common search terms mapping for synonyms
SYNONYMS = {
    "housing": ["awas", "house", "home", "residential"],
    "farmer": ["khedut", "agriculture", "kisan", "crop", "farm"],
    "student": ["vidhyarthi", "scholarship", "education", "school", "college"],
    "health": ["medical", "aarogya", "hospital", "medicine", "treatment"],
    "woman": ["mahila", "lady", "female", "girl"],
    "disability": ["divyang", "handicap", "disabled"],
    "employment": ["job", "rozgaar", "career", "skill"],
    "poultry": ["chicken", "bird", "egg", "murgapalan", "hen", "chick", "hatchery"],
    "animal": ["pashupalan", "cattle", "cow", "buffalo", "veterinary", "livestock", "milk", "dairy"],
    "loan": ["credit", "subsidy", "financial assistance", "sahay"],
    "marriage": ["vivah", "lagana", "shadi", "kanyadan"],
}

# High-accuracy mapping for 'Quick Start' / suggestion box chips
# Keys should be lowercase and without emojis for consistent matching
CATEGORICAL_QUICK_START = {
    "schemes for farmers": {"topic": "Agriculture", "extra_keywords": ["Farmer", "Khedut", "Kisan", "Crop", "Agriculture"]},
    "women welfare schemes": {"topic": "Women", "extra_keywords": ["Women", "Mahila", "Girl", "Lady", "Child"]},
    "education scholarships": {"topic": "Education", "extra_keywords": ["Scholarship", "Student", "Vidhyarthi", "Learning", "Education"]},
    "healthcare schemes": {"topic": "Health", "extra_keywords": ["Health", "Medical", "Hospital", "Aarogya", "Sanitation"]},
    "housing scheme": {"topic": "Housing", "extra_keywords": ["Housing", "Awas", "Makaan", "Home"]},
    "startup schemes for youth": {"topic": "Business", "extra_keywords": ["Business", "Startup", "Entrepreneurship", "Employment", "Job", "Skill"]},
    "schemes in gujarat": {"topic": "Gujarat", "extra_keywords": ["Gujarat"]},
    "skill development programs": {"topic": "Skills", "extra_keywords": ["Skills", "Kaushal", "Training", "Employment"]},
}

def extract_search_topic(question: str) -> str:
    """Strip filler words, keep the core topic for vector DB search."""
    FILLER = {
        "show", "me", "give", "find", "get", "tell", "list", "fetch",
        "search", "look", "for", "please", "can", "you", "i", "want",
        "need", "a", "an", "the", "some", "any", "all", "new", "latest",
        "related", "about", "regarding", "based", "on", "my", "in", "of",
        "is", "are", "there", "do", "does", "what", "which", "how",
        "scheme", "schemes", "yojana", "yojna", "gujarat", "government", "govt",
        "assistant", "ai", "yojana-ai", "show", "list", "fetch", "find",
    }
    
    words = re.findall(r'\b\w+\b', question.lower())
    core = []
    for w in words:
        if w in FILLER: continue
        if len(w) <= 2: continue
        
        # Check if the word is a synonym for a major category
        found_category = False
        for cat, syns in SYNONYMS.items():
            if w == cat or w in syns:
                core.append(cat)
                found_category = True
                break
        
        if not found_category:
            core.append(w)
            
    return " ".join(core) if core else question


def _fetch_schemes_by_category(topic: str, keywords: list, k: int = 5) -> List[SchemeOutput]:
    """Search specifically for schemes matching a category topic or set of keywords."""
    from database.db import SessionLocal
    from database.models import Scheme
    from sqlalchemy import or_
    
    session = SessionLocal()
    try:
        filters = []
        # Match topic or keywords in category OR name
        for phrase in [topic] + keywords:
            pattern = f"%{phrase}%"
            filters.append(Scheme.category.ilike(pattern))
            filters.append(Scheme.scheme_name.ilike(pattern))
            
        schemes = session.query(Scheme).filter(or_(*filters)).limit(k).all()
        results = []
        for s in schemes:
            results.append(SchemeOutput(
                scheme_name=s.scheme_name,
                description=s.description or '',
                category=s.category or '',
                benefits=s.benefits or '',
                eligibility=s.eligibility or '',
                documents_required=s.documents_required or '',
                application_process=s.application_process or '',
                state=s.state or 'Gujarat',
                official_link=s.application_link or ''
            ))
        print(f"  [CATEGORICAL] SQL search for '{topic}' returned {len(results)} matches.")
        return results
    except Exception as e:
        print(f"  [CATEGORICAL] Error: {e}")
        return []
    finally:
        session.close()


def _sql_fallback_search(query: str, k: int = 5):
    """
    Fallback: search the relational 'schemes' table using ILIKE keywords
    when vector search returns empty (e.g. documents table is empty).
    Returns list of Document objects matching the format expected by the RAG pipeline.
    """
    from langchain_core.documents import Document
    try:
        from database.db import SessionLocal
        from database.models import Scheme
        from sqlalchemy import or_, and_, desc, case
        
        session = SessionLocal()
        
        # Identify "noise" words to avoid poisoning the SQL search
        NOISE = {"scheme", "schemes", "yojana", "yojna", "gujarat", "govt", "government", "development"}
        
        all_words = [w.strip() for w in query.lower().split() if len(w.strip()) > 2]
        meaningful_keywords = [w for w in all_words if w not in NOISE]
        
        # If we have no meaningful keywords, the search is too generic
        if not meaningful_keywords:
            print(f"  Query too generic: {query}. Returning empty.")
            return []

        # TIER 1: Match meaningful keywords (Expand synonyms as OR groups)
        filters = []
        for word in meaningful_keywords:
            # Build an OR group for this keyword and its synonyms
            syns = [word]
            for cat, s_list in SYNONYMS.items():
                if word == cat or word in s_list:
                    syns.extend(s_list)
                    if cat not in syns: syns.append(cat)
            
            word_filters = []
            for s_word in set(syns):
                pattern = f"%{s_word}%"
                word_filters.append(Scheme.scheme_name.ilike(pattern))
                word_filters.append(Scheme.category.ilike(pattern))
                word_filters.append(Scheme.description.ilike(pattern))
            
            filters.append(or_(*word_filters))
        
        # Build Case statement for basic ranking: Name matches are best
        rank_conditions = []
        for word in meaningful_keywords:
            syns = [word]
            for cat, s_list in SYNONYMS.items():
                if word == cat or word in s_list:
                    syns.extend(s_list)
                    if cat not in syns: syns.append(cat)
            
            for s_word in set(syns):
                rank_conditions.append((Scheme.scheme_name.ilike(f"%{s_word}%"), 10))
                rank_conditions.append((Scheme.category.ilike(f"%{s_word}%"), 5))

        # Build order_by using Case
        order_case = case(*rank_conditions, else_=0)
        
        # Try finding AND matches first (more specific)
        schemes = session.query(Scheme).filter(and_(*filters)).order_by(desc(order_case)).limit(k).all()
        
        # TIER 2: If no "AND" matches, fall back to simple "OR" on all keywords + synonyms
        if not schemes and len(meaningful_keywords) > 1:
            print(f"  No exact AND matches for {meaningful_keywords}. Trying OR on expanded set...")
            expanded_keywords = []
            for w in meaningful_keywords:
                expanded_keywords.append(w)
                for cat, syns in SYNONYMS.items():
                    if w == cat or w in syns:
                        expanded_keywords.extend(syns)
                        if cat not in expanded_keywords: expanded_keywords.append(cat)
            
            filters = []
            for kw in set(expanded_keywords):
                pattern = f"%{kw}%"
                filters.append(Scheme.scheme_name.ilike(pattern))
                filters.append(Scheme.category.ilike(pattern))
            
            schemes = session.query(Scheme).filter(or_(*filters)).order_by(desc(order_case)).limit(k).all()
        
        docs = []
        for s in schemes:
            text = (
                f"Scheme name: {s.scheme_name}\n"
                f"Description: {s.description or 'Not found'}\n"
                f"Category: {s.category or ''}\n"
                f"Benefits: {s.benefits or 'Not found'}\n"
                f"Eligibility: {s.eligibility or 'Not found'}\n"
                f"Application Process: {s.application_process or 'Not found'}\n"
                f"Documents Required: {s.documents_required or 'Not found'}\n"
                f"State: {s.state or 'Gujarat'}\n"
                f"Official Link: {s.application_link or ''}\n"
            )
            docs.append(Document(page_content=text, metadata={}))
        
        session.close()
        print(f"  SQL fallback returned {len(docs)} documents for: {query[:60]}")
        return docs
    except Exception as e:
        print(f"  SQL fallback error: {e}")
        return []

def _sql_exact_name_match(scheme_name: str) -> List[Document]:
    """
    Tries to find an EXACT or very close match in the SQL database.
    This bypasses vector search noise for known names.
    """
    from langchain_core.documents import Document
    try:
        from database.db import SessionLocal
        from database.models import Scheme
        
        # Clean the input name for SQL searching
        clean_name = scheme_name.strip().lower()
        # Remove common suffixes/prefixes that might not be in DB name
        clean_name = re.sub(r'^(the|any|all)\s+', '', clean_name)
        clean_name = re.sub(r'\s+(scheme|yojana|yojna)$', '', clean_name)
        
        session = SessionLocal()
        # Try finding exact match (case insensitive)
        s = session.query(Scheme).filter(Scheme.scheme_name.ilike(clean_name)).first()
        
        # If no exact match, try if the name is an exact substring of a scheme name
        if not s:
            s = session.query(Scheme).filter(Scheme.scheme_name.ilike(f"%{clean_name}%")).first()
            
        if s:
            # Check if it's a good match (to avoid matching "Matru" to "Matrushakti" if that's too broad)
            from rag.utils import scheme_name_similarity
            if scheme_name_similarity(clean_name, s.scheme_name) < 0.4:
                session.close()
                return []
                
            text = (

                f"Scheme name: {s.scheme_name}\n"
                f"Description: {s.description or 'Not found'}\n"
                f"Category: {s.category or ''}\n"
                f"Benefits: {s.benefits or 'Not found'}\n"
                f"Eligibility: {s.eligibility or 'Not found'}\n"
                f"Application Process: {s.application_process or 'Not found'}\n"
                f"Documents Required: {s.documents_required or 'Not found'}\n"
                f"State: {s.state or 'Gujarat'}\n"
                f"Official Link: {s.application_link or ''}\n"
            )
            print(f"  [ACCURACY] Exact SQL match found for: {scheme_name}")
            return [Document(page_content=text, metadata={})]
        
        session.close()
        return []
    except Exception as e:
        print(f"  SQL exact match error: {e}")
        return []




def fetch_schemes(question: str, chat_history: list, k: int = 5, last_schemes: list = None, minimal_extraction: bool = False) -> List[SchemeOutput]:
    """
    Fetches schemes from vector DB with SQL fallback.
    rewrite_question + DB search run in PARALLEL   saves ~6s when rewrite needs LLM.
    For fresh topic queries, rewrite returns instantly (no LLM), so no waiting.
    """
    specific_name = extract_specific_scheme_name(question, last_schemes or [])
    search_query  = specific_name if specific_name else extract_search_topic(question)
    standalone = rewrite_question(question, chat_history)

    docs = []
    
    # ACCURACY TIER 0: Special handling for Quick Start chips
    # Clean emojis and extra whitespace for robust matching
    clean_q = re.sub(r'[^\w\s]', '', question).lower().strip()
    clean_standalone = re.sub(r'[^\w\s]', '', standalone).lower().strip()
    
    if clean_q in CATEGORICAL_QUICK_START and clean_standalone == clean_q:
        config = CATEGORICAL_QUICK_START[clean_q]
        print(f"  [ACCURACY] Quick Start chip detected: {clean_q}. Fetching direct categorical matches...")
        cat_matches = _fetch_schemes_by_category(config['topic'], config['extra_keywords'], k=k)
        if cat_matches:
            return cat_matches

    def _do_search(q, limit):
        # Fallback logic: "bajri (millet)" -> try "bajri", then "millet"
        if "(" in q and ")" in q:
            first_q = re.sub(r'(\w+)\s*\((.*?)\)', r'\1', q)
            print(f"  Trying primary search topic: {first_q}")
            res = get_vector_db().as_retriever(search_kwargs={"k": limit}).invoke(first_q)
            if not res:
                second_q = re.sub(r'(\w+)\s*\((.*?)\)', r'\2', q)
                print(f"  No results. Trying fallback topic: {second_q}")
                res = get_vector_db().as_retriever(search_kwargs={"k": limit}).invoke(second_q)
            return res
        return get_vector_db().as_retriever(search_kwargs={"k": limit}).invoke(q)
    
    # ACCURACY TIER 1: If user provided a specific name, check SQL first!
    if specific_name:
        docs = _sql_exact_name_match(specific_name)
        
    # ACCURACY TIER 2: Vector Search
    if not docs:
        docs = _do_search(search_query, 5 if specific_name else k)

    # If rewrite actually changed the query, re-run DB search with the better query
    if not docs and not specific_name and standalone.lower().strip() != question.lower().strip():
        docs = _do_search(standalone, k)

    # ACCURACY TIER 3: SQL FALLBACK (Keyword search)
    # Trigger if vector search returned nothing OR if the results feel irrelevant
    context = format_docs(docs)
    if specific_name and docs:
        # Check if the context actually contains the requested scheme
        # We use a simple but effective check here
        similarity = max([scheme_name_similarity(specific_name, d.page_content.split('\n')[0].replace("Scheme name:", "").strip()) for d in docs])
        if similarity < 0.4:
            print(f"  [ACCURACY] Vector matches too weak (sim={similarity:.2f}). Triggering SQL fallback...")
            docs = []

    if not docs:
        print("   Vector search returned 0 docs or irrelevant docs. falling back to SQL keyword search...")
        fallback_query = specific_name or search_query or question
        docs = _sql_fallback_search(fallback_query, k)


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
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    #    SPEED OPTIMIZATION: Branching Logic                                           
    # If minimal_extraction=True, we use get_minimal_structured_llm().
    # This instructs the AI to ONLY find names, cutting generation time by 75%.
    if minimal_extraction:
        result = (prompt | get_minimal_structured_llm()).invoke({
            "context": context, "question": question, "chat_history": chat_history,
        })
    else:
        # Full extraction (takes longer) used only when user requests full details.
        result = (prompt | get_structured_llm()).invoke({
            "context": context, "question": question, "chat_history": chat_history,
        })

    if specific_name:
        if not result or not hasattr(result, 'schemes') or not result.schemes:
            return []

        
        # ACCURACY TIER 4: Final verification of similarity
        scored = sorted(
            result.schemes,
            key=lambda s: scheme_name_similarity(specific_name, s.scheme_name),
            reverse=True
        )
        best = scored[0]
        similarity = scheme_name_similarity(specific_name, best.scheme_name)
        print(f"  [ACCURACY] Final Similarity Score: {similarity:.2f} for '{best.scheme_name}'")
        
        # Relaxed threshold slightly to handle minor typos or translation variations
        if similarity >= 0.45:
            return [best]
            
        # If similarity is still too low, we return empty so the agent can apologize
        print(f"  [ACCURACY] Discarding result '{best.scheme_name}' due to low similarity threshold.")
        return []


    if not result or not hasattr(result, 'schemes'):
        return []

    return result.schemes


def fetch_random_schemes(k: int = 5) -> List[SchemeOutput]:
    """
    Fetches k random schemes from the database where state is Gujarat.
    Used for the 'Schemes in Gujarat' suggestion chip.
    """
    from database.db import SessionLocal
    from database.models import Scheme
    from sqlalchemy.sql import func
    
    session = SessionLocal()
    try:
        # Fetch random schemes specifically for Gujarat
        schemes = session.query(Scheme).filter(
            Scheme.state.ilike("%gujarat%")
        ).order_by(func.random()).limit(k).all()
        
        results = []
        for s in schemes:
            results.append(SchemeOutput(
                scheme_name=s.scheme_name,
                description=s.description or 'Not available',
                category=s.category or 'General',
                benefits=s.benefits or 'Not available',
                eligibility=s.eligibility or 'Not available',
                documents_required=s.documents_required or 'Not available',
                application_process=s.application_process or 'Not available',
                state=s.state or 'Gujarat',
                official_link=s.application_link or ''
            ))
        return results
    except Exception as e:
        print(f"[fetch_random_schemes] Error: {e}")
        return []
    finally:
        session.close()
