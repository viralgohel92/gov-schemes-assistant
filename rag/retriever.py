import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from rag.llm import get_vector_db, get_structured_llm, get_minimal_structured_llm, SchemeOutput
from rag.utils import format_docs, scheme_name_similarity
from rag.intent import is_direct_scheme_name_query, rewrite_question
from langchain_core.documents import Document


EXTRACTION_SYSTEM = """You are a helpful government scheme expert.
Your task is to identify and extract relevant schemes from the provided context that match the user's query or category.

RULES:
1. Extract scheme names that match the user's intent OR topic (e.g., if user asks for "women schemes", extract any schemes related to women, pregnancy, or girls).
2. ONLY extract schemes that are explicitly in the provided context.
3. For 'application_process', refactor messy blocks into a clean, numbered list (1, 2, 3...).
4. For 'benefits' and 'eligibility', use clear bullet points.
5. If truly no relevant schemes are found, return an empty list.
6. Return results in JSON format.
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
    "housing": ["awas", "house", "home", "residential", "housing", "ghar", "makan", "rehthan", "awas yojna", "shelter"],
    "farmer": ["khedut", "agriculture", "kisan", "crop", "farm", "farmers", "farming", "krushi", "khet", "rural"],
    "student": ["vidhyarthi", "scholarship", "education", "school", "college", "students", "shikshan", "shishyavrutti", "learning"],
    "health": ["medical", "aarogya", "hospital", "medicine", "treatment", "healthcare", "swasthya", "davakhanu", "wellness"],
    "woman": ["mahila", "lady", "female", "girl", "women", "ladies", "girls", "stree", "beheno", "child", "maternity"],
    "disability": ["divyang", "handicap", "disabled", "viklang"],
    "employment": ["job", "rozgaar", "career", "skill", "employment", "naukri", "kaushalya", "youth", "young"],
    "poultry": ["chicken", "bird", "egg", "murgapalan", "hen", "chick", "hatchery", "paxipalan"],
    "animal": ["pashupalan", "cattle", "cow", "buffalo", "veterinary", "livestock", "milk", "dairy", "maldhari"],
    "loan": ["credit", "subsidy", "financial assistance", "sahay", "loan", "rin", "loan sahay", "banking", "finance"],
    "marriage": ["vivah", "lagana", "shadi", "kanyadan", "marriage"],
    "business": ["startup", "entrepreneurship", "business", "self-employed", "venture", "industry", "msme", "shop"],
    "welfare": ["welfare", "empowerment", "social", "support", "assistance", "sahay", "utility", "sanitation"],
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


def _sql_fallback_search(query: str, k: int = 5):
    """
    Fallback: search the relational 'schemes' table using ILIKE keywords
    when vector search returns empty (e.g. documents table is empty).
    Returns list of Document objects matching the format expected by the RAG pipeline.
    """

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

        # TIER 1: Search for ALL synonyms of all keywords
        all_search_terms = set()
        for word in meaningful_keywords:
            all_search_terms.add(word)
            for cat, s_list in SYNONYMS.items():
                if word == cat or word in s_list:
                    all_search_terms.update(s_list)
                    all_search_terms.add(cat)
        
        # Build Case statement for ranking: Name matches are best
        rank_conditions = []
        or_filters = []
        for s_word in all_search_terms:
            pattern = f"%{s_word}%"
            # Filters
            or_filters.append(Scheme.scheme_name.ilike(pattern))
            or_filters.append(Scheme.category.ilike(pattern))
            or_filters.append(Scheme.description.ilike(pattern))
            # Ranking
            rank_conditions.append((Scheme.scheme_name.ilike(pattern), 50))
            rank_conditions.append((Scheme.category.ilike(pattern), 20))
            rank_conditions.append((Scheme.description.ilike(pattern), 5))

        # Build order_by using Case
        order_case = case(*rank_conditions, else_=0)
        
        # We always use OR search but order by match strength to avoid "no results"
        schemes = session.query(Scheme).filter(or_(*or_filters)).order_by(desc(order_case)).limit(k).all()
        
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

    try:
        from database.db import SessionLocal
        from database.models import Scheme
        
        # Clean the input name for SQL searching
        clean_name = scheme_name.strip().lower()
        # Remove common suffixes/prefixes that might not be in DB name
        clean_name = re.sub(r'^(the|any|all)\s+', '', clean_name)
        # Remove parentheses and everything inside them (e.g. "(Part 1)", "(Developing Case)")
        clean_name = re.sub(r'\(.*?\)', '', clean_name).strip()
        # Remove trailing words that look like they might be cut off (no vowels or very short)
        # Or common filler words
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

    docs = []
    
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
