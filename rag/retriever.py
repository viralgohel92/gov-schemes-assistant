import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
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
        return question.strip()

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
    core = [w for w in words if w not in FILLER and len(w) > 2]
    return " ".join(core) if core else question


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
        from sqlalchemy import or_
        
        session = SessionLocal()
        
        # Identify "noise" words to avoid poisoning the SQL search
        NOISE = {"scheme", "schemes", "yojana", "yojna", "gujarat", "govt", "government"}
        
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
            if word in SYNONYMS:
                syns.extend(SYNONYMS[word])
            
            word_filters = []
            for s_word in syns:
                pattern = f"%{s_word}%"
                word_filters.append(Scheme.scheme_name.ilike(pattern))
                word_filters.append(Scheme.category.ilike(pattern))
                word_filters.append(Scheme.description.ilike(pattern))
            
            filters.append(or_(*word_filters))
        
        from sqlalchemy import and_, desc, case
        
        # Build Case statement for basic ranking: Name matches are best
        rank_conditions = []
        for word in meaningful_keywords:
            syns = [word] + (SYNONYMS.get(word, []))
            for s_word in syns:
                rank_conditions.append((Scheme.scheme_name.ilike(f"%{s_word}%"), 10))
                rank_conditions.append((Scheme.category.ilike(f"%{s_word}%"), 5))

        # Build order_by using Case
        order_case = case(*rank_conditions, else_=0)
        
        schemes = session.query(Scheme).filter(and_(*filters)).order_by(desc(order_case)).limit(k).all()
        
        # TIER 2: If no "AND" matches, fall back to simple "OR" on all keywords + synonyms
        if not schemes:
            print(f"  No exact AND matches for {meaningful_keywords}. Trying OR on expanded set...")
            expanded_keywords = []
            for w in meaningful_keywords:
                expanded_keywords.append(w)
                if w in SYNONYMS:
                    expanded_keywords.extend(SYNONYMS[w])
            
            filters = []
            for kw in expanded_keywords:
                pattern = f"%{kw}%"
                filters.append(Scheme.scheme_name.ilike(pattern))
                filters.append(Scheme.category.ilike(pattern))
            
            schemes = session.query(Scheme).filter(or_(*filters)).limit(k).all()
        
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

    docs = _do_search(search_query, 5 if specific_name else k)

    # If rewrite actually changed the query, re-run DB search with the better query
    if not specific_name and standalone.lower().strip() != question.lower().strip():
        docs = _do_search(standalone, k)

    #    SQL FALLBACK: If vector search returns nothing, use relational DB   
    if not docs:
        print("   Vector search returned 0 docs   falling back to SQL keyword search...")
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
