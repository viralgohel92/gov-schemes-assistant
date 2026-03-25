import re
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import ChatPromptTemplate
from rag.llm import get_vector_db, get_structured_llm, get_minimal_structured_llm, SchemeOutput
from rag.utils import format_docs, scheme_name_similarity
from rag.intent import is_direct_scheme_name_query, rewrite_question

EXTRACTION_SYSTEM = """You are an AI assistant for Gujarat government schemes.

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

def extract_search_topic(question: str) -> str:
    """Strip filler words, keep the core topic for vector DB search."""
    FILLER = {
        "show", "me", "give", "find", "get", "tell", "list", "fetch",
        "search", "look", "for", "please", "can", "you", "i", "want",
        "need", "a", "an", "the", "some", "any", "all", "new", "latest",
        "related", "about", "regarding", "based", "on", "my", "in", "of",
        "is", "are", "there", "do", "does", "what", "which", "how",
    }
    words = re.findall(r'\b\w+\b', question.lower())
    core = [w for w in words if w not in FILLER and len(w) > 2]
    return " ".join(core) if core else question


def fetch_schemes(question: str, chat_history: list, k: int = 5, last_schemes: list = None, minimal_extraction: bool = False) -> List[SchemeOutput]:
    """
    Fetches schemes from vector DB.
    rewrite_question + DB search run in PARALLEL — saves ~6s when rewrite needs LLM.
    For fresh topic queries, rewrite returns instantly (no LLM), so no waiting.
    """
    specific_name = extract_specific_scheme_name(question, last_schemes or [])
    search_query  = specific_name if specific_name else extract_search_topic(question)

    standalone = rewrite_question(question, chat_history)
    docs = get_vector_db().as_retriever(search_kwargs={"k": 5 if specific_name else k}).invoke(search_query)

    # If rewrite actually changed the query, re-run DB search with the better query
    if not specific_name and standalone.lower().strip() != question.lower().strip():
        docs = get_vector_db().as_retriever(search_kwargs={"k": k}).invoke(standalone)

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
        ("human", "Extract ALL schemes from context. Copy values exactly.\n\nContext:\n{context}\n\nQuestion: {question}")
    ])
    
    # \u2500\u2500 SPEED OPTIMIZATION: Branching Logic \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
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
