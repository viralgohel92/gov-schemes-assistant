from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from pydantic import BaseModel, Field
from typing import List
import re

# -------------------------------------------------
# Structured Output Schema
# -------------------------------------------------

class SchemeOutput(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    description: str = Field(description="Brief description of the scheme")
    category: str = Field(description="Category of the scheme")
    benefits: str = Field(description="Benefits provided under the scheme")
    eligibility: str = Field(description="Eligibility criteria for the scheme")
    documents_required: str = Field(description="Documents required to apply")
    application_process: str = Field(description="Steps to apply for the scheme")
    state: str = Field(description="State where the scheme is applicable")
    official_link: str = Field(description="Official website or link for the scheme")

class SchemesListOutput(BaseModel):
    schemes: List[SchemeOutput] = Field(description="List of all government schemes found in the context")

# -------------------------------------------------
# Embedding + Vector DB
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

# -------------------------------------------------
# LLMs
# -------------------------------------------------

llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
structured_llm = llm.with_structured_output(SchemesListOutput)

# -------------------------------------------------
# Memory
# -------------------------------------------------

store: dict = {}

def get_session(session_id: str) -> dict:
    if session_id not in store:
        store[session_id] = {"history": [], "last_schemes": [], "last_limit": None}
    return store[session_id]

def save_to_history(session_id: str, question: str, answer: str):
    session = get_session(session_id)
    session["history"].append(HumanMessage(content=question))
    session["history"].append(AIMessage(content=answer))

# -------------------------------------------------
# Parse requested number
# -------------------------------------------------

NUM_MAP = {"one":1,"two":2,"three":3,"four":4,"five":5,
           "six":6,"seven":7,"eight":8,"nine":9,"ten":10}

def parse_limit(question: str):
    q = question.lower()
    for word, num in NUM_MAP.items():
        if word in q:
            return num
    match = re.search(r'\b(\d+)\b', q)
    return int(match.group(1)) if match else None

# -------------------------------------------------
# Intent detection
# -------------------------------------------------

def detect_intent(question: str, chat_history: list) -> str:
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    intent_prompt = f"""
You are an intent classifier for a government scheme chatbot.

Conversation so far:
{history_text}

New user message: "{question}"

Classify the intent into EXACTLY one of these:
- names_only       → user wants only scheme name(s) or a list of names
- full_detail      → user wants complete details of scheme(s)
- specific_field   → user wants one specific field (eligibility, benefits, link, documents, etc.)
- conversational   → greeting, thanks, clarification, or non-scheme question

Reply with ONLY the intent label, nothing else.
"""
    return llm.invoke(intent_prompt).content.strip().lower()

# -------------------------------------------------
# Detect which field user wants
# -------------------------------------------------

def detect_field(question: str) -> str:
    field_prompt = f"""
The user asked: "{question}"

Which ONE field are they asking about? Reply with ONLY the field name from this list:
scheme_name, description, category, benefits, eligibility, documents_required, application_process, state, official_link

Reply with ONLY the field name.
"""
    return llm.invoke(field_prompt).content.strip().lower()

# -------------------------------------------------
# Check if question refers to previous schemes
# -------------------------------------------------

def is_followup_on_previous(question: str, chat_history: list) -> bool:
    if not chat_history:
        return False
    q = question.lower()
    followup_hints = ["first", "second", "third", "fourth", "fifth",
                      "1st", "2nd", "3rd", "4th", "5th",
                      "that", "it", "this", "above", "those",
                      "same", "previous", "last", "shown", "these"]
    return any(hint in q for hint in followup_hints)

# -------------------------------------------------
# Rewrite question using history
# -------------------------------------------------

def rewrite_question(question: str, chat_history: list) -> str:
    if not chat_history:
        return question
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    rewrite_prompt = f"""
Rewrite the follow-up question as a standalone search query.

Conversation:
{history_text}

Follow-up: {question}

Standalone search query:"""
    return llm.invoke(rewrite_prompt).content.strip()

# -------------------------------------------------
# Format docs — keep raw text so LLM sees exact field names
# -------------------------------------------------

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# Fetch schemes from vector DB
# -------------------------------------------------

EXTRACTION_SYSTEM = """You are an AI assistant for Indian government schemes.

The context contains scheme documents. Each document has these fields — read them carefully and map them:

  "Scheme name"         → scheme_name
  "Description"         → description
  "category"            → category
  "benefits"            → benefits
  "eligibility"         → eligibility
  "application_process" → application_process
  "required_documents"  → documents_required
  "state"               → state
  "Link"                → official_link

IMPORTANT RULES:
1. Extract EVERY scheme present in the context as a separate item.
2. Copy field values EXACTLY as written — do NOT summarize or shorten.
3. Only use 'Not Available' if the field is truly absent from the document.
4. Do NOT skip any scheme found in the context."""

def fetch_schemes(question: str, chat_history: list, k: int = 5) -> List[SchemeOutput]:
    standalone = rewrite_question(question, chat_history)
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(standalone)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_SYSTEM),
        ("placeholder", "{chat_history}"),
        ("human", """Extract ALL schemes from the context below as a structured list.
Copy all field values exactly as they appear in the document.

Context:
{context}

Question: {question}""")
    ])

    result: SchemesListOutput = (prompt | structured_llm).invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })
    return result.schemes

# -------------------------------------------------
# Resolve ordinal references
# -------------------------------------------------

def resolve_scheme_reference(question: str, schemes: list) -> list:
    q = question.lower()
    ordinals = {
        "first": 0, "1st": 0,
        "second": 1, "2nd": 1,
        "third": 2, "3rd": 2,
        "fourth": 3, "4th": 3,
        "fifth": 4, "5th": 4,
    }
    for word, idx in ordinals.items():
        if word in q and idx < len(schemes):
            return [schemes[idx]]
    return schemes

# -------------------------------------------------
# Conversational reply
# -------------------------------------------------

def conversational_reply(question: str, chat_history: list) -> str:
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history[-6:]
    ])
    prompt = f"""You are a helpful Indian government scheme assistant chatbot.

Conversation so far:
{history_text}

User: {question}
AI:"""
    return llm.invoke(prompt).content.strip()

# -------------------------------------------------
# Apply "Visit Site" fallback for missing fields
# -------------------------------------------------

MISSING = {"not available", "", "n/a", "none", "na"}

def apply_visit_site_fallback(scheme_dict: dict) -> dict:
    link = scheme_dict.get("official_link", "")
    fields = ["description", "benefits", "eligibility",
              "documents_required", "application_process"]
    for f in fields:
        val = scheme_dict.get(f, "")
        if not val or val.strip().lower() in MISSING:
            if link and link.strip().lower() not in MISSING:
                scheme_dict[f] = f"Not available in database. 👉 Visit: {link}"
            else:
                scheme_dict[f] = "Not available."
    return scheme_dict

# -------------------------------------------------
# Main ask function
# -------------------------------------------------

def ask_agent(question: str, session_id: str = "user_1"):
    session = get_session(session_id)
    chat_history = session["history"]
    intent = detect_intent(question, chat_history)

    # --- Conversational ---
    if intent == "conversational":
        reply = conversational_reply(question, chat_history)
        save_to_history(session_id, question, reply)
        return {"type": "conversational", "reply": reply}

    limit = parse_limit(question)
    followup = is_followup_on_previous(question, chat_history)

    if followup and session["last_schemes"]:
        schemes = resolve_scheme_reference(question, session["last_schemes"])
    else:
        fetch_k = max(limit or 3, 5)
        schemes = fetch_schemes(question, chat_history, k=fetch_k)
        session["last_schemes"] = schemes
        session["last_limit"] = limit

    # --- Names only ---
    if intent == "names_only":
        selected = schemes[:limit] if limit else schemes
        names = [s.scheme_name for s in selected]
        reply = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
        save_to_history(session_id, question, reply)
        return {"type": "names_only", "reply": reply}

    # --- Specific field ---
    if intent == "specific_field":
        field = detect_field(question)
        lines = []
        for s in schemes:
            d = apply_visit_site_fallback(s.model_dump())
            val = d.get(field, "Not Available")
            lines.append(f"• {s.scheme_name}:\n  {val}")
        reply = "\n\n".join(lines)
        save_to_history(session_id, question, reply)
        return {"type": "specific_field", "field": field, "reply": reply}

    # --- Full detail ---
    selected = schemes[:limit] if limit else schemes
    names = ", ".join(s.scheme_name for s in selected)
    save_to_history(session_id, question, f"Showed full details for: {names}")
    return {
        "type": "full_detail",
        "schemes": [apply_visit_site_fallback(s.model_dump()) for s in selected]
    }