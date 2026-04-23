from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------------------------
# Memory
# -------------------------------------------------

store: dict = {}

def get_session(session_id: str) -> dict:
    if session_id not in store:
        store[session_id] = {
            "history": [],
            "last_schemes": [],
            "last_limit": None,
            "user_profile": None,
            "awaiting_profile": False,
            "lang": "en",
        }
    return store[session_id]

def save_to_history(session_id: str, question: str, answer: str):
    if not answer or not answer.strip():
        # Prevent Mistral API 400 errors by not saving empty messages
        return
    s = get_session(session_id)
    s["history"].append(HumanMessage(content=question))
    s["history"].append(AIMessage(content=answer))

def seed_session_from_db(session_id: str, db_messages: list):
    """
    Pre-populate in-memory RAG history from a persisted ChatHistory.messages list.
    This ensures context survives server restarts when a user resumes a thread.

    db_messages format (from ChatHistory.messages JSON column):
        [{"role": "user", "content": "..."},
         {"role": "assistant", "result": {...}}]
    """
    if session_id in store:
        # Already seeded for this session, no-op
        return
    s = get_session(session_id)
    for msg in db_messages:
        role = msg.get("role")
        if role == "user":
            s["history"].append(HumanMessage(content=msg.get("content", "")))
        elif role == "assistant":
            # Reconstruct a plain-text summary of the AI turn for history context
            result = msg.get("result", {})
            text = (
                result.get("reply")
                or result.get("text")
                or ""
            )
            if not text and result.get("schemes"):
                # For card-type results, create a brief text representation
                names = [s.get("scheme_name", "") for s in result["schemes"] if s.get("scheme_name")]
                text = "Schemes found: " + ", ".join(names)
            if text:
                s["history"].append(AIMessage(content=text))
