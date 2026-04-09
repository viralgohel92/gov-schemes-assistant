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
