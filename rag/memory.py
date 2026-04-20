import json
from langchain_core.messages import HumanMessage, AIMessage
from database.db import SessionLocal
from database.models import SessionState
import datetime

# -------------------------------------------------
# Memory (Database Backed for Production)
# -------------------------------------------------

def serialize_history(history):
    """Converts LangChain message objects to JSON-serializable dicts."""
    res = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            res.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            res.append({"type": "ai", "content": msg.content})
    return res

def deserialize_history(history_json):
    """Converts JSON dicts back to LangChain message objects."""
    res = []
    for msg in history_json:
        if msg["type"] == "human":
            res.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            res.append(AIMessage(content=msg["content"]))
    return res

def get_session(session_id: str) -> dict:
    """Loads session state from DB, or returns a new default session."""
    db = SessionLocal()
    try:
        row = db.query(SessionState).filter(SessionState.session_id == session_id).first()
        if row:
            data = row.data
            # Convert history back to message objects
            if "history" in data:
                data["history"] = deserialize_history(data["history"])
            return data
        
        # Default session
        return {
            "history": [],
            "last_schemes": [],
            "last_limit": None,
            "user_profile": None,
            "awaiting_profile": False,
            "lang": "en",
        }
    finally:
        db.close()

def save_session(session_id: str, data: dict):
    """Persists the session data back to the database."""
    # Create a copy to avoid modifying original message objects
    data_copy = data.copy()
    if "history" in data_copy:
        data_copy["history"] = serialize_history(data_copy["history"])
        
    db = SessionLocal()
    try:
        row = db.query(SessionState).filter(SessionState.session_id == session_id).first()
        if not row:
            row = SessionState(session_id=session_id)
            db.add(row)
        
        row.data = data_copy
        row.updated_at = datetime.datetime.utcnow()
        db.commit()
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")
        db.rollback()
    finally:
        db.close()

def save_to_history(session_id: str, question: str, answer: str):
    """Helper to quickly add a Q&A pair to the persistent history."""
    if not answer or not answer.strip():
        return
    
    s = get_session(session_id)
    # Add new messages
    s["history"].append(HumanMessage(content=question))
    s["history"].append(AIMessage(content=answer))
    
    # Prune history to last 10 messages to keep LLM context light
    if len(s["history"]) > 10:
        s["history"] = s["history"][-10:]
        
    save_session(session_id, s)
