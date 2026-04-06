import warnings
import uuid
import os
import sys
import threading

# Ensure repo root is on PYTHONPATH so `rag/` and `database/` imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import SessionLocal
from database.models import User, ChatHistory, Notification
from utils.notifier import broadcast_new_schemes

warnings.filterwarnings("ignore")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

app = Flask(__name__)
app.secret_key = "your-secret-key-change-this"  # Change this in production

# ── Notifications Utility ───────────────────────────────────────────────────
# We now use utils/notifier.py for all email broadcasts.


# -------------------------------------------------
# ✅ Background warmup — runs as soon as Flask starts.
#    Loads the HuggingFace embedding model (~300MB) and
#    initializes the Mistral LLM client in the background,
#    so they are ready before the first user request arrives.
# -------------------------------------------------

def _warmup():
    try:
        from rag.agent import warmup
        warmup()
    except Exception as e:
        print(f"⚠️  Warmup failed (non-fatal): {e}")

# daemon=True means this thread won't block app shutdown
threading.Thread(target=_warmup, daemon=True).start()


@app.route("/me")
def get_me():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"user": None})
    
    db = SessionLocal()
    try:
        from database.models import User
        user = db.query(User).get(user_id)
        if user:
            return jsonify({"user": {
                "id": user.id, 
                "name": user.full_name,
                "age": user.age,
                "income": user.income,
                "category": user.category,
                "occupation": user.occupation,
                "email_notifications": bool(user.email_notifications)
            }})
        return jsonify({"user": None})
    finally:
        db.close()

@app.route("/update_profile", methods=["POST"])
def update_profile():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.json
    db = SessionLocal()
    try:
        from database.models import User
        user = db.query(User).get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        user.full_name = data.get("full_name") or user.full_name
        user.age = int(data.get("age")) if data.get("age") else user.age
        user.income = int(data.get("income")) if data.get("income") else user.income
        user.category = data.get("category") or user.category
        user.occupation = data.get("occupation") or user.occupation
        user.gender = data.get("gender") or user.gender
        user.email_notifications = 1 if data.get("email_notifications") else 0
        
        db.commit()
        db.refresh(user)
        
        session["user_name"] = user.full_name
        
        # Clear the cached RAG session profile so it re-fetches from DB on next message
        session_id = session.get("session_id")
        if session_id:
            from rag.memory import get_session
            rag_session = get_session(session_id)
            rag_session["user_profile"] = None

        return jsonify({"status": "ok", "user": {"id": user.id, "name": user.full_name}})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    from flask import Response, stream_with_context
    import json
    from rag.agent import ask_agent
    
    data = request.get_json()
    q = data.get("question", "").strip()
    ui_lang = data.get("lang")
    session_id = session.get("session_id", "user_1")
    
    if not q:
        return jsonify({"error": "Empty question"}), 400
    
    # Check if user is logged in to provide context
    user_context = None
    if "user_id" in session:
        db = SessionLocal()
        user = db.query(User).filter(User.id == session["user_id"]).first()
        if user:
            user_context = {
                "age": user.age,
                "gender": user.gender,
                "income": str(user.income) if user.income else None,
                "occupation": user.occupation,
                "state": user.residence, # Map residence to state
                "caste_category": user.category # Map category to caste_category
            }
        db.close()

    def generate():
        try:
            for chunk in ask_agent(q, session_id=session_id, ui_lang=ui_lang, user_context=user_context):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    full_name = data.get("full_name")
    age = data.get("age")
    gender = data.get("gender")
    income = data.get("income")
    category = data.get("category")
    residence = data.get("residence") or "Gujarat"
    occupation = data.get("occupation")

    db = SessionLocal()
    try:
        user_exists = db.query(User).filter(User.email == email).first()
        if user_exists:
            return jsonify({"error": "User already exists"}), 400

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name,
            age=int(age) if age else None,
            gender=gender,
            income=int(income) if income else None,
            category=category,
            residence=residence,
            occupation=occupation
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        
        session["user_id"] = user.id
        session["user_name"] = user.full_name
        
        # Clear the cached RAG session profile so it re-fetches from DB on next message
        session_id = session.get("session_id")
        if session_id:
            from rag.memory import get_session
            rag_session = get_session(session_id)
            rag_session["user_profile"] = None

        return jsonify({"status": "ok", "user": {"id": user.id, "name": user.full_name}})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid email or password"}), 401

        session["user_id"] = user.id
        session["user_name"] = user.full_name
        
        # Clear the cached RAG session profile so it re-fetches from DB on next message
        session_id = session.get("session_id")
        if session_id:
            from rag.memory import get_session
            rag_session = get_session(session_id)
            rag_session["user_profile"] = None

        return jsonify({"status": "ok", "user": {"id": user.id, "name": user.full_name}})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "ok"})

@app.route("/get_history", methods=["GET"])
def get_history():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    db = SessionLocal()
    try:
        chats = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).order_by(ChatHistory.created_at.desc()).all()
        return jsonify([{
            "id": c.id,
            "title": c.title,
            "messages": c.messages,
            "date": c.created_at.isoformat()
        } for c in chats])
    finally:
        db.close()

@app.route("/save_chat", methods=["POST"])
def save_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    chat_id = data.get("chat_id")
    title = data.get("title")
    messages = data.get("messages")

    db = SessionLocal()
    try:
        if chat_id:
            chat = db.query(ChatHistory).get(chat_id)
            if chat and chat.user_id == user_id:
                chat.messages = messages
                if title: chat.title = title
                db.commit()
                return jsonify({"status": "ok", "chat_id": chat.id})
        
        chat = ChatHistory(user_id=user_id, title=title or "New Chat", messages=messages)
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return jsonify({"status": "ok", "chat_id": chat.id})
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/get_notifications", methods=["GET"])
def get_notifications():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"unread_count": 0, "notifications": []})

    db = SessionLocal()
    try:
        from database.models import User, Notification
        user = db.query(User).get(user_id)
        if not user:
            return jsonify({"unread_count": 0, "notifications": []})

        import datetime
        # Load last 10 notifications
        notifs = db.query(Notification).order_by(Notification.created_at.desc()).limit(15).all()
        
        # Filter out notifications the user has already deleted
        deleted_ids = user.deleted_notifications or []
        filtered_notifs = [n for n in notifs if n.id not in deleted_ids][:10]

        unread_count = 0
        if user.last_notified_at:
            # Count only those that are NEWER and NOT in the deleted list
            unread_count = db.query(Notification).filter(
                Notification.created_at > user.last_notified_at,
                Notification.id.notin_(deleted_ids)
            ).count()
        else:
            unread_count = len(filtered_notifs)

        return jsonify({
            "unread_count": unread_count,
            "notifications": [{
                "id": n.id,
                "title": n.title,
                "message": n.message,
                "date": n.created_at.isoformat()
            } for n in filtered_notifs]
        })
    finally:
        db.close()

@app.route("/delete_notification", methods=["POST"])
def delete_notification():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "error"}), 401
    
    data = request.json
    notif_id = data.get("id")
    if not notif_id:
        return jsonify({"status": "error"}), 400
    
    db = SessionLocal()
    try:
        from database.models import User
        from sqlalchemy.orm.attributes import flag_modified
        user = db.query(User).get(user_id)
        if user:
            current_deleted = list(user.deleted_notifications) if user.deleted_notifications else []
            if notif_id not in current_deleted:
                current_deleted.append(notif_id)
                user.deleted_notifications = current_deleted
                # Tell SQLAlchemy to actually save the list
                flag_modified(user, "deleted_notifications")
                db.commit()
            return jsonify({"status": "ok"})
        return jsonify({"status": "error"}), 404
    except Exception as e:
        print(f"Error deleting notif: {e}")
        return jsonify({"status": "error"}), 500
    finally:
        db.close()

@app.route("/mark_read", methods=["POST"])
def mark_read():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"status": "error"}), 401

    db = SessionLocal()
    try:
        from database.models import User, Notification
        import datetime
        user = db.query(User).get(user_id)
        if user:
            # 1. Get the timestamp of the LATEST notification in the database
            latest_notif = db.query(Notification).order_by(Notification.created_at.desc()).first()
            if latest_notif:
                # Set user's last_notified_at to 1 second AFTER the latest notification
                # This ensures even futuristic test notifications are marked as 'read'
                user.last_notified_at = latest_notif.created_at + datetime.timedelta(seconds=1)
            else:
                user.last_notified_at = datetime.datetime.utcnow()
                
            db.commit()
            return jsonify({"status": "ok"})
        return jsonify({"status": "error"}), 404
    except Exception as e:
        print(f"Error marking read: {e}")
        return jsonify({"status": "error"}), 500
    finally:
        db.close()

@app.route("/rename_chat", methods=["POST"])
def rename_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    chat_id = data.get("chat_id")
    title = data.get("title")

    db = SessionLocal()
    try:
        chat = db.query(ChatHistory).get(chat_id)
        if chat and chat.user_id == user_id:
            chat.title = title
            db.commit()
            return jsonify({"status": "ok"})
        return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    chat_id = data.get("chat_id")

    db = SessionLocal()
    try:
        chat = db.query(ChatHistory).get(chat_id)
        if chat and chat.user_id == user_id:
            db.delete(chat)
            db.commit()
            return jsonify({"status": "ok"})
        return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/reset", methods=["POST"])
def reset():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # use_reloader=False prevents Flask from running _warmup twice in debug mode
    # (Flask's reloader spawns a child process, which would double the warmup work)
    app.run(debug=True, port=5000, use_reloader=False)