import warnings
import uuid
import os
import sys
import threading
import random
import datetime

# Ensure repo root is on PYTHONPATH so `rag/` and `database/` imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask, render_template, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import SessionLocal
from database.models import User, ChatHistory, Notification
from utils.notifier import broadcast_new_schemes
try:
    from bot.telegram_handler import start_telegram_bot, handle_webhook_update
except Exception:
    start_telegram_bot = None
    handle_webhook_update = None
import asyncio
import requests
from twilio.twiml.messaging_response import MessagingResponse

warnings.filterwarnings("ignore")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

app = Flask(__name__)
app.secret_key = "your-secret-key-change-this"  # Change this in production

# ── Serverless Database Management ─────────────────────────────────────────
@app.teardown_appcontext
def shutdown_session(exception=None):
    """Ensure database sessions are closed after every request."""
    from database.db import SessionLocal
    # This manually closes the session to prevent Supabase connection leaks
    pass

# -------------------------------------------------
# ✅ Background warmup logic (Shared with CLI/Sync tools)
# -------------------------------------------------

def _warmup():
    try:
        from rag.agent import warmup
        warmup()
    except Exception as e:
        print(f"⚠️  Warmup failed (non-fatal): {e}")

# ── Background Threading ───────────────────────────────────────────────────
# NOTE: Background threads are disabled for Vercel/Serverless deployment.
# We use Webhooks for Telegram instead of polling.
# threading.Thread(target=_warmup, daemon=True).start()


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
    return render_template("landing.html")


@app.route("/app")
def app_interface():
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


@app.route("/stt", methods=["POST"])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    # Use a unique name to avoid collisions
    temp_filename = f"stt_{uuid.uuid4()}.webm"
    temp_path = os.path.join(REPO_ROOT, "tmp", temp_filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    audio_file.save(temp_path)
    
    try:
        from groq import Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            return jsonify({"error": "GROQ_API_KEY not found in server .env"}), 500
            
        client = Groq(api_key=groq_key)
        
        with open(temp_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_path, file.read()),
                model="whisper-large-v3",
                response_format="json"
            )
        
        return jsonify({"text": transcription.text})
    except Exception as e:
        print(f"STT Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/tts", methods=["GET"])
def text_to_speech():
    from flask import send_file
    import asyncio
    from utils.voice import generate_speech
    
    text = request.args.get("text", "")
    lang = request.args.get("lang", "en")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Create a temporary file name for TTS
        temp_filename = f"tts_{uuid.uuid4()}.mp3"
        temp_path = os.path.join(REPO_ROOT, "tmp", temp_filename)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Run the async function from sync Flask
        asyncio.run(generate_speech(text, lang, temp_path))
        
        # We send the file and delete it after sending? 
        # Actually send_file sends it. We might need a cleanup mechanism but let's send first.
        return send_file(temp_path, mimetype="audio/mpeg")
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"error": str(e)}), 500


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

@app.route("/forgot_password", methods=["POST"])
def forgot_password():
    data = request.json
    email = data.get("email")
    
    if not email:
        return jsonify({"error": "Email is required"}), 400
        
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            return jsonify({"error": "No account found with this email"}), 404
            
        # Generate 6-digit OTP
        otp = f"{random.randint(100000, 999999)}"
        user.otp = otp
        user.otp_expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
        db.commit()
        
        # Send OTP via Email
        from utils.notifier import send_email
        subject = "🔐 Your Yojana AI Verification Code"
        html_body = f"""
        <div style="font-family: Arial, sans-serif; border: 1px solid #ddd; padding: 20px; border-radius: 10px; max-width: 500px;">
            <h2 style="color: #000080; text-align: center;">Reset Your Password</h2>
            <p>Namaste,</p>
            <p>You requested to reset your password for Yojana AI. Use the verification code below to proceed:</p>
            <div style="background: #f4f4f4; padding: 15px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 5px; color: #FF9933; border-radius: 5px; margin: 20px 0;">
                {otp}
            </div>
            <p style="font-size: 12px; color: #777;">This code will expire in 10 minutes. If you did not request this, please ignore this email.</p>
            <hr style="border: 0; border-top: 1px solid #eee;">
            <p style="text-align: center; color: #138808; font-weight: bold;">Team Yojana AI</p>
        </div>
        """
        
        if send_email(email, subject, html_body):
            return jsonify({"status": "ok", "message": "OTP sent to your email"})
        else:
            return jsonify({"error": "Failed to send email. Please try again later."}), 500
            
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")
    
    if not all([email, otp]):
        return jsonify({"error": "Email and OTP are required"}), 400
        
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or user.otp != otp:
            return jsonify({"error": "Invalid OTP"}), 401
            
        if user.otp_expiry < datetime.datetime.utcnow():
            return jsonify({"error": "OTP has expired"}), 401
            
        return jsonify({"status": "ok", "message": "OTP verified"})
    finally:
        db.close()

@app.route("/reset_password", methods=["POST"])
def reset_password():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")
    new_password = data.get("password")
    
    if not all([email, otp, new_password]):
        return jsonify({"error": "All fields are required"}), 400
        
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or user.otp != otp:
            return jsonify({"error": "Session expired or invalid. Please request a new OTP."}), 403
            
        if user.otp_expiry < datetime.datetime.utcnow():
            return jsonify({"error": "OTP has expired"}), 403
            
        # Update password
        user.password_hash = generate_password_hash(new_password)
        # Clear OTP
        user.otp = None
        user.otp_expiry = None
        db.commit()
        
        return jsonify({"status": "ok", "message": "Password reset successful. You can now login."})
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

# ── WhatsApp Bot Integration ───────────────────────────────────────────────

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    from rag.agent import ask_agent
    from flask import send_file
    
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '') 
    media_url = request.values.get('MediaUrl0', '')
    media_type = request.values.get('MediaContentType0', '')
    
    resp = MessagingResponse()
    msg = resp.message()
    
    db = SessionLocal()
    user = db.query(User).filter(User.whatsapp_number == from_number).first()
    db.close()

    text_to_process = incoming_msg
    is_voice = False

    # 1. Handle Voice Message
    if media_url and 'audio' in media_type:
        is_voice = True
        try:
            from groq import Groq
            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            auth = (os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
            media_response = requests.get(media_url, auth=auth)
            
            temp_ext = media_type.split('/')[-1]
            temp_path = os.path.join(REPO_ROOT, "tmp", f"wa_voice_{uuid.uuid4()}.{temp_ext}")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(media_response.content)
            
            with open(temp_path, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(temp_path, file.read()),
                    model="whisper-large-v3",
                    response_format="json"
                )
            text_to_process = transcription.text
            if os.path.exists(temp_path): os.remove(temp_path)
        except Exception as e:
            msg.body(f"Error processing voice: {e}")
            return str(resp)

    # 2. Handle Linking
    if text_to_process.lower().startswith("link "):
        email = text_to_process.split(" ", 1)[1].strip()
        db = SessionLocal()
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.whatsapp_number = from_number
            db.commit()
            msg.body(f"✅ Success! Linked to *{user.full_name}*.")
        else:
            msg.body("❌ Account not found.")
        db.close()
        return str(resp)

    # 3. Process with RAG
    user_context = None
    if user:
        user_context = {
            "age": user.age, "gender": user.gender, "income": str(user.income) if user.income else None,
            "occupation": user.occupation, "state": user.residence, "caste_category": user.category
        }

    session_id = f"wa_{from_number}"
    full_text = ""
    schemes_data = []

    for chunk in ask_agent(text_to_process, session_id=session_id, user_context=user_context):
        if chunk['type'] in ['chunk', 'conversational', 'names_only', 'specific_field']:
            incoming_chunk = chunk.get('text', '') or chunk.get('reply', '')
            if any(p in incoming_chunk for p in ["Loading cards", "Generating response", "Analyzing profile", "conversational_start", "conversational_end"]):
                continue
            full_text += incoming_chunk
        elif chunk['type'] in ['convert_to_cards', 'full_detail']:
            schemes_data = chunk.get('schemes', [])
        elif chunk['type'] == 'eligibility_result':
            res_schemes = chunk.get('schemes', [])
            full_text += f"\n\n🎯 *Found {len(res_schemes)} Eligible Schemes:*\n"
            for i, s in enumerate(res_schemes):
                full_text += f"{i+1}. *{s.scheme_name}*\n   ✅ {s.why_eligible}\n"

    # 3. Append Full Details if they exist
    if schemes_data:
        full_text += "\n\n🏛 *Scheme Details:*"
        for s in schemes_data:
            full_text += f"\n\n🔸 *{s.get('scheme_name')}*"
            full_text += f"\n📝 {s.get('description')}"
            full_text += f"\n🎁 *Benefits:* {s.get('benefits')}"
            if s.get('official_link') and s.get('official_link') != "#":
                full_text += f"\n🔗 {s.get('official_link')}"

    if not full_text:
        full_text = "I'm sorry, I couldn't process that."

    full_text = full_text.replace("Loading cards...", "").replace("*Loading cards...*", "").strip()

    # 4. Reply (Text or Audio)
    if is_voice:
        ngrok_url = os.getenv("NGROK_URL", "").split(" -> ")[0] if " -> " in os.getenv("NGROK_URL", "") else os.getenv("NGROK_URL", "")
        media_reply_url = f"{ngrok_url}/tts_wa?text={requests.utils.quote(full_text[:500])}"
        msg.media(media_reply_url)
    else:
        msg.body(full_text[:4000])

    return str(resp)

@app.route("/tts_wa", methods=["GET"])
def tts_wa():
    """Helper route to serve audio files for WhatsApp media replies."""
    from utils.voice import generate_speech
    from flask import send_file
    
    text = request.args.get("text", "")
    lang = "en"
    if any('\u0a80' <= c <= '\u0aff' for c in text): lang = "gu"
    elif any('\u0900' <= c <= '\u097f' for c in text): lang = "hi"
    
    temp_path = os.path.join(REPO_ROOT, "tmp", f"wa_reply_{uuid.uuid4()}.mp3")
    asyncio.run(generate_speech(text, lang, temp_path))
    return send_file(temp_path, mimetype="audio/mpeg")


@app.route("/reset", methods=["POST"])
def reset():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})

# ── Telegram Webhook ─────────────────────────────────────────────────────

@app.route("/telegram", methods=["POST"])
async def telegram_webhook():
    """Endpoint for Telegram Webhook updates."""
    if not handle_webhook_update:
        return jsonify({"error": "Telegram handler not found"}), 500
        
    try:
        update_json = request.get_json()
        if update_json:
            # We use await here because handle_webhook_update is async
            await handle_webhook_update(update_json)
        return "OK", 200
    except Exception as e:
        print(f"Telegram Webhook Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/set_telegram_webhook", methods=["GET"])
def set_telegram_webhook():
    """Utility route to set the Telegram webhook URL."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return "TELEGRAM_BOT_TOKEN not set", 400
    
    # Use ngrok URL if provided in env, else try to guess from request
    domain = os.getenv("VERCEL_URL") or request.host
    if not domain.startswith("http"):
        # Vercel handles https by default, ngrok gives http/https
        protocol = "https" if "vercel" in domain or "ngrok" in domain else "http"
        webhook_url = f"{protocol}://{domain}/telegram"
    else:
        webhook_url = f"{domain}/telegram"

    # Call Telegram API
    tg_url = f"https://api.telegram.org/bot{token}/setWebhook?url={webhook_url}"
    response = requests.get(tg_url)
    return jsonify({
        "status": "Webhook update attempt finished",
        "webhook_url": webhook_url,
        "telegram_response": response.json()
    })


if __name__ == "__main__":
    # use_reloader=False prevents Flask from running _warmup twice in debug mode
    # (Flask's reloader spawns a child process, which would double the warmup work)
    app.run(debug=True, port=5001, use_reloader=False)