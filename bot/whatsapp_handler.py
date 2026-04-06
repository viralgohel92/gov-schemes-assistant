import os
import requests
import uuid
import asyncio
from flask import Flask, request, jsonify, send_file
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# Ensure repo root is on PYTHONPATH
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.agent import ask_agent
from database.db import SessionLocal
from database.models import User
from groq import Groq
from utils.voice import generate_speech

load_dotenv()

app = Flask(__name__)
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

# We need the public URL for sending audio back via WhatsApp
# In production this is your domain. In development, it's your Ngrok URL.
NGROK_URL = os.getenv("NGROK_URL", "") # Add this to .env or detect automatically? 

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
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
            # Twilio Media is protected by basic auth (SID/Auth Token)
            auth = (os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
            media_response = requests.get(media_url, auth=auth)
            
            # Save temporary file
            temp_ext = media_type.split('/')[-1]
            temp_path = os.path.join(REPO_ROOT, "tmp", f"wa_voice_{uuid.uuid4()}.{temp_ext}")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(media_response.content)
            
            # Transcribe with Groq/Whisper
            with open(temp_path, "rb") as file:
                transcription = GROQ_CLIENT.audio.transcriptions.create(
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
        # 1. Collect conversational text but SKIP UI-only placeholders
        if chunk['type'] in ['chunk', 'conversational', 'names_only', 'specific_field']:
            incoming_chunk = chunk.get('text', '') or chunk.get('reply', '')
            # Filter UI placeholders
            if any(p in incoming_chunk for p in ["Loading cards", "Generating response", "Analyzing profile", "conversational_start", "conversational_end"]):
                continue
            full_text += incoming_chunk
            
        # 2. Collect rich scheme data (this is the "Full Details" data)
        elif chunk['type'] == 'convert_to_cards':
            schemes_data = chunk.get('schemes', [])
        elif chunk['type'] == 'full_detail':
            schemes_data = chunk.get('schemes', [])
        elif chunk['type'] == 'eligibility_result':
            # Append eligibility list nicely
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

    # Remove any stray "Loading cards..." or "*Loading cards...*" that might have slipped through
    full_text = full_text.replace("Loading cards...", "").replace("*Loading cards...*", "").strip()

    # 4. Reply (Text or Audio)
    if is_voice:
        media_reply_url = f"{NGROK_URL}/tts_wa?text={requests.utils.quote(full_text[:500])}"
        msg.media(media_reply_url)
    else:
        # WhatsApp limit is ~4000
        msg.body(full_text[:4000])

    return str(resp)

@app.route("/tts_wa", methods=["GET"])
def tts_wa():
    """Helper route to serve audio files for WhatsApp media replies."""
    text = request.args.get("text", "")
    # Default to Gujarati for common tasks if not specified
    lang = "en"
    if any('\u0a80' <= c <= '\u0aff' for c in text): lang = "gu"
    elif any('\u0900' <= c <= '\u097f' for c in text): lang = "hi"
    
    temp_path = os.path.join(REPO_ROOT, "tmp", f"wa_reply_{uuid.uuid4()}.mp3")
    asyncio.run(generate_speech(text, lang, temp_path))
    return send_file(temp_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(port=5001, debug=True)
