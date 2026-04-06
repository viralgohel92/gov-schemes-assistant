import os
import asyncio
import logging
import uuid
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Ensure repo root is on PYTHONPATH
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.agent import ask_agent
from database.db import SessionLocal
from database.models import User
from utils.voice import generate_speech
from groq import Groq

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_chat_id == chat_id).first()
    db.close()
    
    if user:
        await update.message.reply_text(f"Namaste {user.full_name}! 🙏\nWelcome back to Yojana AI. How can I help you today?")
    else:
        await update.message.reply_text(
            "Namaste! 🙏 Welcome to *Yojana AI*.\n\nTo link your account, send: `link your-email@example.com`",
            parse_mode='Markdown'
        )

async def process_text_and_reply(update: Update, text: str, chat_id: str, is_voice=False):
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_chat_id == chat_id).first()
    user_context = None
    if user:
        user_context = {
            "age": user.age, "gender": user.gender, "income": str(user.income) if user.income else None,
            "occupation": user.occupation, "state": user.residence, "caste_category": user.category
        }
    db.close()

    session_id = f"tg_{chat_id}"
    full_text = ""
    schemes_data = []

    # Collect RAG response
    for chunk in ask_agent(text, session_id=session_id, user_context=user_context):
        # 1. Collect conversational text but SKIP UI-only placeholders
        if chunk['type'] in ['chunk', 'conversational', 'names_only', 'specific_field']:
            incoming_chunk = chunk.get('text', '') or chunk.get('reply', '')
            # Filter UI placeholders
            if any(p in incoming_chunk for p in ["Loading cards", "Generating response", "Analyzing profile", "conversational_start", "conversational_end"]):
                continue
            full_text += incoming_chunk
            
        # 2. Collect rich scheme data
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
            full_text += f"\n\n🔸 **{s.get('scheme_name')}**"
            full_text += f"\n📝 {s.get('description')}"
            full_text += f"\n🎁 **Benefits:** {s.get('benefits')}"
            if s.get('official_link') and s.get('official_link') != "#":
                full_text += f"\n🔗 [Official Link]({s.get('official_link')})"

    if not full_text:
        full_text = "I'm sorry, I couldn't process that."

    # Clean up any leftover placeholders
    full_text = full_text.replace("Loading cards...", "").replace("*Loading cards...*", "").strip()

    # 4. Reply (Text or Audio)
    if is_voice:
        lang = "en"
        if any('\u0a80' <= c <= '\u0aff' for c in full_text): lang = "gu"
        elif any('\u0900' <= c <= '\u097f' for c in full_text): lang = "hi"
        
        audio_path = await generate_speech(full_text, lang=lang)
        with open(audio_path, 'rb') as voice_file:
            await update.message.reply_voice(voice=voice_file)
        if os.path.exists(audio_path): os.remove(audio_path)
    else:
        # Telegram has a 4096 char limit
        await update.message.reply_text(full_text[:4000], parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = str(update.effective_chat.id)

    if text.lower().startswith("link "):
        email = text.split(" ", 1)[1].strip()
        db = SessionLocal()
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.telegram_chat_id = chat_id
            db.commit()
            await update.message.reply_text(f"✅ Success! Linked to *{user.full_name}*.", parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Account not found.")
        db.close()
        return

    await process_text_and_reply(update, text, chat_id)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    voice_file = await update.message.voice.get_file()
    temp_ogg = f"tg_voice_{uuid.uuid4()}.ogg"
    await voice_file.download_to_drive(temp_ogg)
    
    try:
        with open(temp_ogg, "rb") as file:
            transcription = GROQ_CLIENT.audio.transcriptions.create(
                file=(temp_ogg, file.read()),
                model="whisper-large-v3",
                response_format="json"
            )
        await process_text_and_reply(update, transcription.text, chat_id, is_voice=True)
    finally:
        if os.path.exists(temp_ogg): os.remove(temp_ogg)

if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("❌ Error: TELEGRAM_BOT_TOKEN not found")
    else:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.add_handler(MessageHandler(filters.VOICE, handle_voice))
        print("🚀 Telegram Bot (with Voice) is running...")
        app.run_polling()
