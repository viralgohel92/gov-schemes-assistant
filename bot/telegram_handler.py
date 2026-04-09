import os
import asyncio
import logging
import uuid
import requests
from telegram import Update, ReplyKeyboardMarkup

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
    level=logging.WARNING
)
# Suppress noisy polling logs from httpx and telegram internals
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")  # Optional — only for Telegram voice

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_chat_id == chat_id).first()
    db.close()
    
    reply_keyboard = [
        ["🌾 Schemes for farmers", "🧑‍🎓 Education scholarships"],
        ["🏥 Healthcare schemes", "🏠 Housing scheme"],
        ["🇺🇸 EN", "🇮🇳 HI", "🇮🇳 GU"]
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, input_field_placeholder="Ask about schemes...", one_time_keyboard=False)

    if user:
        await update.message.reply_text(f"Namaste {user.full_name}! 🙏\nWelcome back to Yojana AI. How can I help you today?", reply_markup=markup)
    else:
        await update.message.reply_text(
            "Namaste! 🙏 Welcome to *Yojana AI*.\n\nTo link your account, send: `link your-email@example.com`\nOr just ask me a question!",
            parse_mode='Markdown',
            reply_markup=markup
        )

async def process_text_and_reply(update: Update, text: str, chat_id: str, context: ContextTypes.DEFAULT_TYPE = None, is_voice=False):
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
    ui_lang = context.user_data.get('lang', 'en') if context else 'en'
    for chunk in ask_agent(text, session_id=session_id, user_context=user_context, ui_lang=ui_lang):
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

    # 2. Handle Language Selection Chips
    if text == "🇺🇸 EN":
        context.user_data['lang'] = 'en'
        await update.message.reply_text("Language set to English. 🇺🇸 Ask your question or select a category below.")
        return
    elif text == "🇮🇳 HI":
        context.user_data['lang'] = 'hi'
        await update.message.reply_text("भाषा हिंदी में सेट कर दी गई है। 🇮🇳 अपना प्रश्न पूछें या नीचे से श्रेणी चुनें।")
        return
    elif text == "🇮🇳 GU":
        context.user_data['lang'] = 'gu'
        await update.message.reply_text("ભાષા ગુજરાતીમાં પસંદ કરાઈ છે. 🇮🇳 તમારો પ્રશ્ન પૂછો અથવા નીચેથી શ્રેણી પસંદ કરો.")
        return

    await process_text_and_reply(update, text, chat_id, context)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    voice_file = await update.message.voice.get_file()
    temp_ogg = f"tg_voice_{uuid.uuid4()}.ogg"
    await voice_file.download_to_drive(temp_ogg)
    
    try:
        if not GROQ_API_KEY:
            await update.message.reply_text("⚠️ Voice not configured on this server.")
            return
        groq_client = Groq(api_key=GROQ_API_KEY)
        with open(temp_ogg, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_ogg, file.read()),
                model="whisper-large-v3",
                response_format="json"
            )
        await process_text_and_reply(update, transcription.text, chat_id, context, is_voice=True)
    finally:
        if os.path.exists(temp_ogg): os.remove(temp_ogg)

def create_telegram_app():
    """Initializes and returns the Telegram Application instance."""
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found")
        return None

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    return app

_telegram_app = create_telegram_app()

async def handle_webhook_update(update_json: dict):
    """Processes a single update received via webhook using an async context manager."""
    if not _telegram_app:
        return
    
    try:
        print(f"DEBUG: Processing Telegram update: {update_json.get('update_id')}")
        async with _telegram_app:
            update = Update.de_json(update_json, _telegram_app.bot)
            await _telegram_app.process_update(update)
        print("DEBUG: Telegram update processed successfully.")
    except Exception as e:
        print(f"Error processing Telegram update: {e}")

def start_telegram_bot():
    """Starts the bot in polling mode (for local testing)."""
    # Create a fresh app for polling to avoid collision with global one
    app = create_telegram_app()
    if app:
        print("Telegram Bot (with Voice) is running in Polling mode...")
        app.run_polling()

if __name__ == '__main__':
    start_telegram_bot()
