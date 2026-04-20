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
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")  # Optional   only for Telegram voice

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    db = SessionLocal()
    user = db.query(User).filter(User.telegram_chat_id == chat_id).first()
    db.close()
    
    reply_keyboard = [
        ["  Schemes for farmers", "    Education scholarships"],
        ["  Healthcare schemes", "  Housing scheme"],
        ["   EN", "   HI", "   GU"]
    ]
    markup = ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True, input_field_placeholder="Ask about schemes...", one_time_keyboard=False)

    if user:
        await update.message.reply_text(f"Namaste {user.full_name}!  \nWelcome back to Yojana AI. How can I help you today?", reply_markup=markup)
    else:
        await update.message.reply_text(
            "Namaste!  Welcome to *Yojana AI*.\n just ask me a question!",
            parse_mode='Markdown',
            reply_markup=markup
        )

async def process_text_and_reply(update: Update, text: str, chat_id: str, context: ContextTypes.DEFAULT_TYPE = None, is_voice=False):
    # Show typing status
    try:
        await update.message.reply_chat_action(action="typing")
    except Exception:
        pass

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
        ctype = chunk.get('type')
        
        # 1. Collect conversational text but SKIP UI-only placeholders
        if ctype in ['chunk', 'conversational', 'names_only', 'names_only_start', 'names_only_end', 'specific_field']:
            incoming_chunk = chunk.get('text', '') or chunk.get('reply', '')
            # Filter UI placeholders
            if any(p in incoming_chunk for p in ["Loading cards", "Generating response", "Analyzing profile", "conversational_start", "conversational_end"]):
                continue
            full_text += incoming_chunk
            
        elif ctype == 'names_only_pill':
            s = chunk.get('scheme', {})
            name = s.get('scheme_name') or s.get('name', 'Unknown')
            full_text += f"\n• {name}"

        # 2. Collect rich scheme data
        elif ctype in ['convert_to_cards', 'full_detail', 'schemes_end']:
            # Use provided list if available, otherwise stay with aggregated schemes_data
            if chunk.get('schemes'):
                schemes_data = chunk.get('schemes')
        
        elif ctype == 'scheme_card':
            s = chunk.get('scheme')
            if s and s not in schemes_data:
                schemes_data.append(s)
            
        elif ctype in ['eligibility_result', 'eligibility_for_shown', 'eligibility_start']:
            res_schemes = chunk.get('schemes', [])
            if res_schemes:
                if "\n  *Matching Schemes:*" not in full_text:
                    full_text += "\n\n  *Matching Schemes:*\n"
                for i, s in enumerate(res_schemes):
                    name = s.get('scheme_name') or s.get('name', 'Unknown Scheme')
                    why  = s.get('why_eligible') or s.get('reason', '')
                    full_text += f"{i+1}. *{name}*\n     {why}\n"

    # 3. Append Full Details if they exist
    if schemes_data:
        full_text += "\n\n  *Scheme Details:*"
        for s in schemes_data:
            full_text += f"\n\n  **{s.get('scheme_name')}**"
            full_text += f"\n  {s.get('description')}"
            full_text += f"\n  **Benefits:** {s.get('benefits')}"
            if s.get('official_link') and s.get('official_link') != "#":
                full_text += f"\n  [Official Link]({s.get('official_link')})"

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
        # Telegram has a 4096 char limit. Split into chunks to avoid truncation.
        if len(full_text) <= 4000:
            await update.message.reply_text(full_text, parse_mode='Markdown')
        else:
            # Simple chunking by length
            for i in range(0, len(full_text), 4000):
                await update.message.reply_text(full_text[i:i+4000], parse_mode='Markdown')

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
            await update.message.reply_text(f"  Success! Linked to *{user.full_name}*.", parse_mode='Markdown')
        else:
            await update.message.reply_text("  Account not found.")
        db.close()
        return

    # 2. Handle Language Selection Chips
    if text == "   EN":
        context.user_data['lang'] = 'en'
        await update.message.reply_text("Language set to English.    Ask your question or select a category below.")
        return
    elif text == "   HI":
        context.user_data['lang'] = 'hi'
        await update.message.reply_text("\u092d\u093e\u0937\u093e \u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u0938\u0947\u091f \u0915\u0940 \u0917\u0908 \u0939\u0948\u0964 \u0905\u092a\u0928\u093e \u092a\u094d\u0930\u0936\u094d\u0928 \u092a\u0942\u091b\u0947\u0902 \u092f\u093e \u0928\u0940\u091a\u0947 \u090f\u0915 \u0936\u094d\u0930\u0947\u0923\u0940 \u091a\u0941\u0928\u0947\u0902\u0964")
        return
    elif text == "   GU":
        context.user_data['lang'] = 'gu'
        await update.message.reply_text("\u0aad\u0abe\u0ab7\u0abe \u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0\u0aae\u0abe\u0a82 \u0ab8\u0ac7\u0a9f \u0a95\u0ab0\u0ab5\u0abe\u0aae\u0abe\u0a82 \u0a86\u0ab5\u0ac0 \u0a9b\u0ac7. \u0aa4\u0aae\u0abe\u0ab0\u0acb \u0aaa\u0acd\u0ab0\u0ab6\u0acd\u0aa8 \u0aaa\u0ac2\u0a9b\u0acb \u0a85\u0aa5\u0ab5\u0abe \u0aa8\u0ac0\u0a9a\u0ac7 \u0a8f\u0a95 \u0ab6\u0acd\u0ab0\u0ac7\u0aa3\u0ac0 \u0aaa\u0ab8\u0a82\u0aa6 \u0a95\u0ab0\u0acb.")
        return

    await process_text_and_reply(update, text, chat_id, context)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import tempfile
    chat_id = str(update.effective_chat.id)
    voice_file = await update.message.voice.get_file()
    
    # Use system temp directory for production compatibility
    temp_dir = tempfile.gettempdir()
    temp_ogg = os.path.join(temp_dir, f"tg_voice_{uuid.uuid4()}.ogg")
    
    await voice_file.download_to_drive(temp_ogg)
    
    ui_lang = context.user_data.get('lang', 'en') if context else 'en'
    print(f"DEBUG: Processing Telegram voice message in {ui_lang}... Output: {temp_ogg}")

    try:
        if not GROQ_API_KEY:
            await update.message.reply_text("🎙️ Voice not configured on this server.")
            return

        groq_client = Groq(api_key=GROQ_API_KEY)
        
        # Multilingual prompt to help Whisper with domain and language context
        prompt_text = (
            "This is Yojana AI, helpful assistant for Gujarat government schemes. "
            "Keywords: Kunwarbai Nu Mameru, Vahali Dikri Yojana, Namo Saraswati, "
            "Jan Arogya Yojana, Farmers, Education, Scholarship, Gujarat Govt, "
            "ખેડૂત, વિદ્યાર્થી, કુંવરબાઇનું મામેરું, વહાલી દીકરી યોજના, "
            "નમો સરસ્વતી વિદ્યા સાધના, આરોગ્ય, સહાય, ગુજરાત સરકાર, "
            "कुंवरबाई नु मामेरू, वहाली डिक्री योजना, नमो सरस्वती, "
            "जन आरोग्य योजना, किसान, शिक्षा, छात्रवृत्ति, गुजरात सरकार।"
        )

        with open(temp_ogg, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(temp_ogg, file.read()),
                model="whisper-large-v3",
                language=ui_lang,
                prompt=prompt_text,
                response_format="json"
            )
        
        print(f"DEBUG: Transcription successful: {transcription.text}")
        await process_text_and_reply(update, transcription.text, chat_id, context, is_voice=True)
    finally:
        if os.path.exists(temp_ogg):
            try:
                os.remove(temp_ogg)
            except Exception as e:
                print(f"Cleanup Error (Voice): {e}")

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
_app_initialized = False

async def handle_webhook_update(update_json: dict):
    """Processes a single update received via webhook."""
    global _app_initialized
    if not _telegram_app:
        print("ERROR: Telegram App not initialized (Check TELEGRAM_BOT_TOKEN)")
        return
    
    try:
        if not _app_initialized:
            print("DEBUG: First webhook request. Initializing and Starting Telegram App...")
            await _telegram_app.initialize()
            await _telegram_app.start() # Critical for v20+ webhooks
            _app_initialized = True
            print("DEBUG: Telegram App Started Successfully.")
            
        update = Update.de_json(update_json, _telegram_app.bot)
        
        # Log message content for debugging
        if update.message:
            chat_id = update.effective_chat.id
            text = update.message.text or "[Non-text message]"
            print(f"📥 TELEGRAM RECEIVE: Chat={chat_id}, Msg='{text}'")
        
        await _telegram_app.process_update(update)
        print("📤 TELEGRAM PROCESSED: Update handled.")
    except Exception as e:
        print(f"Error in handle_webhook_update: {e}")

def start_telegram_bot():
    """Starts the bot in polling mode (for local testing)."""
    # Create a fresh app for polling to avoid collision with global one
    app = create_telegram_app()
    if app:
        print("\n" + "="*50)
        print("  🚀 Yojana AI Telegram Bot is starting...")
        print("  📡 Mode: POLLING (Local Development)")
        print("  📜 Note: Webhooks are disabled while polling.")
        print("="*50 + "\n")
        app.run_polling()

if __name__ == '__main__':
    start_telegram_bot()
