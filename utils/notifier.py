import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import requests
from twilio.rest import Client
from dotenv import load_dotenv

# Ensure repo root is on PYTHONPATH
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from database.db import SessionLocal
from database.models import User, Notification

load_dotenv()

# Logger setup
log = logging.getLogger("notifier")
logging.basicConfig(level=logging.INFO)
APP_URL = os.getenv("APP_URL", "https://yojana-ai-seven.vercel.app").rstrip("/")

def send_email(to_email, subject, body_html, body_text=None):
    """Sends a single HTML email using SMTP."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USERNAME") or os.getenv("EMAIL_USER")
    smtp_pass = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASS")
    smtp_sender = os.getenv("SMTP_SENDER", "Yojana AI <notifications@yojana.ai>")

    if not smtp_user or not smtp_pass:
        missing = []
        if not smtp_user: missing.append("SMTP_USERNAME/EMAIL_USER")
        if not smtp_pass: missing.append("SMTP_PASSWORD/EMAIL_PASS")
        log.warning(f"    SMTP credentials not fully configured. Missing: {', '.join(missing)}. Skipping email.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_sender
        msg["To"] = to_email

        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_sender, to_email, msg.as_string())
        
        return True
    except Exception as e:
        log.error(f"  Failed to send email to {to_email}: {e}")
        return False

def send_telegram_notification(chat_id, message):
    """Sends a Telegram message to a specific chat_id."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        # Telegram Markdown requires escaping some characters, but we'll use a simple version
        response = requests.post(url, json={"chat_id": chat_id, "text": message})
        return response.status_code == 200
    except Exception as e:
        log.error(f"  Telegram error ({chat_id}): {e}")
        return False

def send_whatsapp_notification(phone_number, message):
    """Sends a WhatsApp message via Twilio."""
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_wa = os.getenv("TWILIO_WHATSAPP_NUMBER")
    if not all([sid, auth_token, from_wa, phone_number]):
        return False
    try:
        client = Client(sid, auth_token)
        # Twilio WhatsApp numbers must be prefixed with 'whatsapp:'
        target = phone_number if phone_number.startswith('whatsapp:') else f"whatsapp:{phone_number}"
        client.messages.create(body=message, from_=from_wa, to=target)
        return True
    except Exception as e:
        log.error(f"  WhatsApp error ({phone_number}): {e}")
        return False

def broadcast_new_schemes(new_scheme_names, is_update=False, is_delete=False):
    """
    1. Creates a Notification record in DB.
    2. Sends email alerts to all subscribed users.
    """
    if not new_scheme_names:
        return

    count = len(new_scheme_names)
    title = f"  {count} New Government Scheme{'s' if count > 1 else ''} Added!"
    if is_update:
        title = f"  {count} Scheme{'s' if count > 1 else ''} Updated with New Details!"
    elif is_delete:
        title = f"   {count} Scheme{'s' if count > 1 else ''} Removed from Portal"

    scheme_list_str = "\n".join([f"  {name}" for name in new_scheme_names])
    scheme_list_html = "<ul>" + "".join([f"<li>{name}</li>" for name in new_scheme_names]) + "</ul>"
    
    if is_delete:
        action_text = "removed"
    elif is_update:
        action_text = "updated"
    else:
        action_text = "found"

    message = f"Namaste,\n\nWe have {action_text} {count} government scheme{'s' if count > 1 else ''} for Gujarat with new information:\n\n{scheme_list_str}\n\nCheck your eligibility now at: {APP_URL}/\n\nTeam Yojana AI"
    if is_delete:
        message = f"Namaste,\n\nThe following {count} government scheme{'s' if count > 1 else ''} are no longer available on the official portal and have been removed from our database:\n\n{scheme_list_str}\n\nExplore other schemes at: {APP_URL}/\n\nTeam Yojana AI"

    # 1. Create DB Notification
    db = SessionLocal()
    try:
        if is_delete:
            notif_type = "delete_scheme"
        elif is_update:
            notif_type = "update_scheme"
        else:
            notif_type = "new_scheme"

        new_notif = Notification(
            title=title,
            message=f"Schemes {action_text}: {', '.join(new_scheme_names[:3])}{'...' if count > 3 else ''}",
            type=notif_type,
            created_at=datetime.utcnow()
        )
        db.add(new_notif)
        db.commit()
        db.refresh(new_notif)
        
        # 2. Get all users with notifications enabled
        users = db.query(User).filter(User.email_notifications == 1).all()
        log.info(f"  Broadcasting {'updates' if is_update else 'new schemes'} to {len(users)} users...")

        for user in users:
            intro_text = f"We have just {'discovered' if not is_update else 'updated'} <strong>{count} government scheme{'s' if count > 1 else ''}</strong> that might be relevant to you:"
            if is_delete:
                intro_text = f"The following <strong>{count} government scheme{'s' if count > 1 else ''}</strong> have been removed from the official portal:"

            html_content = f"""
            <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; border: 2px solid #FF9933; padding: 25px; border-radius: 12px; max-width: 600px; margin: auto;">
                <h2 style="color: #000080; text-align: center;">Yojana AI Alert</h2>
                <hr style="border: 1px solid #138808; margin-bottom: 20px;">
                <p>Namaste <strong>{user.full_name}</strong>,</p>
                <p style="font-size: 16px; color: #333; line-height: 1.6;">
                    {intro_text}
                </p>
                <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #FF9933; margin: 20px 0;">
                    {scheme_list_html}
                </div>
                <div style="text-align: center; margin-top: 30px;">
                    <a href="{APP_URL}/" style="display: inline-block; padding: 12px 25px; background-color: #FF9933; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; font-size: 16px; transition: background 0.3s;">  {'Explore Other Schemes' if is_delete else 'Check My Eligibility'}</a>
                </div>
                <p style="font-size: 12px; color: #777; margin-top: 40px; border-top: 1px solid #eee; padding-top: 10px;">
                    You are receiving this because you signed up for Gujarat Government Scheme alerts. 
                    You can manage your preferences or unsubscribe in your Profile settings.
                </p>
                <p style="font-size: 12px; color: #000080; text-align: center; font-weight: bold;">
                    Team Yojana AI
                </p>
            </div>
            """
            
            send_email(
                to_email=user.email,
                subject=f"  Yojana AI Alert: {title}",
                body_html=html_content,
                body_text=message
            )
            log.info(f"  Notification sent to {user.email}")
            
            # --- Telegram Alert ---
            if user.telegram_chat_id:
                tg_msg = f"  *{title}*\n\n{message}\n\nCheck now: {APP_URL}/"
                if send_telegram_notification(user.telegram_chat_id, tg_msg):
                    log.info(f"  Telegram notification sent to {user.full_name}")
            
            # --- WhatsApp Alert ---
            if user.whatsapp_number:
                wa_msg = f"  *{title}*\n\n{message}\n\nCheck now: {APP_URL}/"
                if send_whatsapp_notification(user.whatsapp_number, wa_msg):
                    log.info(f"  WhatsApp notification sent to {user.full_name}")

    except Exception as e:
        db.rollback()
        log.error(f"  Error during broadcast: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Test broadcast
    broadcast_new_schemes(["Namo Saraswati Vigyan Sadhana", "Gujarat Student Startup Policy"])
