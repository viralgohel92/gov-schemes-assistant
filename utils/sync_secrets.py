import os
import subprocess
from dotenv import load_dotenv

def sync_secrets():
    print("🔐 Yojana AI — GitHub Secrets Synchronizer")
    print("=" * 45)
    
    if not os.path.exists(".env"):
        print("❌ Error: .env file not found.")
        return

    load_dotenv()
    
    # List of keys to sync
    keys_to_sync = [
        "DATABASE_URL",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "EMAIL_USER",
        "EMAIL_PASS",
        "DB_USERNAME",
        "DB_PASSWORD",
        "DB_HOSTNAME",
        "DB_PORT",
        "DB_NAME",
        "SMTP_SERVER",
        "SMTP_PORT",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "SMTP_SENDER",
        "TELEGRAM_BOT_TOKEN",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_NUMBER"
    ]
    
    for key in keys_to_sync:
        val = os.getenv(key)
        if val:
            print(f"🚀 Syncing {key}...", end=" ", flush=True)
            try:
                # Use 'gh' CLI to set secret
                # subprocess.run handles the passing of value via stdin to avoid shell escaping issues
                process = subprocess.Popen(
                    ["gh", "secret", "set", key],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=val)
                
                if process.returncode == 0:
                    print("✅ DONE")
                else:
                    print(f"❌ FAILED: {stderr.strip()}")
            except Exception as e:
                print(f"❌ ERROR: {e}")
        else:
            print(f"⚠️  Skipping {key} (not in .env)")

    print("=" * 45)
    print("🎉 All secrets synchronized to GitHub!")

if __name__ == "__main__":
    sync_secrets()
