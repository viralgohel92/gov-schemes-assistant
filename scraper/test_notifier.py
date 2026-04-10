import os
import logging
from dotenv import load_dotenv

# Set up PYTHONPATH
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.notifier import broadcast_new_schemes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("test_notifier")

def test_notifications():
    """Sends a test notification to all users with email_notifications=1."""
    load_dotenv()
    
    print("\n" + "="*50)
    print("  Yojana AI   SMTP/Notification Test Trigger")
    print("="*50)
    
    # Check secrets
    smtp_user = os.getenv("SMTP_USERNAME") or os.getenv("EMAIL_USER")
    smtp_server = os.getenv("SMTP_SERVER")
    
    if not smtp_user:
        print("  ERROR: SMTP_USERNAME/EMAIL_USER not found in .env")
        return
        
    print(f"  Using SMTP Server: {smtp_server or 'smtp.gmail.com'}")
    print(f"  From: {smtp_user}")
    
    test_names = ["Test Scheme A (AI Verified)", "Test Scheme B (Data Recovery)"]
    
    print(f"\n  Triggering test broadcast for: {', '.join(test_names)}...")
    try:
        # We'll use is_update=True for the test to distinguish it from a real new scheme
        broadcast_new_schemes(test_names, is_update=True)
        print("\n  Test Triggered! Check the logs above for 'Notification sent to...' messages.")
        print("Check your inbox (and spam folder) for the Yojana AI Alert.")
    except Exception as e:
        print(f"\n  CRITICAL FAILURE during test: {e}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    test_notifications()
