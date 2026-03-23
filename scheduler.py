"""
scheduler.py
════════════════════════════════════════════════════════════════
Run this file on your server instead of app.py.
It starts Flask AND auto-sync scheduler together.

Usage:
    python scheduler.py

It will:
    - Start Flask app on port 5000
    - Run sync_schemes.py every day at 2 AM automatically
════════════════════════════════════════════════════════════════
"""

import schedule
import time
import threading
import logging

from scraper.sync_schemes import run_sync   # sync_schemes.py is in scraper/ folder

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger("scheduler")

# ── Set your sync schedule here ───────────────────────────
# schedule.every().day.at("02:00").do(run_sync)   # every day at 2 AM

schedule.every(10).minutes.do(run_sync)   # for testing: every 2 minutes

# ── Scheduler loop runs in background thread ──────────────
def run_scheduler():
    log.info("✅ Scheduler started — sync runs every day at 02:00 AM")
    while True:
        schedule.run_pending()   # checks if any task is due
        time.sleep(60)           # check every 60 seconds

# Start scheduler in background (won't block Flask)
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Start Flask app ───────────────────────────────────────
from frontend.app import app   # app.py is in frontend/ folder

if __name__ == "__main__":
    log.info("🚀 Starting Yojana AI with auto-sync scheduler...")
    app.run(debug=False, port=5000)