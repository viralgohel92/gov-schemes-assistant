from flask import Flask, render_template, request, jsonify, session
import warnings
import uuid
import os
import sys
import threading

warnings.filterwarnings("ignore")
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ModuleNotFoundError:
    pass

# Ensure repo root is on PYTHONPATH so `rag/` imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

app = Flask(__name__)
app.secret_key = "your-secret-key-change-this"  # Change this in production


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


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    ui_lang  = data.get("lang", None)   # "en", "hi", or "gu" sent from UI language button

    if not question:
        return jsonify({"error": "Empty question"}), 400

    session_id = session.get("session_id", "default_user")

    try:
        try:
            from rag.agent import ask_agent  # type: ignore
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            return (
                jsonify({"error": (
                    f"Missing Python dependency: {missing}. "
                    "Install the agent requirements and restart the server."
                )}),
                500,
            )

        import json
        def generate():
            try:
                for chunk in ask_agent(question, session_id=session_id, ui_lang=ui_lang):
                    yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return app.response_class(generate(), mimetype='text/event-stream')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # use_reloader=False prevents Flask from running _warmup twice in debug mode
    # (Flask's reloader spawns a child process, which would double the warmup work)
    app.run(debug=True, port=5000, use_reloader=False)