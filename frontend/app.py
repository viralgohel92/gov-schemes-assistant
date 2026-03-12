from flask import Flask, render_template, request, jsonify, session
import warnings
import uuid
import os
import sys

warnings.filterwarnings("ignore")
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ModuleNotFoundError:
    # Allow running without python-dotenv; env vars can still be set normally.
    pass

# Ensure repo root is on PYTHONPATH so `rag/` imports work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

app = Flask(__name__)
app.secret_key = "your-secret-key-change-this"  # Change this in production


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    session_id = session.get("session_id", "default_user")

    try:
        try:
            # Import lazily so the web UI can start even if
            # the heavy RAG dependencies aren't installed yet.
            from rag.agent import ask_agent  # type: ignore
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None) or str(e)
            return (
                jsonify(
                    {
                        "error": (
                            f"Missing Python dependency: {missing}. "
                            "Install the agent requirements and restart the server."
                        )
                    }
                ),
                500,
            )

        result = ask_agent(question, session_id=session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    session["session_id"] = str(uuid.uuid4())
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
