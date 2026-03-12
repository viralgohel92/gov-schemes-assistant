from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uuid, os, sys

# Add parent folder so rag/agent.py can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)

from rag.agent import ask_agent

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    session_id = body.get("session_id", "default")

    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    try:
        result = ask_agent(question, session_id=session_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/reset")
async def reset(request: Request):
    # Return a fresh session_id — frontend stores it and sends it with future requests
    return JSONResponse({"new_session_id": str(uuid.uuid4())})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)