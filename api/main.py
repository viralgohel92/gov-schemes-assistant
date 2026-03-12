from fastapi import FastAPI
from rag.agent import ask_agent

app = FastAPI()

@app.get("/ask")

def ask(question: str):
    answer = ask_agent(question)

    return{
        "question": question,
        "answer": answer
    }

