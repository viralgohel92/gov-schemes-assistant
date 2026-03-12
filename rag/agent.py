from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# Vector DB + Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_mistralai import ChatMistralAI

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Structured Output
from pydantic import BaseModel, Field
from typing import List

# -------------------------------------------------
# Structured Output Schema
# -------------------------------------------------

class SchemeOutput(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    description: str = Field(description="Brief description of the scheme")
    category: str = Field(description="Category of the scheme (e.g., Agriculture, Health, Education)")
    benefits: str = Field(description="Benefits provided under the scheme")
    eligibility: str = Field(description="Eligibility criteria for the scheme")
    documents_required: str = Field(description="Documents required to apply")
    application_process: str = Field(description="Steps to apply for the scheme")
    state: str = Field(description="State where the scheme is applicable (or 'Central' if national)")
    official_link: str = Field(description="Official website or link for the scheme")

class SchemesListOutput(BaseModel):
    schemes: List[SchemeOutput] = Field(description="List of all government schemes found in the context")

# -------------------------------------------------
# Embedding model
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# Vector Database
# -------------------------------------------------

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------
# LLM (base + structured)
# -------------------------------------------------

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2
)

structured_llm = llm.with_structured_output(SchemesListOutput)

# -------------------------------------------------
# Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are an AI assistant that helps users find Indian government schemes.
     The context contains MULTIPLE schemes separated by blank lines.
     Extract ALL schemes present in the context and return every single one.
     If a field is not available, use 'Not Available'."""),

    ("placeholder", "{chat_history}"),

    ("human",
"""
Using ONLY the context below, extract ALL schemes and return them as a list.
Do NOT skip any scheme — return every scheme found.

Context:
{context}

Question:
{question}
""")
])

# -------------------------------------------------
# Format documents
# -------------------------------------------------

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# Rewrite question using history
# -------------------------------------------------

def rewrite_question(question: str, chat_history: list) -> str:

    if not chat_history:
        return question

    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in chat_history
    ])

    rewrite_prompt = f"""
Rewrite the follow-up question into a standalone search query.

Conversation:
{history_text}

Follow-up question:
{question}

Standalone search query:
"""
    response = llm.invoke(rewrite_prompt)
    return response.content.strip()

# -------------------------------------------------
# Manual Memory Store
# -------------------------------------------------

store: dict = {}

def get_chat_history(session_id: str) -> list:
    return store.get(session_id, [])

def save_to_history(session_id: str, question: str, answer: str):
    if session_id not in store:
        store[session_id] = []
    store[session_id].append(HumanMessage(content=question))
    store[session_id].append(AIMessage(content=answer))

# -------------------------------------------------
# Ask function — returns list of dicts
# -------------------------------------------------

def ask_agent(question: str, session_id: str = "user_1") -> list:

    chat_history = get_chat_history(session_id)

    standalone_question = rewrite_question(question, chat_history)

    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    chain = prompt | structured_llm

    result: SchemesListOutput = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })

    # Save summary to memory
    names = ", ".join(s.scheme_name for s in result.schemes)
    save_to_history(session_id, question, f"Schemes found: {names}")

    return [s.model_dump() for s in result.schemes]