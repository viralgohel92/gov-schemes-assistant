from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore") 

load_dotenv()

# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatMistralAI(model="mistral-small-latest")

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that helps users find Indian government schemes."),
    ("human", """
Answer the question using the context below.

Context:
{context}

Question:
{question}
""")
])

# RAG Chain
chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask function
def ask_agent(question):
    return chain.invoke(question)