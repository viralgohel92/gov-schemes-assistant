import pandas as pd 
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


df = pd.read_csv(r"data/processed/scraped_schemes.csv")

documents = []

for _,row in df.iterrows():
    text = f"""
    Scheme name :{row['scheme_name']}
    Description :{row['details']}
    Link : {row['scheme_link']}
    """

    documents.append(Document(page_content=text))

print("Documents prepared :",len(documents))

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma.from_documents(
    documents,
    embedding_model,
    persist_directory="vector_db"
)

vector_db.persist()

print("vector database created ")
