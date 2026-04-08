import os
from langchain_community.vectorstores import Chroma, SupabaseVectorStore
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Optional
from supabase.client import create_client

# -------------------------------------------------
# Schemas
# -------------------------------------------------

class SchemeOutput(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    description: str = Field(description="Brief description of the scheme")
    category: str = Field(description="Category of the scheme")
    benefits: str = Field(description="Benefits provided under the scheme")
    eligibility: str = Field(description="Eligibility criteria for the scheme")
    documents_required: str = Field(description="Documents required to apply")
    application_process: str = Field(description="Steps to apply for the scheme")
    state: str = Field(description="State where the scheme is applicable")
    official_link: str = Field(description="Official website or link for the scheme")

class SchemesListOutput(BaseModel):
    schemes: List[SchemeOutput] = Field(description="List of all government schemes found in the context")

class MinimalSchemeOutput(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")

class MinimalSchemesListOutput(BaseModel):
    schemes: List[MinimalSchemeOutput] = Field(description="List of all government schemes found")

class TranslatedScheme(BaseModel):
    scheme_name: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    benefits: Optional[str] = Field(None)
    eligibility: Optional[str] = Field(None)
    documents_required: Optional[str] = Field(None)
    application_process: Optional[str] = Field(None)
    category: Optional[str] = Field(None)

class UserProfile(BaseModel):
    age: Optional[int] = Field(None)
    income: Optional[str] = Field(None)
    occupation: Optional[str] = Field(None)
    state: Optional[str] = Field(None)
    gender: Optional[str] = Field(None)
    caste_category: Optional[str] = Field(None)
    extra: Optional[str] = Field(None)

class QueryPreprocessor(BaseModel):
    detected_lang: str = Field(description="Detected language code (en, hi, gu)")
    question_en: str = Field(description="Translation of the user question into English")
    intent: str = Field(description="One of: conversational, scheme_search, eligibility_check, names_only, specific_field")
    field: Optional[str] = Field(None, description="Specific field if intent is 'specific_field'")
    profile: Optional[UserProfile] = Field(None)

# -------------------------------------------------
# Lazy Loaders
# -------------------------------------------------

_embedding_model = None
_vector_db = None
_llm = None
_structured_llm = None
_minimal_structured_llm = None
_profile_llm = None
_preprocessor_llm = None

def get_preprocessor_llm():
    global _preprocessor_llm
    if _preprocessor_llm is None:
        _preprocessor_llm = get_llm().with_structured_output(QueryPreprocessor)
    return _preprocessor_llm

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("⏳ Loading Mistral embedding model (API)...")
        # Use Mistral API for embeddings to save bundle size on Vercel
        _embedding_model = MistralAIEmbeddings(model="mistral-embed")
        print("✅ Mistral embeddings ready.")
    return _embedding_model

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if supabase_url and supabase_key:
            print("🌐 Connecting to Supabase Vector Store...")
            supabase_client = create_client(supabase_url, supabase_key)
            _vector_db = SupabaseVectorStore(
                client=supabase_client,
                embedding=get_embedding_model(),
                table_name="documents",
                query_name="match_documents",
            )
        else:
            print("📂 Using local Chroma Vector Store (Fallback)...")
            _vector_db = Chroma(persist_directory="vector_db", embedding_function=get_embedding_model())
    return _vector_db

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2, streaming=False, max_retries=6)
    return _llm

def get_structured_llm():
    global _structured_llm
    if _structured_llm is None:
        _structured_llm = get_llm().with_structured_output(SchemesListOutput)
    return _structured_llm

def get_minimal_structured_llm():
    global _minimal_structured_llm
    if _minimal_structured_llm is None:
        _minimal_structured_llm = get_llm().with_structured_output(MinimalSchemesListOutput)
    return _minimal_structured_llm

def get_profile_llm():
    global _profile_llm
    if _profile_llm is None:
        _profile_llm = get_llm().with_structured_output(UserProfile)
    return _profile_llm

def warmup():
    """
    Pre-load models. On Vercel, this is less useful but kept for local performance.
    """
    print("🔥 Warming up models...")
    get_embedding_model()
    get_vector_db()
    get_llm()
    get_structured_llm()
    get_profile_llm()
    print("✅ Warmup complete.")
