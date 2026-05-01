import os
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from langchain_core.documents import Document

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

    @field_validator("description", "benefits", "eligibility", "documents_required", "application_process", mode="before")
    @classmethod
    def ensure_string(cls, v):
        if isinstance(v, list):
            return "\n".join(v)
        return v

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

    @field_validator("description", "benefits", "eligibility", "documents_required", "application_process", mode="before")
    @classmethod
    def ensure_string(cls, v):
        if isinstance(v, list):
            return "\n".join(v)
        return v

class SuggestionOutput(BaseModel):
    name: str = Field(description="Localized scheme name")
    category: Optional[str] = Field(None, description="Localized category")

class SuggestionListOutput(BaseModel):
    suggestions: List[SuggestionOutput] = Field(description="List of localized suggestions")

class UserProfile(BaseModel):
    name: Optional[str] = Field(None)
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
# Custom Supabase Direct Retriever
# Bypasses langchain-community's SupabaseVectorStore which has broken
# compatibility with supabase-py v2.x (SyncRPCFilterRequestBuilder API change).
# Calls the match_documents RPC directly   works with any supabase-py v2 version.
# -------------------------------------------------

class _SupabaseDirectRetriever:
    """Thin wrapper around the Supabase pgvector RPC   no langchain-community needed."""

    def __init__(self, client, embedding, query_name: str):
        self._client = client
        self._embedding = embedding
        self._query_name = query_name

    def _similarity_search(self, query: str, k: int = 5):
        query_embedding = self._embedding.embed_query(query)
        result = self._client.rpc(
            self._query_name,
            {"query_embedding": query_embedding, "match_count": k, "match_threshold": 0.0},
        ).execute()
        docs = []
        for row in (result.data or []):
            content = row.get("content", "")
            meta = {key: val for key, val in row.items() if key not in ("content", "embedding")}
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def as_retriever(self, search_kwargs=None):
        k = (search_kwargs or {}).get("k", 5)
        parent = self

        class _Ret:
            def invoke(self_, query: str):
                return parent._similarity_search(query, k)

        return _Ret()


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
        print("  Loading Hugging Face Inference embedding model (API)...")
        # Use BGE-Large-EN (API) for Live Vercel compatibility
        # Ensure 'Inference API' is enabled on your HF token
        _embedding_model = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-large-en-v1.5",
            huggingfacehub_api_token=os.getenv("HF_TOKEN")
        )
        print("Hugging Face Inference embeddings ready.")
    return _embedding_model

class NativeSupabaseVectorStore:
    def __init__(self, client, embedding, table_name="documents", query_name="match_documents"):
        self.client = client
        self.embedding = embedding
        self.table_name = table_name
        self.query_name = query_name

    def add_texts(self, texts: List[str]):
        """Inserts documents, allowing Supabase to auto-generate the ID."""
        embeddings = self.embedding.embed_documents(texts)
        for text, embed in zip(texts, embeddings):
            self.client.table(self.table_name).insert({
                "content": text,
                "metadata": {},
                "embedding": embed
            }).execute()

    def as_retriever(self, search_kwargs=None):
        k = search_kwargs.get("k", 5) if search_kwargs else 5
        return self.NativeRetriever(self, k)

    class NativeRetriever:
        def __init__(self, store, k):
            self.store = store
            self.k = k
            
        def invoke(self, query):
            embed = self.store.embedding.embed_query(query)
            try:
                res = self.store.client.rpc(self.store.query_name, {
                    "query_embedding": embed,
                    "match_threshold": 0.2,
                    "match_count": self.k
                }).execute()
                docs = []
                if res.data:
                    for row in res.data:
                        docs.append(Document(page_content=row.get("content", ""), metadata=row.get("metadata", {})))
                    print(f"  Vector search returned {len(docs)} documents")
                else:
                    print(f"   Vector search returned no results for query: {query[:80]}...")
                return docs
            except Exception as e:
                print(f"  NativeRetriever error: {e}")
                return []

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if supabase_url and supabase_key:
            print("  Connecting to Supabase Vector Store via API (Native Wrapper)...")
            from supabase.client import create_client
            
            try:
                supabase_client = create_client(supabase_url, supabase_key)
                _vector_db = NativeSupabaseVectorStore(
                    client=supabase_client,
                    embedding=get_embedding_model(),
                    table_name="documents",
                    query_name="match_documents"
                )
                print("  Supabase Native Vector Store ready.")
            except Exception as e:
                print(f"   Warning: Supabase client failed: {e}. Falling back to local.")
        
        if _vector_db is None:
            print("  Using local Chroma Vector Store (Fallback)...")
            try:
                from langchain_community.vectorstores import Chroma
                # Support both relative and absolute paths for vector_db
                persist_dir = os.path.join(os.getcwd(), "chroma_db_1024")
                _vector_db = Chroma(persist_directory=persist_dir, embedding_function=get_embedding_model())
            except Exception as e:
                print(f"  ChromaDB fallback failed: {e}")
                return None
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
    print("  Warming up models...")
    get_embedding_model()
    get_vector_db()
    get_llm()
    get_structured_llm()
    get_profile_llm()
    print("  Warmup complete.")
