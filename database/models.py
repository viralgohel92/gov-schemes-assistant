from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Scheme(Base):
    __tablename__ = "schemes"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String)
    scheme_name = Column(String)
    application_link = Column(String)
    description = Column(Text)
    benefits = Column(Text)
    eligibility = Column(Text)
    documents_required = Column(Text)
    application_process = Column(Text)
    state = Column(String)
    missing_count = Column(Integer, default=0)
    last_updated_at = Column(DateTime, default=datetime.datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    full_name = Column(String)
    
    # Eligibility Info
    age = Column(Integer)
    gender = Column(String)
    income = Column(Integer)
    category = Column(String)
    residence = Column(String, default="Gujarat")
    occupation = Column(String)
    
    # Bot Identifiers
    telegram_chat_id = Column(String, unique=True, index=True)
    whatsapp_number = Column(String, unique=True, index=True)
    
    last_notified_at = Column(DateTime, default=datetime.datetime.utcnow)
    email_notifications = Column(Integer, default=1) # 1 for enabled, 0 for disabled
    deleted_notifications = Column(JSON, default=list) # List of dismissed notification IDs
    
    # Forgot Password
    otp = Column(String)
    otp_expiry = Column(DateTime)
    
    chats = relationship("ChatHistory", back_populates="user")

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String)
    messages = Column(JSON) # Store conversation as a list of dicts
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="chats")

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)                                  # e.g. "Namo Saraswati Vigyan"
    message = Column(Text)                                  # e.g. "Newly added scheme for science students."
    type = Column(String, default="new_scheme")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class SessionState(Base):
    __tablename__ = "session_states"
    session_id = Column(String, primary_key=True, index=True) # e.g. "tg_123456"
    data = Column(JSON) # Stores profile, last_schemes, etc.
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)