# Yojana AI Architecture: Sessions vs. Threads

This document outlines the memory and authentication architecture of the Gov-Scheme-Assistant (Yojana AI) application. It explains exactly **where**, **how**, and **why** both Threads and Sessions are utilized to create a fast, secure, and scalable AI platform.

---

## 1. Threads (Long-Term Conversation Storage)
Threads act as the permanent, isolated record of user interactions.

*   **Where it is used**: 
    *   **Database**: The `ChatHistory` table in Supabase.
    *   **Frontend**: The sidebar displaying past conversations (e.g., "chat_id: 45").
    *   **Backend Routing (`app.py`)**: Defines the specific context bucket for the AI (`rag_session_id = f"thread_{chat_id}"`).
*   **Why it is used**: 
    *   **Context Isolation**: It ensures that a conversation about "Farmer Schemes" does not bleed into a conversation about "Student Scholarships". Every topic remains strictly cordoned off.
    *   **Persistence**: Users expect to be able to close their browser, return days later, and continue an old conversation. Threads make this permanent storage possible.

---

## 2. Web Sessions (Authentication & Anonymous Fallback)
Web sessions are the traditional HTTP cookies managed by Flask to track the user's browser state.

*   **Where it is used**:
    *   **Flask Backend (`app.py`)**: Stored securely via Flask's `session` object (e.g., `session['user_id']`).
*   **Why it is used**:
    *   **Logged-In Users**: It securely holds the `user_id` so the system knows exactly who is making the request without requiring the user to send their password on every message.
    *   **Global Profile Injection**: Because the session knows the `user_id`, the backend can dynamically fetch the user's name, income, and occupation from the database and inject it globally into *all* threads.
    *   **Anonymous Fallback**: If a user is not logged in, the Flask session provides a temporary `session_id`. This allows an anonymous user to chat without their messages getting mixed up with other anonymous users currently on the website. Once the browser closes, this session dies.

---

## 3. AI In-Memory Sessions (Short-Term Working Memory)
AI in-memory sessions are temporary, lightning-fast storage buckets used specifically by the Langchain AI framework during active processing.

*   **Where it is used**:
    *   **AI Memory (`rag/memory.py`)**: A Python dictionary (`_sessions`) sitting in the server's RAM.
    *   **RAG Agent (`rag/agent.py`)**: Temporarily stores arrays like `session["last_schemes"]`.
*   **Why it is used**:
    *   **Re-hydration (Speed)**: Reading a massive conversation history from a PostgreSQL database token-by-token is far too slow for real-time AI generation. Instead, when a user clicks a Thread, the database history is loaded *once* into this lightning-fast RAM session.
    *   **Immediate Context**: If the AI shows you 5 schemes, it saves that exact list into the in-memory session (`last_schemes`). If your very next message is *"Tell me about the second one"*, the AI instantly looks at the RAM session rather than making an expensive, slow trip back to the database.

---

## Summary Diagram
1. **User opens browser** ➔ Flask creates a **Web Session**.
2. **User logs in** ➔ Web Session stores `user_id`. Backend fetches **Global Identity** (Name, Age, Occupation).
3. **User starts a chat** ➔ Database creates a **Thread** (`chat_id = 45`).
4. **User asks a question** ➔ Thread history is "re-hydrated" into the **AI In-Memory Session** (`thread_45`) for lightning-fast token generation.
5. **AI responds** ➔ New messages are saved back to the **Thread** in the database for permanent storage.
