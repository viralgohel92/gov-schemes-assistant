# 🇮🇳 Yojana AI - Gujarat Government Scheme Assistant

[![GitHub license](https://img.shields.io/github/license/viralgohel92/gov-schemes-assistant)](https://github.com/viralgohel92/gov-schemes-assistant/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/viralgohel92/gov-schemes-assistant)](https://github.com/viralgohel92/gov-schemes-assistant/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/viralgohel92/gov-schemes-assistant/pulls)

Yojana AI is an intelligent, omnichannel platform designed to bridge the gap between complex government policies and the citizens who need them. It uses advanced RAG (Retrieval-Augmented Generation) to provide instant, accurate, and multilingual information about Gujarat government schemes.

---

## 🚀 Features

-   **Semantic Search**: Find schemes based on natural language queries (e.g., "schemes for small farmers" or "education loans").
-   **Eligibility Checking**: Personalized matching based on user profile details (age, income, caste, etc.).
-   **Multilingual Support**: Fully functional in **English**, **Hindi (हिन्दी)**, and **Gujarati (ગુજરાતી)**.
-   **Omnichannel Access**: Chat via **Web UI**, **Telegram**, or **WhatsApp**.
-   **Voice Integration**: Multilingual speech-to-text (STT) and text-to-speech (TTS) for accessibility.
-   **Automated Data Sync**: Daily automated scraping and vectorization of new schemes from official government portals.

---

## 🛠️ Tech Stack

### AI & RAG Pipeline
-   **LLM**: [Mistral AI API](https://mistral.ai/) (via LangChain)
-   **Embeddings**: Mistral AI Embeddings (1024-dimension)
-   **Framework**: [LangChain](https://www.langchain.com/) for orchestration
-   **Voice**: Edge Native STT & Edge-TTS (TTS)

### Backend & Database
-   **Framework**: Flask (Python)
-   **Database**: [Supabase](https://supabase.com/) (PostgreSQL + `pgvector`)
-   **ORMapper**: SQLAlchemy
-   **Serverless**: Vercel (for API and Web deployment)

### Frontend
-   **UI**: HTML5, Vanilla JavaScript, Tailwind CSS
-   **Styling**: Glassmorphism design system

### Scraper & Ops
-   **Scraping**: Playwright
-   **Automation**: GitHub Actions (Cron jobs)
-   **Communication**: `python-telegram-bot`, Twilio (WhatsApp API)

---

## 📁 Project Structure

```text
├── .github/workflows/      # Automated scrapers and sync jobs (GitHub Actions)
├── api/                    # Vercel serverless entry points
├── bot/                    # Telegram and bot integration logic
├── database/               # SQL migrations, DB models, and Supabase connection
├── frontend/               # Web UI assets, templates, and Flask routes
├── rag/                    # Core AI logic (Intent detection, Retrievers, LLM Orchestration)
├── scraper/                # Playwright scraping logic for govt portals
├── scripts/                # Utility scripts for database initialization
├── utils/                  # TTS, Notifier, and Secret helpers
├── workflow.md             # Detailed technical architecture and data flow
└── vercel.json             # Vercel deployment configuration
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
-   Python 3.10+
-   Supabase Account (with `pgvector` enabled)
-   Mistral AI API Key

### 2. Local Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/viralgohel92/gov-schemes-assistant.git
    cd gov-schemes-assistant
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/macOS:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**:
    Create a `.env` file in the root directory:
    ```env
    # Database
    SUPABASE_URL="your_supabase_url"
    SUPABASE_SERVICE_ROLE_KEY="your_key"
    DATABASE_URL="postgresql://postgres:[PASSWORD]@db.[PROJECT_ID].supabase.co:5432/postgres"

    # AI Keys
    MISTRAL_API_KEY="your_mistral_key"

    # Bots
    TELEGRAM_BOT_TOKEN="your_telegram_token"
    TWILIO_ACCOUNT_SID="your_twilio_sid"
    TWILIO_AUTH_TOKEN="your_twilio_token"
    ```

5.  **Run Locally**:
    ```bash
    python frontend/app.py
    ```
    Visit `http://127.0.0.1:5000` in your browser.

---

## 🚢 Deployment

-   **Web**: Hosted on **Vercel** as a serverless application.
-   **Database**: Managed by **Supabase**.
-   **Cron Sync**: GitHub Actions runs the scraper daily at 02:00 UTC to sync new schemes.

---

## 📖 Technical Deep Dive

For a detailed explanation of the RAG pipeline, intent detection, and system architecture, please refer to **[workflow.md](workflow.md)**.

---

## 🤝 Contributors

This project is built and maintained by:

-   **Viral Gohel** ([@viralgohel92](https://github.com/viralgohel92))
-   **Parth Raval** ([@ParthRaval20](https://github.com/ParthRaval20))

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
