# CorpBrain: RAG-Based Knowledge Agent 🧠

An enterprise-grade AI assistant that enables users to "chat" with private documentation (PDFs) with zero hallucinations. Built with **Google Gemini 2.5**, **Pinecone Vector Database**, and **LangChain**.

![Demo](./screenshot.png)

## 🚀 Key Features
* **Verifiable AI:** Returns exact page citations for every answer to ensure trust.
* **Hybrid Search:** Uses 768-dimensional vector embeddings for semantic understanding.
* **Production Architecture:** Decoupled FastAPI backend and Streamlit frontend.

## 🛠️ Tech Stack
* **LLM:** Google Gemini 2.5 Flash
* **Vector DB:** Pinecone (Serverless)
* **Orchestration:** LangChain (Custom Chains)
* **Backend:** FastAPI (Python)
* **Frontend:** Streamlit

## ⚡ How to Run
1. Clone the repo
2. Add API Keys to `.env`
3. Run `docker-compose up` (Coming Soon)