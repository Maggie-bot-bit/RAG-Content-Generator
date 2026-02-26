# 🤖 RAG Content Studio

> Fast, local, document-grounded content generation with source traceability and LinkedIn posting support.

---

## ✨ What it does

- Upload **PDF / TXT / MD** documents
- Build a local RAG knowledge base (FAISS)
- Generate:
  - 📄 Structured summaries (with chunk citations)
  - 📝 Blog intros
  - 💼 Catchy LinkedIn posts (with relevant hashtags)
  - 🔍 Custom Q&A
  - 🖼️ Direct image generation or fast prompt-only image mode
- Post generated LinkedIn content directly (OAuth flow)
- Toggle **Light / Dark** UI theme

---

## ⚡ Performance-first design

This project is optimized for low/medium laptops:

- **Fast Mode** for low-latency text generation
- **Extractive summary path** (no heavy LLM required)
- Optional reranker usage (disabled in fast mode)
- Small default text model for responsiveness

---

## 🧠 Models (current defaults)

- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM (text):** `Qwen/Qwen2.5-0.5B-Instruct`
- **Reranker (optional):** `BAAI/bge-reranker-base`
- **Image model:** `stabilityai/sd-turbo`

---

## 🚀 Quick start

```bash
cd C:\Users\HP\.openclaw\projects\rag_content_generator
python -m pip install -r requirements_rag.txt
python -m streamlit run app.py
```

Open: `http://localhost:8501`

---

## 🔗 LinkedIn integration setup

Create `.env` in project root:

```env
LINKEDIN_CLIENT_ID=your_client_id
LINKEDIN_CLIENT_SECRET=your_client_secret
LINKEDIN_REDIRECT_URI=http://localhost:8501
```

LinkedIn app requirements:
- Authorized redirect URL: `http://localhost:8501`
- Products enabled:
  - Sign In with LinkedIn using OpenID Connect
  - Share on LinkedIn
- Scope: `w_member_social`

---

## 📁 Project structure

```text
rag_content_generator/
├── app.py
├── rag_local.py
├── linkedin_integration.py
├── requirements_rag.txt
├── README.md
├── RESUME_DESCRIPTION.md
└── rag_store/   (created after processing docs)
```

---

## 🛠️ Notes

- First run may download models (can be slow once).
- For faster image output, keep image steps low.
- If quality > speed is needed, you can switch to larger local LLMs.

---

## ✅ Outcome

A professional, local-first RAG application with modern UI, fast mode, traceable outputs, and direct LinkedIn publishing workflow.
