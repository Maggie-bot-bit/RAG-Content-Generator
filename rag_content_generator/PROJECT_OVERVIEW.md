# 📊 Project Overview — RAG Content Studio

## 🎯 Project Goal
Build a local-first, production-style RAG app that can:
- Understand user documents (PDF/TXT/MD)
- Generate high-value content quickly
- Keep outputs traceable to source chunks
- Support direct LinkedIn posting
- Run well on low/medium laptops

---

## ✅ Current Capabilities

### 1) Document Intelligence
- Upload and process PDF/TXT/MD files
- Text cleaning + chunking pipeline
- Embedding-based retrieval with FAISS
- Optional reranking (disabled in fast mode for speed)

### 2) Content Generation
- 📄 Structured summary (extractive, chunk-cited)
- 📝 Blog intro generation
- 💼 LinkedIn post generation (catchy + hashtag-enriched)
- 🔍 Custom query answering

### 3) Image Output
- Direct image generation via `sd-turbo`
- Optional prompt-only fast image mode
- Better fallback behavior if image model fails

### 4) LinkedIn Integration
- OAuth connect flow
- In-app posting support
- Sidebar-based connect/post UX for reliability

### 5) UX / Frontend
- Professional Streamlit UI
- Light/Dark theme toggle with vibrant palette
- Fast mode controls
- Cleaner output cards and modern layout

---

## 🧱 Architecture Snapshot

```text
User Docs -> Cleaning/Chunking -> Embeddings -> FAISS Store
                                    ↓
                              Query Retrieval
                                    ↓
                  Fast Extractive Path / LLM Generation Path
                                    ↓
                 Summary | Blog | LinkedIn | Q&A | Image
```

---

## ⚙️ Core Files

- `app.py` — Streamlit web app and UX flow
- `rag_local.py` — ingestion, retrieval, generation logic
- `linkedin_integration.py` — OAuth + LinkedIn posting
- `requirements_rag.txt` — dependencies
- `README.md` — usage + setup docs
- `RESUME_DESCRIPTION.md` — resume-ready project description

---

## 🧠 Model Defaults (Speed-Oriented)

- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: `Qwen/Qwen2.5-0.5B-Instruct`
- Reranker: `BAAI/bge-reranker-base` (optional)
- Image: `stabilityai/sd-turbo`

---

## 🚀 Why this version is better

- Faster on CPU laptops
- Better structured summaries with source references
- More practical LinkedIn/blog outputs in fast mode
- Improved UI polish and theme support
- Direct LinkedIn publishing workflow included

---

## 📌 Status

**Current state:** production-ready for local demos, academic projects, portfolio showcase, and practical content workflows.
