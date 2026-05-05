# 📄 Resume Project Description (Updated)

**RAG Content Studio — AI-Based Document Understanding & Content Generation System**

- Built an end-to-end local RAG platform that ingests PDF/TXT/MD files and generates document-grounded summaries, blog intros, LinkedIn posts, custom Q&A, and image outputs.
- Implemented fast semantic retrieval with Sentence Transformers + FAISS and added optional reranking for quality-focused retrieval.
- Designed a dual-path generation architecture: **Fast Mode** (extractive/low-latency) and standard generative mode for richer outputs.
- Added structured, citation-aware summaries with chunk-level source attribution for transparency and traceability.
- Developed a modern Streamlit frontend with professional UX, Light/Dark theme toggle, and performance-centric controls.
- Integrated LinkedIn OAuth workflow with direct posting support from within the app.
- Optimized low-resource execution using lightweight default models (`Qwen2.5-0.5B`, `all-MiniLM-L6-v2`, `sd-turbo`).
- Implemented robust error handling for model loading, Windows file-lock edge cases, and fallback behavior for image generation.
- Created modular architecture (`rag_local.py`, `app.py`, `linkedin_integration.py`) for maintainability and extensibility.
- Tech stack: Python, Streamlit, Transformers, Sentence-Transformers, FAISS, PyTorch, Diffusers, pypdf, NumPy.
