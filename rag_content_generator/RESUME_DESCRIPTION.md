# 📄 Resume Project Description (10 Lines)

**AI-Based Document Understanding & Content Generation System using RAG**

• Developed a complete Retrieval-Augmented Generation (RAG) system that processes PDF, TXT, and MD documents to generate accurate, document-grounded content including summaries, blog posts, and LinkedIn content.

• Implemented semantic search using Sentence Transformers for embeddings and FAISS vector database for efficient similarity search, reducing AI hallucinations by grounding responses in actual document content.

• Built a Streamlit web interface with 5 content generation modules (Summary, Blog Post, LinkedIn Post, Image Generation, Custom Query) and a command-line interface for automation.

• Designed and implemented document processing pipeline with intelligent chunking (512 words, 64 overlap), batch embedding generation, and memory-optimized architecture handling documents up to 500K characters.

• Integrated multiple open-source AI models (Qwen2.5-0.5B-Instruct for text generation, Stable Diffusion v1.5 for images) with local execution, ensuring 100% privacy and zero API costs.

• Implemented source attribution system that tracks and displays document sources and chunk references for generated content, enhancing transparency and verifiability.

• Optimized system performance with batch processing, memory management techniques, and configurable parameters (max tokens, top-K chunks) for different use cases.

• Created comprehensive documentation including README, setup scripts, and troubleshooting guides, enabling easy deployment and usage.

• Technologies: Python, Streamlit, PyTorch, Transformers, Sentence Transformers, FAISS, Stable Diffusion, pypdf, NumPy.

• Result: A production-ready, API-free RAG system that processes documents locally, generates accurate content, and works completely offline after initial setup.

