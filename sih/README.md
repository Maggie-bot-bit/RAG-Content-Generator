# 🤖 AI-Based Document Understanding & Content Generation System

> **Generate accurate, document-grounded content using RAG (Retrieval-Augmented Generation) - No API keys required!**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Open%20Source-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This system uses **Retrieval-Augmented Generation (RAG)** to generate accurate, document-grounded content from your uploaded documents. Unlike traditional AI models that rely only on pre-trained knowledge, this system:

- ✅ **Retrieves relevant information** from your documents before generating content
- ✅ **Reduces hallucinations** by grounding responses in actual document content
- ✅ **Works completely offline** - no API keys or internet required after setup
- ✅ **Protects your privacy** - all processing happens locally

### What is RAG?

**Retrieval-Augmented Generation** combines document retrieval with AI generation. Instead of generating responses from memory alone, the system:

1. Splits your documents into chunks
2. Converts them into searchable vectors
3. Retrieves relevant chunks when you ask questions
4. Uses those chunks as context for accurate, document-based responses

---

## ✨ Features

### 📄 Content Generation
- **Document Summaries** - Comprehensive summaries of your documents
- **Blog Post Introductions** - Engaging blog post openings
- **LinkedIn Posts** - Professional social media content
- **Custom Queries** - Ask any question about your documents

### 🖼️ Image Generation
- Generate context-aware images from document content
- Uses Stable Diffusion (runs locally, no API needed)

### 🎨 User Interface
- **Web Interface** - Beautiful Streamlit-based UI
- **Command Line** - Full CLI support for automation

### 🔒 Privacy & Cost
- **100% Local Processing** - Your data never leaves your machine
- **No API Costs** - All models run locally using open-source tools
- **Offline Capable** - Works without internet after initial setup

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- (Optional) GPU for faster processing

### Installation (3 Steps)

```bash
# 1. Navigate to project directory
cd sih

# 2. Install dependencies (takes 10-15 minutes)
pip install -r requirements_rag.txt

# 3. Run the application
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 📦 Installation

### Step 1: Install Python

If Python is not installed:

1. Download from [python.org](https://www.python.org/downloads/)
2. **Important:** Check ✅ "Add Python to PATH" during installation
3. Restart your terminal after installation

Verify installation:
```bash
python --version
```

### Step 2: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements_rag.txt
```

**Note:** This will download:
- Transformers & Sentence Transformers (~500MB)
- PyTorch (~1-2GB)
- FAISS vector database
- Streamlit web framework
- Stable Diffusion for image generation (~4GB on first use)

### Step 3: Verify Installation

```bash
# Check if Streamlit is installed
python -c "import streamlit; print('✅ Streamlit installed')"

# Check if other key packages are installed
python -c "import transformers, sentence_transformers, faiss; print('✅ All packages installed')"
```

---

## 💻 Usage

### Web Interface (Recommended)

1. **Start the app:**
   ```bash
   streamlit run app.py
   ```

2. **Upload documents:**
   - Click sidebar → Upload PDF, TXT, or MD files
   - Click "🔄 Process Documents"
   - Wait for processing (first time downloads models)

3. **Generate content:**
   - Use tabs: Summary, Blog Post, LinkedIn Post, Image Generation, or Custom Query
   - Adjust parameters (max tokens, top K chunks)
   - Click generate button

4. **View results:**
   - Generated content appears in main area
   - Expand "View Retrieved Chunks" to see source documents

### Command Line Interface

```bash
# 1. Process documents (create vector store)
python rag_local.py ingest --docs ./docs --store ./rag_store

# 2. Generate a summary
python rag_local.py query --question "Summarize the main points" --store ./rag_store --content-type summary

# 3. Generate a LinkedIn post
python rag_local.py query --question "Create a LinkedIn post" --store ./rag_store --content-type linkedin_post --max-tokens 256

# 4. Generate a blog introduction
python rag_local.py query --question "Write a blog introduction" --store ./rag_store --content-type blog_intro --max-tokens 512

# 5. Custom query
python rag_local.py query --question "What are the key findings?" --store ./rag_store
```

### CLI Options

```bash
# View all options
python rag_local.py --help

# Ingest options
python rag_local.py ingest --help

# Query options
python rag_local.py query --help
```

---

## 📁 Project Structure

```
sih/
├── app.py                  # Streamlit web interface (main entry point)
├── rag_local.py           # Core RAG pipeline (CLI interface)
├── requirements_rag.txt   # Python dependencies
├── README.md              # This file
│
├── docs/                  # Place your documents here (PDF, TXT, MD)
│   └── test.txt           # Example document
│
├── rag_store/             # Generated vector store (created after ingestion)
│   ├── index.faiss        # FAISS vector index
│   ├── chunks.json        # Document chunks
│   └── metadata.json      # Chunk metadata
│
└── results/               # Generated content outputs (optional)
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Embedding Model** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Database** | FAISS |
| **Language Model** | Qwen2.5-0.5B-Instruct (or Mistral/LLaMA) |
| **Image Generation** | Stable Diffusion v1.5 |
| **Architecture** | Retrieval-Augmented Generation (RAG) |

### Default Models

- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` (fast, efficient)
- **LLM:** `Qwen/Qwen2.5-0.5B-Instruct` (lightweight, fast)
- **Image:** `runwayml/stable-diffusion-v1-5` (high quality)

### Model Recommendations

**For Better Quality** (requires more resources):
- LLM: `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Llama-2-7b-chat-hf`
- Embedding: `sentence-transformers/all-mpnet-base-v2`

**For Faster Processing** (lower resource usage):
- Keep default models (already optimized)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Python Not Found
```bash
# Solution: Add Python to PATH
# Windows: Search "Environment Variables" → Edit Path → Add Python directory
# Then restart terminal
```

#### 2. Module Not Found Errors
```bash
# Solution: Reinstall dependencies
pip install --upgrade pip
pip install -r requirements_rag.txt
```

#### 3. FAISS Installation Issues
```bash
# For CPU only:
pip install faiss-cpu

# For GPU (if you have CUDA):
pip install faiss-gpu
```

#### 4. CUDA Out of Memory
- Use smaller models
- Reduce batch size in code
- Use CPU mode (slower but works)

#### 5. Model Download Fails
- Check internet connection (needed only for first-time model downloads)
- Models are cached after first download
- Try downloading manually from HuggingFace

#### 6. Image Generation is Slow
- First run downloads ~4GB model (one-time)
- Use GPU for faster generation
- Reduce `num_inference_steps` parameter

#### 7. Memory Errors
- Close other applications
- Use smaller documents
- Split large documents into smaller files
- Use lighter models

### Getting Help

1. Check error messages carefully
2. Verify all dependencies are installed: `pip list`
3. Check Python version: `python --version` (should be 3.8+)
4. Review logs in terminal output

---

## 📊 System Architecture

```
┌─────────────┐
│   User      │
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Document       │
│  Chunking       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Embedding      │
│  Generation     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  FAISS Vector   │
│  Database       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Query: User    │
│  Question       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Semantic       │
│  Search         │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Retrieve Top K │
│  Relevant Chunks│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  LLM Generation │
│  with Context   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Generated      │
│  Content        │
└─────────────────┘
```

---

## 🎯 Use Cases

- **Students** - Summarize study materials and research papers
- **Professionals** - Generate LinkedIn content from company documents
- **Bloggers** - Create blog post drafts from source material
- **Researchers** - Extract insights and generate summaries
- **Companies** - Analyze internal documents and generate reports

---

## 🔐 Privacy & Security

- ✅ All processing happens **locally** on your machine
- ✅ No data is sent to external servers
- ✅ No API keys or authentication required
- ✅ Your documents remain private
- ✅ Works completely **offline** after initial setup

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Report bugs by opening an issue
2. Suggest new features
3. Submit pull requests
4. Improve documentation

---

## 📝 License

This project is open-source and available for academic and research purposes.

---

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for model hosting
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io/) for the web framework
- [Stable Diffusion](https://stability.ai/) for image generation

---
