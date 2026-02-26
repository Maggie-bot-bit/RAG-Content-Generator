
# 📊 Complete Project Overview

## 🎯 What is This Project?

**AI-Based Document Understanding & Content Generation System** - A complete RAG (Retrieval-Augmented Generation) system that:

- **Understands your documents** by processing PDF, TXT, and MD files
- **Generates accurate content** grounded in your documents (summaries, blog posts, LinkedIn posts)
- **Works 100% locally** - No API keys, no internet needed after setup
- **Protects your privacy** - All data stays on your machine

---

## 🏗️ System Architecture

### High-Level Flow

```
📄 User Documents (PDF/TXT/MD)
    ↓
✂️ Document Chunking (512 words per chunk, 64 overlap)
    ↓
🔢 Embedding Generation (Sentence Transformers)
    ↓
💾 Vector Storage (FAISS Index)
    ↓
🔍 Query Processing
    ↓
📊 Semantic Search (Find Top-K Relevant Chunks)
    ↓
🤖 LLM Generation (Qwen2.5 with Retrieved Context)
    ↓
✨ Generated Content (Summary/Blog/LinkedIn/Image)
```

### Detailed Components

1. **Document Ingestion** (`rag_local.py` → `build_store()`)
   - Reads PDF/TXT/MD files
   - Splits into chunks (512 words, 64 overlap)
   - Creates embeddings using Sentence Transformers
   - Stores in FAISS vector database

2. **Query Processing** (`rag_local.py` → `run_query()`)
   - User asks a question
   - System retrieves top-K relevant chunks
   - Passes chunks as context to LLM
   - Generates grounded response

3. **Content Generation** (`rag_local.py` → Content Templates)
   - **Summary**: 2-4 sentence concise overview
   - **Blog Intro**: Engaging opening paragraphs
   - **LinkedIn Post**: Professional social media content
   - **Image**: Context-aware image generation

---

## 📁 Project Structure

```
sih/
├── app.py                    # 🌐 Streamlit Web Interface (600 lines)
│   ├── Document upload UI
│   ├── 5 Content Generation Tabs:
│   │   ├── 📄 Summary
│   │   ├── 📝 Blog Post
│   │   ├── 💼 LinkedIn Post
│   │   ├── 🖼️ Image Generation
│   │   └── 🔍 Custom Query
│   └── Session state management
│
├── rag_local.py              # ⚙️ Core RAG Pipeline (428 lines)
│   ├── build_store()         # Document ingestion
│   ├── load_store()          # Load vector database
│   ├── retrieve()           # Semantic search
│   ├── run_query()           # RAG query execution
│   ├── generate_image_from_context()  # Image generation
│   └── Content templates (summary, blog, linkedin, general)
│
├── requirements_rag.txt      # 📦 Dependencies (11 packages)
│   ├── transformers
│   ├── sentence-transformers
│   ├── faiss-cpu
│   ├── streamlit
│   ├── torch
│   ├── diffusers
│   └── pypdf
│
├── setup.bat                 # 🚀 Windows setup script
│
├── README.md                 # 📖 Complete documentation
│
├── docs/                     # 📂 User documents folder
│   └── (Upload PDF/TXT/MD here)
│
└── rag_store/                # 💾 Generated vector store
    ├── index.faiss           # FAISS vector index
    ├── chunks.json           # Document chunks
    ├── meta.json             # Chunk metadata
    └── config.json           # Configuration
```

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Convert text to vectors |
| **Vector DB** | FAISS (Facebook AI Similarity Search) | Fast similarity search |
| **Language Model** | Qwen2.5-0.5B-Instruct | Text generation |
| **Image Model** | Stable Diffusion v1.5 | Image generation |
| **PDF Reader** | pypdf | Extract text from PDFs |
| **ML Framework** | PyTorch | Model execution |

### Model Details

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Size: ~80MB
  - Dimensions: 384
  - Speed: Fast, optimized for CPU

- **LLM**: `Qwen/Qwen2.5-0.5B-Instruct`
  - Size: ~1GB
  - Parameters: 500M
  - Speed: Fast inference, good quality

- **Image Model**: `runwayml/stable-diffusion-v1-5`
  - Size: ~4GB (first download)
  - Quality: High-quality image generation

---

## 🔑 Key Features

### 1. **Document Processing**
- ✅ Supports PDF, TXT, MD files
- ✅ Automatic chunking (512 words, 64 overlap)
- ✅ Memory-optimized (handles large documents)
- ✅ Batch processing for multiple files

### 2. **Content Generation Types**

#### 📄 Summary
- **Output**: 2-4 sentence concise summary
- **Features**: 
  - Describes document purpose
  - Lists key sections
  - No source citations in output
  - Professional abstract style

#### 📝 Blog Post Introduction
- **Output**: Engaging blog opening
- **Features**:
  - Hooks the reader
  - Sets up topic
  - Professional yet accessible

#### 💼 LinkedIn Post
- **Output**: Professional social media content
- **Features**:
  - 2-3 paragraphs
  - Actionable takeaways
  - Professional hashtags

#### 🖼️ Image Generation
- **Output**: Context-aware images
- **Features**:
  - Based on document content
  - Uses Stable Diffusion
  - Customizable inference steps

#### 🔍 Custom Query
- **Output**: Answer to any question
- **Features**:
  - Flexible content types
  - Adjustable parameters
  - Source attribution

### 3. **User Interface**

#### Web Interface (Streamlit)
- **Sidebar**: Document upload and management
- **Main Area**: 5 tabs for different content types
- **Features**:
  - Drag-and-drop file upload
  - Real-time processing progress
  - Source attribution display
  - Retrieved chunks viewer

#### Command Line Interface
- **Ingest**: `python rag_local.py ingest --docs ./docs --store ./rag_store`
- **Query**: `python rag_local.py query --question "..." --store ./rag_store`
- **Options**: Content type, max tokens, top-K chunks

### 4. **Privacy & Performance**
- ✅ 100% local processing
- ✅ No API calls
- ✅ No data sent to external servers
- ✅ Offline capable after setup
- ✅ Memory-optimized for large documents

---

## 💻 Code Structure

### `app.py` (600 lines)

**Main Components:**
1. **Page Configuration** (lines 19-24)
   - Streamlit page setup
   - Theme configuration

2. **Custom CSS** (lines 27-139)
   - Light theme styling
   - Sidebar visibility fixes
   - Responsive design

3. **Session State** (lines 141-153)
   - Store vector database in memory
   - Track loaded state
   - Cache encoder and index

4. **Document Management** (lines 160-288)
   - File uploader
   - Document processing
   - Store loading

5. **Content Generation Tabs** (lines 300-547)
   - Summary tab (lines 300-346)
   - Blog Post tab (lines 348-394)
   - LinkedIn Post tab (lines 396-445)
   - Image Generation tab (lines 447-491)
   - Custom Query tab (lines 494-547)

### `rag_local.py` (428 lines)

**Main Functions:**

1. **`build_store(docs_dir, store_dir, embed_model)`** (lines 118-207)
   - Loads documents
   - Chunks text
   - Generates embeddings
   - Creates FAISS index
   - Saves to disk

2. **`load_store(store_dir)`** (lines 210-217)
   - Loads FAISS index
   - Loads chunks and metadata
   - Initializes encoder

3. **`retrieve(query, index, encoder, chunks, meta, top_k)`** (lines 220-226)
   - Encodes query
   - Searches FAISS index
   - Returns top-K chunks with scores

4. **`run_query(question, store_dir, ...)`** (lines 301-343)
   - Retrieves relevant chunks
   - Formats prompt
   - Generates response
   - Cleans output

5. **`generate_image_from_context(retrieved, ...)`** (lines 346-380)
   - Extracts key phrases
   - Creates image prompt
   - Generates image with Stable Diffusion

**Content Templates** (lines 38-66):
- `summary`: Concise 2-4 sentence summary
- `blog_intro`: Engaging blog opening
- `linkedin_post`: Professional social media content
- `general`: Q&A format

---

## 🚀 Usage Examples

### Example 1: Generate Summary

**Input Document**: Research paper about AI

**Process**:
1. Upload PDF via web interface
2. Click "Process Documents"
3. Go to Summary tab
4. Click "Generate Summary"

**Output**:
```
This research paper presents a comprehensive analysis of artificial 
intelligence applications in healthcare. The document covers key 
sections including introduction to AI in medicine, machine learning 
algorithms for diagnosis, ethical considerations, and future 
implications. It demonstrates how AI technologies can enhance 
healthcare delivery while addressing privacy and accuracy concerns.
```

**Sources**: 📄 research_paper.pdf (Chunks 5, 12, 18)

---

### Example 2: Generate LinkedIn Post

**Input**: Company annual report

**Command**:
```bash
python rag_local.py query \
  --question "Create a LinkedIn post" \
  --store ./rag_store \
  --content-type linkedin_post \
  --max-tokens 256
```

**Output**:
```
🎉 Excited to share our company's annual achievements! 

This year we've seen remarkable growth in innovation and customer 
satisfaction. Our team's dedication to excellence has driven 
significant improvements across all departments.

Key highlights include 30% revenue growth, expansion into 5 new 
markets, and recognition as a top workplace. Looking forward to 
building on this momentum in the coming year!

#BusinessGrowth #Innovation #TeamSuccess #AnnualReport
```

---

### Example 3: Custom Query

**Question**: "What are the main findings?"

**Output**: Detailed answer based on retrieved document chunks with source attribution.

---

## 📊 Data Flow Example

### Step-by-Step: Generating a Summary

1. **User Action**: Uploads `document.pdf` and clicks "Generate Summary"

2. **Document Processing**:
   ```
   document.pdf (10 pages)
   → Extract text (pypdf)
   → Clean text (remove extra whitespace)
   → Chunk into 25 chunks (512 words each)
   → Generate 25 embeddings (384 dimensions each)
   → Store in FAISS index
   ```

3. **Query Processing**:
   ```
   Question: "Summarize the main points"
   → Encode question to vector
   → Search FAISS (cosine similarity)
   → Retrieve top 5 chunks (scores: 0.89, 0.85, 0.82, 0.78, 0.75)
   ```

4. **Generation**:
   ```
   Retrieved chunks → Format prompt
   → Pass to Qwen2.5-0.5B-Instruct
   → Generate summary (256 tokens)
   → Clean output (remove source refs)
   → Display to user
   ```

5. **Output Display**:
   - Summary text
   - Sources: 📄 document.pdf (Chunks 3, 7, 12, 15, 22)
   - Expandable chunk viewer

---

## 🎨 UI Features

### Web Interface Highlights

1. **Sidebar**:
   - File uploader (PDF, TXT, MD)
   - Process Documents button
   - Load existing store option
   - Status indicator

2. **Main Tabs**:
   - **Summary**: Slider for max tokens (128-1024), Top-K selector
   - **Blog Post**: Similar controls, blog-specific generation
   - **LinkedIn Post**: Shorter max tokens (128-512)
   - **Image Generation**: Top-K chunks, inference steps slider
   - **Custom Query**: Full flexibility

3. **Output Display**:
   - Generated content in styled box
   - Sources section with emojis
   - Expandable chunk viewer
   - Relevance scores

---

## 🔧 Configuration Options

### Adjustable Parameters

- **Max Tokens**: 128-1024 (controls output length)
- **Top-K Chunks**: 1-20 (number of retrieved chunks)
- **Content Type**: summary, blog_intro, linkedin_post, general
- **Inference Steps**: 10-50 (for image generation)

### Model Swapping

You can change models in `rag_local.py`:

```python
# For better quality (requires more resources)
DEFAULT_LLM = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# For faster processing (current defaults)
DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 📈 Performance Characteristics

### Memory Usage
- **Embedding Model**: ~200MB RAM
- **LLM (Qwen2.5-0.5B)**: ~2GB RAM
- **FAISS Index**: ~50MB per 1000 chunks
- **Stable Diffusion**: ~4GB RAM (or VRAM if GPU)

### Processing Speed
- **Document Ingestion**: ~1-2 seconds per page
- **Query Processing**: ~2-5 seconds (CPU)
- **Image Generation**: ~30-60 seconds (CPU), ~5-10 seconds (GPU)

### Scalability
- **Max Document Size**: 500K characters (auto-truncated)
- **Max Chunks**: 1000 chunks per document
- **Batch Processing**: Supports multiple documents

---

## 🎯 Key Advantages

1. **No API Costs**: All processing is local
2. **Privacy**: Data never leaves your machine
3. **Offline**: Works without internet after setup
4. **Accurate**: Grounded in actual documents (reduces hallucinations)
5. **Flexible**: Multiple content types from same pipeline
6. **Open Source**: Uses open-source models and libraries
7. **Easy to Use**: Both web UI and CLI interfaces

---

## 🔄 Workflow Summary

### Typical User Journey

1. **Setup** (one-time):
   ```bash
   pip install -r requirements_rag.txt
   ```

2. **Add Documents**:
   - Place PDF/TXT/MD files in `docs/` folder
   - OR upload via web interface

3. **Process Documents**:
   - Web: Click "Process Documents"
   - CLI: `python rag_local.py ingest --docs ./docs --store ./rag_store`

4. **Generate Content**:
   - Web: Select tab → Adjust parameters → Generate
   - CLI: `python rag_local.py query --question "..." --store ./rag_store`

5. **View Results**:
   - Generated content
   - Source attribution
   - Retrieved chunks (optional)

---

## 🎓 Learning Resources

### Understanding RAG
- **Retrieval**: Finding relevant information from documents
- **Augmentation**: Adding context to the prompt
- **Generation**: Creating response based on context

### Key Concepts
- **Embeddings**: Vector representations of text
- **Semantic Search**: Finding similar meaning, not just keywords
- **Chunking**: Breaking documents into manageable pieces
- **Vector Database**: Fast similarity search (FAISS)

---

## 🚦 Project Status

✅ **Complete Features**:
- Document ingestion (PDF, TXT, MD)
- Vector storage (FAISS)
- Semantic search
- Content generation (Summary, Blog, LinkedIn)
- Image generation
- Web interface
- CLI interface
- Source attribution
- Memory optimization

🔄 **Future Enhancements** (Potential):
- Support for more file formats (DOCX, HTML)
- Multi-language support
- Advanced chunking strategies
- Model fine-tuning capabilities
- Export generated content
- Batch processing UI

---

## 📝 Quick Reference

### Essential Commands

```bash
# Install
pip install -r requirements_rag.txt

# Run Web App
streamlit run app.py

# Process Documents (CLI)
python rag_local.py ingest --docs ./docs --store ./rag_store

# Generate Summary (CLI)
python rag_local.py query --question "Summarize" --store ./rag_store --content-type summary

# Generate LinkedIn Post (CLI)
python rag_local.py query --question "LinkedIn post" --store ./rag_store --content-type linkedin_post --max-tokens 256
```

### File Locations

- **Documents**: `docs/` folder
- **Vector Store**: `rag_store/` folder (auto-created)
- **Web App**: `app.py`
- **RAG Pipeline**: `rag_local.py`
- **Dependencies**: `requirements_rag.txt`

---

**This project demonstrates a complete, production-ready RAG system that works entirely offline with no API dependencies!** 🚀

