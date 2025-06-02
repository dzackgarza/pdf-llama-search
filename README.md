# Mathematical PDF Search Application

🔎 An advanced, AI-powered search system for mathematical research papers with precise citations and specialized prompting.

## ✨ Features

- 📚 **PDF Processing** - Advanced text extraction optimized for mathematical papers
- 🔍 **Hybrid Search** - Combines BM25 keyword search with semantic embeddings
- 🤖 **Mathematical AI** - Specialized prompts for theorems, definitions, proofs, and concepts
- 📍 **Precise Citations** - Exact page numbers, section references, and theorem citations
- ⚡ **Smart Chunking** - Intelligent text splitting with sentence boundary respect
- 💾 **Caching System** - Fast startup with persistent index caching
- 🔧 **Configurable** - Environment-based configuration with UI controls
- 🎯 **Query Analysis** - Automatic detection of query types for optimal prompting

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this project
git clone <your-repo-url>
cd pdfsearch

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template and configure
cp env_template .env
# Edit .env with your API keys and settings
```

### 3. Setup PDF Directory

```bash
# Create PDF directory and add your research papers
mkdir pdfs
# Copy your mathematical PDF papers to the pdfs/ directory
```

### 4. Run the Application

```bash
# Start the Streamlit application
streamlit run MainApp.py

# Or with custom port
streamlit run MainApp.py --server.port 8502
```

## 📁 Directory Structure

```
pdfsearch/
├── MainApp.py              # Main Streamlit application
├── enhanced_prompts.py     # Specialized mathematical prompts
├── requirements.txt        # Python dependencies
├── env_template           # Environment configuration template
├── README.md              # This file
├── .env                   # Your configuration (create from template)
├── pdfs/                  # Directory for your PDF files
└── pdf_index_cache.pkl    # Generated index cache (auto-created)
```

## ⚙️ Configuration Options

### Environment Variables (.env file)

```bash
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_API_BASE=http://localhost:8000/v1

# PDF Settings
PDF_DIR=pdfs
MAX_PDF_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Search Settings
DEFAULT_TOP_K=10
USE_HYBRID_SEARCH=true

# Embedding Settings
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

### UI Configuration

The application provides a sidebar with runtime configuration options:

- **PDF Directory**: Change the source directory for PDF files
- **Search Settings**: Adjust number of results and search method
- **Embedding Settings**: Choose embedding model and device
- **LLM Settings**: Configure API endpoints

## 🔍 Query Types & Examples

The system automatically detects query types and applies specialized prompts:

### Theorem Queries
```
"Find all theorems about convergence of numerical methods"
"What are the main results for stability analysis?"
"Show me existence theorems for differential equations"
```

### Definition Queries
```
"Definition of compact operator"
"What is a Banach space? Provide exact definitions"
"Define Sobolev spaces with complete mathematical notation"
```

### Proof Queries
```
"Show me proofs of the fundamental theorem"
"How is the uniqueness theorem proved?"
"Outline the proof strategy for the main result"
```

### Concept Queries
```
"Explain the concept of functional analysis"
"Discuss variational methods in partial differential equations"
"What is the theory behind numerical optimization?"
```

## 🛠️ Advanced Features

### Hybrid Search

Combines two search methods for optimal results:

1. **BM25 (Keyword)**: Exact term matching, good for specific mathematical terms
2. **Semantic Embeddings**: Contextual understanding, good for concept queries

### Smart Chunking

- Respects sentence boundaries
- Configurable chunk size and overlap
- Preserves mathematical context
- Maintains page number references

### Query Analysis

Automatically detects:
- Query type (theorem, definition, proof, concept)
- Focus areas for specialized prompting
- Response structure requirements
- Citation format needs

### Caching System

- **Index Caching**: Persistent storage of processed PDFs
- **Embedding Caching**: Reuse of computed embeddings
- **Fast Startup**: 2-3 second load time after first run

## 📚 Mathematical Notation Support

The system handles LaTeX notation in responses:

- **Inline Math**: `$x^2$`, `$\alpha \in \mathbb{R}$`
- **Display Math**: `$$\int_0^1 f(x) dx$$`
- **Standard Symbols**: `\mathbb{R}`, `\mathbb{C}`, `\mathbb{Z}`, `\mathbb{Q}`
- **Greek Letters**: `\alpha`, `\beta`, `\gamma`, `\theta`, `\lambda`

## 🔧 Troubleshooting

### Common Issues

1. **PDF Processing Errors**
   - Install both PyPDF2 and pdfplumber: `pip install PyPDF2 pdfplumber`
   - Check PDF file permissions and corruption

2. **Haystack Import Errors**
   - Install correct version: `pip install haystack-ai>=2.0.0`
   - Check for conflicting installations

3. **Embedding Errors**
   - Ensure sufficient memory for embedding models
   - Try CPU device if CUDA issues occur
   - Use smaller embedding models for low-resource systems

4. **API Connection Issues**
   - Verify API keys and endpoints in .env file
   - Check network connectivity to API services
   - Ensure API service is running for local endpoints

### Performance Optimization

- **Low Memory**: Use smaller embedding models and reduce chunk size
- **CPU-Only**: Set `EMBEDDING_DEVICE=cpu` and use lightweight models
- **Large Collections**: Enable caching and use hybrid search for better relevance

## 🎯 Best Practices

### PDF Organization
- Use descriptive filenames for your research papers
- Organize by topic or research area
- Ensure PDFs are searchable (not scanned images)

### Query Writing
- Be specific about what you're looking for
- Use mathematical terminology consistent with your papers
- Include context for better results

### Citation Usage
- The system provides exact page references
- Cross-reference with original papers for verification
- Use section numbers when available in responses

## 📄 License

This project is for personal academic use. Ensure compliance with API provider terms and PDF copyright restrictions.

## 🤝 Contributing

This is a minimal viable product designed for personal use. Feel free to extend and customize based on your specific research needs.

## 📞 Support

For technical issues:
1. Check the troubleshooting section
2. Verify your configuration in .env
3. Review the debug information in the UI
4. Check logs in the Streamlit terminal output
