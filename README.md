# Document Q&A Chatbot

A **free, local AI-powered chatbot** that answers questions about your PDF documents using **Retrieval-Augmented Generation (RAG)**. Built with Ollama (Llama2), LangChain, FAISS, and Streamlit.

## Features

- **100% Free & Local** - No API keys or cloud services required
- **Multi-PDF Support** - Process multiple PDF documents at once  
- **Intelligent Search** - FAISS vector database for fast, semantic document retrieval
- **Interactive Chat Interface** - Clean Streamlit web UI with chat history
- **Source Citations** - See which document sections were used for each answer
- **Customizable Settings** - Adjust response temperature and number of sources
- **Privacy First** - All processing happens locally on your machine

## Architecture

```
PDFs → Text Chunks → Vector Embeddings → FAISS Database
                                              ↓
User Query → Retrieval → Context + Query → Ollama LLM → Response
```

## Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull Llama2 model
   ollama pull llama2:7b
   
   # Start Ollama server (keep this running)
   ollama serve
   ```

### Installation

1. **Clone & Setup**
   ```bash
   git clone <your-repo-url>
   cd document-qa-chatbot
   pip install -r requirements.txt
   ```

2. **Add Your PDFs**
   ```bash
   mkdir data
   # Copy your PDF files into the 'data/' folder
   ```

3. **Create Vector Database**
   ```bash
   python create_memory_for_llm.py
   ```
   This will:
   - Load all PDFs from `data/` folder
   - Split documents into chunks
   - Create embeddings using HuggingFace
   - Store vectors in FAISS database

4. **Launch Chatbot**
   ```bash
   streamlit run chatbot.py
   ```
   
   The web interface will open at `http://localhost:8501`

## Project Structure

```
document-qa-chatbot/
├── data/                          # Place your PDF files here
├── vectorstore/
│   └── db_faiss/                  # FAISS vector database (auto-generated)
├── create_memory_for_llm.py       # Step 1: Build vector database from PDFs
├── connect_memory_with_llm.py     # Step 2: Connect LLM with database (CLI version)
├── chatbot.py                     # Step 3: Streamlit web interface
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (optional)
└── README.md                     # This file
```

## Usage Options

### 1. Web Interface (Recommended)
```bash
streamlit run chatbot.py
```
- Modern chat interface with history
- Adjustable settings in sidebar
- Source document viewer
- Real-time status monitoring

### 2. Command Line Interface
```bash
python connect_memory_with_llm.py
```
- Terminal-based interaction
- Good for debugging or server environments
- Type `quit` to exit

## Configuration

### Environment Variables (.env)
```bash
# Optional: Add any environment-specific settings
OLLAMA_MODEL=llama2:7b
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Customizable Parameters

**In `create_memory_for_llm.py`:**
- `chunk_size=500` - Size of text chunks (tokens)
- `chunk_overlap=50` - Overlap between chunks
- `model_name="sentence-transformers/all-MiniLM-L6-v2"` - Embedding model

**In `chatbot.py` (via sidebar):**
- Response Temperature (0.0-1.0)
- Number of source documents (1-10)
- Show/hide source documents

## Dependencies

Create `requirements.txt`:
```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.1
langchain-huggingface>=0.0.1
faiss-cpu>=1.7.4
PyPDF2>=3.0.1
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
ollama>=0.1.0
```

Install with: `pip install -r requirements.txt`

## Troubleshooting

### Common Issues

**1. "Ollama connection failed"**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**2. "FAISS database not found"**
```bash
# Run the vector creation script first
python create_memory_for_llm.py
```

**3. "No PDFs found"**
- Ensure PDF files are in `data/` folder
- Check file permissions
- Verify PDFs are not corrupted

**4. "Out of memory errors"**
- Reduce `chunk_size` in `create_memory_for_llm.py`
- Use smaller embedding model
- Process fewer PDFs at once

### Performance Tips

- **Faster responses**: Use smaller models like `llama2:7b`
- **Better accuracy**: Use larger models like `llama2:13b` or `llama2:70b`
- **Less memory**: Reduce chunk size and overlap
- **Better retrieval**: Increase number of source documents

## How It Works

1. **Document Processing** (`create_memory_for_llm.py`)
   - Loads PDFs from `data/` directory
   - Splits into overlapping text chunks
   - Generates embeddings using sentence-transformers
   - Stores in FAISS vector database

2. **Query Processing** (`connect_memory_with_llm.py` + `chatbot.py`)
   - User asks a question
   - System searches FAISS for relevant chunks
   - Retrieved context + question sent to Ollama
   - LLM generates answer based on context

3. **Answer Generation**
   - Ollama (Llama2) processes the prompt
   - Returns answer with source citations
   - Web interface displays results with sources

## Updating Documents

To add new PDFs or update existing ones:

1. Add/update PDFs in `data/` folder
2. Run `python create_memory_for_llm.py` again
3. Restart the chatbot

## Advanced Usage

### Custom Prompt Templates

Edit the prompt in `connect_memory_with_llm.py`:

```python
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the context provided to answer questions.
Be concise and accurate. If unsure, say so.

Context: {context}
Question: {question}

Answer:"""
```

### Different Models

Change the Ollama model:
```bash
# Pull different model
ollama pull codellama:7b

# Update model name in code
llm = Ollama(model="codellama:7b")
```

### Batch Processing

Process multiple document sets:
```python
# Create separate vector stores
DB_FAISS_PATH = "vectorstore/legal_docs"
DB_FAISS_PATH = "vectorstore/technical_docs"
```

## License

This project is open source. Feel free to use, modify, and distribute.

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify Ollama is running properly
4. Open an issue with error details

---

**Happy chatting with your documents!**
