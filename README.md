# 🤖 Document Chatbot with RAG

A powerful document chatbot application built with Streamlit, LangChain, LangGraph, FAISS, and Gemini API. This app allows you to upload documents and chat with them using AI-powered Retrieval-Augmented Generation (RAG).

## ✨ Features

- 📄 **Multi-format Document Support**: PDF, DOCX, TXT, MD, Excel, CSV, and Images (with OCR)
- 💬 **Intelligent Chat Interface**: Context-aware conversations with document references
- 🔍 **RAG Pipeline**: Advanced retrieval-augmented generation using FAISS vector store
- 🌍 **Multilingual Support**: Multilingual embedding model for better language understanding
- 🧠 **Memory Management**: Persistent chat history and user memory
- 📊 **Analytics Dashboard**: Detailed usage statistics and document insights
- 🎯 **Reference Display**: Source citations for AI responses
- 🔧 **Easy Setup**: Simple configuration with environment variables

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd document-chatbot-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```env
GOOGLE_API_KEY=AIzaSyD0J47YWxp6ZbyAl0bKEep4N2hCfZFInMk
LLM_MODEL=gemini-2.0-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000
TEMPERATURE=0.1
DB_PATH=chatbot.db
VECTOR_STORE_PATH=vector_store```

### 3. Get API Keys

**Gemini API Key:**
1. Visit [Gemini Console](https://aistudio.google.com/)


### 4. Run the Application

```bash
streamlit run app.py
# or
python run.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
document-chatbot-rag/
├── app.py                 # Main Streamlit application
├── run.py                 # Entry point script
├── config.py              # Configuration management
├── database.py            # SQLite database operations
├── document_processor.py  # Document processing and text extraction
├── vector_store_manager.py # FAISS vector store management
├── rag_pipeline.py        # RAG pipeline with LangGraph
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # This file
├── uploads/              # Uploaded documents storage
├── vector_store/         # FAISS vector stores
└── chatbot.db           # SQLite database
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, LangChain, LangGraph
- **Vector Database**: FAISS
- **LLM**: Gemini API 
- **Database**: SQLite
- **Document Processing**: PyPDF2, python-docx, pandas, PIL

## 📚 Supported Document Types

| Format | Extension | Processing Method |
|--------|-----------|-------------------|
| PDF | `.pdf` | Text extraction |
| Word | `.docx`, `.doc` | Text extraction |
| Text | `.txt`, `.md` | Direct reading |
| Excel | `.xlsx`, `.xls`, `.csv` | DataFrame to text |
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` | OCR (requires Tesseract) |

## 🔧 Optional: OCR Support

For image processing with OCR, install Tesseract:

**Windows:**
1. Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add to system PATH

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

Then add to requirements.txt:
```
pytesseract>=0.3.10
```

## 🎯 Usage Guide

### 1. Upload Documents
- Navigate to "📤 Upload Document"
- Choose a supported file type
- Add optional description
- Click "Process Document"

### 2. View Documents
- Go to "📄 My Documents"
- See all uploaded documents with status
- View document information and insights
- Start chat sessions

### 3. Chat with Documents
- Click "💬 Chat" on any document
- Ask questions about the content
- Get AI responses with references
- View chat history and references

### 4. View Statistics
- Check "📊 Statistics" for usage analytics
- See document insights and activity
- Monitor system performance

## 🧠 Advanced Features

### Memory Management
- Automatic chat history persistence
- User memory for personalized interactions
- Session-based conversations

### RAG Pipeline
- Document chunking with overlap
- Vector similarity search
- Context-aware responses
- Source citation tracking

### Document Insights
- Automatic key insights extraction
- Document summarization
- Content analytics

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure both Gemini API keys are valid in `.env`
   - Check if you have sufficient quota

2. **Document Processing Failed**
   - Verify file format is supported
   - Check file size (large files may take time)
   - Ensure sufficient disk space

3. **Vector Store Errors**
   - Clear `vector_store/` directory and reprocess
   - Check embedding model compatibility

4. **Memory Issues**
   - Reduce `CHUNK_SIZE` in configuration
   - Close unused applications
   - Use smaller documents for testing

5. **Multilingual Support**
   - The app supports over 100 languages with Multilingual model
   - For best results, use documents with consistent language per document
   - Mixed-language documents are also supported

### Debug Mode

Enable debug logging by setting:
```env
DEBUG=True
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for the LLM orchestration
- [Gemini](https://aistudio.google.com/) for fast LLM inference
- [FAISS](https://faiss.ai/) for efficient similarity search


## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Happy Document Chatting! 🤖📚**