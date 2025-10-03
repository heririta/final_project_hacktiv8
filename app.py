import streamlit as st
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Import our custom modules
from config import Config, validate_config
from database import db
from document_processor import document_processor
from rag_pipeline import rag_pipeline
from vector_store_manager import vector_store_manager

# Page configuration
st.set_page_config(
    page_title="Chatbot Dokumen dengan RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    """Load CSS from external file"""
    try:
        with open("styles.css", "r") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styling.")
    except Exception as e:
        st.warning(f"Error loading CSS: {str(e)}")

# Load the CSS
load_css()

def validate_document_vector_store(document_id: int) -> Dict:
    """Validate that a document has an accessible vector store"""
    try:
        print(f"validate_document_vector_store: Validating document {document_id}")

        # Get document info
        doc = db.get_document(document_id)
        if not doc:
            return {
                "valid": False,
                "error": "Document not found in database",
                "document": None
            }

        print(f"validate_document_vector_store: Document found: {doc['original_filename']}")

        # Check if document is processed
        if not doc['processed']:
            return {
                "valid": False,
                "error": "Document not processed yet",
                "document": doc
            }

        # Try to load vector store
        try:
            vector_store_loaded = vector_store_manager.load_vector_store(document_id)
            if not vector_store_loaded:
                return {
                    "valid": False,
                    "error": "Vector store not accessible",
                    "document": doc
                }
        except Exception as vs_error:
            return {
                "valid": False,
                "error": f"Vector store loading error: {str(vs_error)}",
                "document": doc
            }

        # Test vector store search (optional - don't fail if search test fails due to network issues)
        try:
            test_search = vector_store_manager.search(document_id, "test", k=1)
            print(f"validate_document_vector_store: Search test returned {len(test_search)} results")
            search_success = True
        except Exception as search_error:
            print(f"validate_document_vector_store: Search test failed (but allowing access): {str(search_error)}")
            search_success = False
            test_search = []

        # Get vector store info
        try:
            vs_info = vector_store_manager.get_vector_store_info(document_id)
            chunk_count = vs_info.get('num_documents', 0) if vs_info else 0
        except Exception as info_error:
            print(f"validate_document_vector_store: Could not get vector store info: {str(info_error)}")
            vs_info = None
            chunk_count = 0

        # If vector store loaded successfully and has chunks, consider it valid
        # even if search test fails (might be network issues)
        if chunk_count > 0:
            return {
                "valid": True,
                "error": None,
                "document": doc,
                "vector_store_info": vs_info,
                "chunk_count": chunk_count,
                "search_test_results": len(test_search),
                "search_test_success": search_success
            }
        else:
            return {
                "valid": False,
                "error": f"Vector store has no chunks (found {chunk_count} chunks)",
                "document": doc
            }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "document": None
        }

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_document_id' not in st.session_state:
        st.session_state.current_document_id = None
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'page' not in st.session_state:
        st.session_state.page = 'homepage'

    # Validate document and vector store consistency
    if st.session_state.current_document_id:
        print(f"initialize_session_state: Validating document {st.session_state.current_document_id}")

        validation_result = validate_document_vector_store(st.session_state.current_document_id)

        if not validation_result['valid']:
            print(f"initialize_session_state: Document validation failed: {validation_result['error']}")

            # Clear session state if document is not accessible
            if validation_result['document']:
                st.warning(f"⚠️ Dokumen '{validation_result['document']['original_filename']}' tidak dapat digunakan: {validation_result['error']}")
            else:
                st.warning(f"⚠️ Dokumen tidak tersedia: {validation_result['error']}")

            # Reset session state
            st.session_state.current_document_id = None
            st.session_state.current_session_id = None
            st.session_state.chat_messages = []
        else:
            # Document and vector store are valid
            doc = validation_result['document']
            vs_info = validation_result.get('vector_store_info', {})
            chunk_count = validation_result.get('chunk_count', 0)

            print(f"initialize_session_state: Document '{doc['original_filename']}' validated successfully")
            print(f"initialize_session_state: - Vector store accessible: ✅")
            print(f"initialize_session_state: - Document chunks: {chunk_count}")
            print(f"initialize_session_state: - Search test results: {validation_result.get('search_test_results', 0)}")

def render_sidebar():
    """Render the sidebar with navigation"""
    print(f"render_sidebar: Starting sidebar rendering")
    print(f"render_sidebar: Current page in session state: {st.session_state.get('page', 'Not set')}")

    st.sidebar.title("📚 Chatbot Dokumen")
    st.sidebar.markdown("---")

    # Get current page for conditional display
    current_page = st.session_state.get('page', 'homepage')

    # API Configuration Section - Hide on Chat and Documents pages
    if current_page not in ['chat', 'documents']:
        st.sidebar.subheader("🔑 API Configuration")

        # Google API Key - loaded from .env file (display only)
        google_api_key = Config.GOOGLE_API_KEY
        if google_api_key:
            st.sidebar.success("✅ Google API Key loaded from .env")
            # Show masked version for security
            masked_key = google_api_key[:8] + "..." + google_api_key[-4:] if len(google_api_key) > 12 else "***"
            st.sidebar.code(masked_key, language="text")
        else:
            st.sidebar.error("❌ Google API Key not found in .env file")
            st.sidebar.info("Please set GOOGLE_API_KEY in your .env file")

        # LLM Provider - Google only
        st.sidebar.info("🤖 **LLM Provider:** Google Gemini")
        st.sidebar.caption(f"Model: {Config.LLM_MODEL}")

        # # Embedding Provider - Google or Cohere
        # embedding_provider_name = "Cohere" if Config.EMBEDDING_PROVIDER == "cohere" else "Google"
        # st.sidebar.info(f"🔒 **Embedding Provider:** {embedding_provider_name}")
        # st.sidebar.caption(f"Model: {Config.EMBEDDING_MODEL}")

        st.sidebar.markdown("---")

    # Navigation options
    navigation_options = ["🏠 Homepage", "📤 Upload Dokumen", "📄 Dokumen Saya", "💬 Chat", "📊 Statistik"]

    # Map session state values to navigation options
    page_to_option = {
        "homepage": "🏠 Homepage",
        "upload": "📤 Upload Dokumen",
        "documents": "📄 Dokumen Saya",
        "chat": "💬 Chat",
        "statistics": "📊 Statistik"
    }

    # Get current option or default
    current_page = st.session_state.get('page', 'homepage')
    current_option = page_to_option.get(current_page, "🏠 Homepage")

    print(f"render_sidebar: current_page = {current_page}")
    # Remove emoji from debug print to avoid encoding issues
    try:
        safe_option = current_option.replace("📤", "[Upload]").replace("📄", "[Docs]").replace("💬", "[Chat]").replace("📊", "[Stats]")
        print(f"render_sidebar: current_option = {safe_option}")
    except UnicodeEncodeError:
        print("render_sidebar: current_option = [Unicode encoding issue]")

    try:
        page = st.sidebar.selectbox(
            "Navigate",
            navigation_options,
            index=navigation_options.index(current_option),
            key="main_navigation"
        )
        # Remove emoji from debug print to avoid encoding issues
        try:
            safe_page = page.replace("📤", "[Upload]").replace("📄", "[Docs]").replace("💬", "[Chat]").replace("📊", "[Stats]")
            print(f"render_sidebar: Selected page: {safe_page}")
        except UnicodeEncodeError:
            print("render_sidebar: Selected page = [Unicode encoding issue]")
    except Exception as e:
        print(f"Error in sidebar selectbox: {str(e)}")
        page = st.sidebar.selectbox("Navigate", navigation_options, index=0, key="fallback_navigation")

    # Map page names to session state
    page_mapping = {
        "🏠 Homepage": "homepage",
        "📤 Upload Dokumen": "upload",
        "📄 Dokumen Saya": "documents",
        "💬 Chat": "chat",
        "📊 Statistik": "statistics"
    }

    new_page = page_mapping[page]
    # Remove emoji from debug prints to avoid encoding issues
    try:
        safe_page_display = page.replace("📤", "[Upload]").replace("📄", "[Docs]").replace("💬", "[Chat]").replace("📊", "[Stats]")
        print(f"render_sidebar: Mapping '{safe_page_display}' to '{new_page}'")
    except UnicodeEncodeError:
        print("render_sidebar: Mapping page with Unicode encoding issue")
    print(f"render_sidebar: Session state page before update: {st.session_state.page}")

    # Store old page for comparison after assignment
    old_page = st.session_state.page

    if st.session_state.page != new_page:
        print(f"render_sidebar: Page changed from {old_page} to {new_page}")

        # Comprehensive refresh logic based on navigation

        # Session state cleanup before navigation
        if old_page == "chat":
            print("render_sidebar: Cleaning up chat session state")
            # Clear chat-specific state but preserve document context
            if 'chat_initialized' in st.session_state:
                st.session_state.chat_initialized = False
            if 'current_session_id' in st.session_state:
                st.session_state.current_session_id = None
            if 'chat_messages' in st.session_state:
                st.session_state.chat_messages = []

        # Document validation when navigating to chat
        if new_page == "chat":
            print("render_sidebar: Validating document before entering chat")
            if not st.session_state.current_document_id:
                print("render_sidebar: No document selected, redirecting to documents")
                st.session_state.page = "documents"
                st.error("Silakan pilih dokumen terlebih dahulu")
                st.rerun()
                return

            # Validate document and vector store
            validation_result = validate_document_vector_store(st.session_state.current_document_id)
            if not validation_result["valid"]:
                print(f"render_sidebar: Document validation failed: {validation_result['error']}")
                st.session_state.page = "documents"
                st.error(f"Dokumen tidak valid: {validation_result['error']}")
                st.rerun()
                return

        # Refresh specific page states
        if new_page == "documents":
            print("render_sidebar: Refreshing documents list")
            # Force refresh of documents list
            if 'documents_refresh_key' not in st.session_state:
                st.session_state.documents_refresh_key = 0
            st.session_state.documents_refresh_key += 1

        elif new_page == "statistics":
            print("render_sidebar: Refreshing statistics")
            # Clear any cached statistics
            if 'stats_cache' in st.session_state:
                del st.session_state.stats_cache

        # Initialize new page state
        if new_page == "upload":
            print("render_sidebar: Initializing upload page")
            # Clear any previous upload state
            if 'upload_success' in st.session_state:
                st.session_state.upload_success = False
            if 'upload_error' in st.session_state:
                st.session_state.upload_error = None

    st.session_state.page = new_page
    print(f"render_sidebar: Session state page after update: {st.session_state.page}")

    # Trigger rerun if page changed to ensure proper state initialization
    if st.session_state.page != old_page:
        print(f"render_sidebar: Triggering rerun for page transition")
        st.rerun()

    # Current document and session info
    if st.session_state.current_document_id:
        print(f"render_sidebar: Current document ID: {st.session_state.current_document_id}")
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Current Document")
        try:
            current_doc = db.get_document(st.session_state.current_document_id)
            if current_doc:
                st.sidebar.warning("⚠️ Document in use")
                print(f"render_sidebar: Current document found: {current_doc['original_filename']}")
                st.sidebar.write(f"**{current_doc['original_filename']}**")
                st.sidebar.write(f"Type: {current_doc['file_type'].upper()}")
                st.sidebar.write(f"Size: {current_doc['file_size'] / 1024:.1f} KB")
            else:
                print(f"render_sidebar: Current document not found!")
                st.sidebar.write("❌ Document not found")
        except Exception as e:
            print(f"render_sidebar: Error getting current document: {str(e)}")
            st.sidebar.write(f"❌ Error: {str(e)}")

            # Show current session info
            if st.session_state.current_session_id:
                current_sessions = db.get_chat_sessions(st.session_state.current_document_id)
                current_session = next((s for s in current_sessions if s['id'] == st.session_state.current_session_id), None)
                if current_session:
                    st.sidebar.write(f"**Session:** {current_session['session_name']}")

    # Quick stats
    st.sidebar.markdown("---")
    documents = db.get_documents()
    st.sidebar.metric("📄 Total Documents", len(documents))

    # Recent chat activity across all documents
    st.sidebar.markdown("---")
    st.sidebar.subheader("🕐 Recent Activity")

    # Get recent sessions from all documents
    all_recent_sessions = []
    for doc in documents[:5]:  # Limit to 5 most recent documents
        doc_sessions = db.get_chat_sessions(doc['id'])
        for session in doc_sessions[:2]:  # Get 2 most recent sessions per document
            all_recent_sessions.append({
                'session': session,
                'document': doc 
            })

    # Sort by last activity and show top 5
    all_recent_sessions.sort(key=lambda x: x['session']['last_activity'], reverse=True)

    # for item in all_recent_sessions[:5]:
    #     session = item['session']
    #     doc = item['document']

    #     if st.button(
    #         f"💬 {doc['original_filename'][:15]}...",
    #         key=f"recent_doc_{doc['id']}_session_{session['id']}",
    #         help=f"{session['session_name']}",
    #         use_container_width=True
    #     ):
    #         st.session_state.current_document_id = doc['id']
    #         st.session_state.current_session_id = session['id']
    #         st.session_state.chat_messages = db.get_chat_messages(session['id'])
    #         st.session_state.page = "chat"
    #         st.rerun()

    # return page

def render_homepage():
    """Render the homepage with README information"""

    # Display logos at the top
    logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 5])

    with logo_col1:
        try:
            st.image("assets/hacktiv8-dark.png", use_container_width=False)
        except:
            pass  # If image not found, continue without it

    with logo_col2:
        try:
            st.image("assets/Google_Gemini_logo.png", width=120, use_container_width=False)
        except:
            pass  # If image not found, continue without it

    # # Main Header
    # st.markdown("""
    # <div class="main-header">
    #     <h1>🤖 Document Chatbot with RAG</h1>
    #     <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">
    #         A powerful document chatbot application that allows you to upload documents and chat with them using AI-powered Retrieval-Augmented Generation (RAG).
    #     </p>
    #     <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">📄</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Multi-format Support</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">💬</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Smart Chat</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">🔍</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">RAG Powered</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">🌍</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Multilingual</div>
    #         </div>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

    # Quick Start Section
    st.markdown("""
    <div style="background: rgba(76, 175, 80, 0.1); border-radius: 15px; padding: 2rem; margin: 2rem 0; border: 2px solid #4CAF50;">
        <h2 style="color: #2E7D32; margin-bottom: 1.5rem;">🚀 Quick Start</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h3 style="color: #4CAF50; margin-bottom: 1rem;">1️⃣ Upload Document</h3>
            <p>Upload any supported document format (PDF, Word, Text) to get started.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <h3 style="color: #4CAF50; margin-bottom: 1rem;">2️⃣ Start Chatting</h3>
            <p>Ask questions about your document and get AI-powered answers with source references.</p>
        </div>
        """, unsafe_allow_html=True)

    # Features Section
    st.subheader("✨ Key Features")

    features = [
        ("📄", "Multi-format Support", "PDF, DOCX, TXT, MD"),
        ("💬", "Intelligent Chat", "Context-aware conversations with document references"),
        ("🔍", "RAG Pipeline", "Advanced retrieval-augmented generation using FAISS vector store"),
        ("🌍", "Multilingual", "multilingual embedding model for better language understanding"),
        ("🧠", "Memory Management", "Persistent chat history and user memory"),
        ("📊", "Analytics", "Detailed usage statistics and document insights"),
        ("🎯", "Reference Display", "Source citations for AI responses"),
        ("🔧", "Easy Setup", "Simple configuration with environment variables")
    ]

    # Display features in a grid
    cols = st.columns(4)
    for i, (icon, title, description) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: white; border-radius: 12px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <h4 style="color: #4CAF50; margin-bottom: 0.5rem;">{title}</h4>
                <p style="font-size: 0.8rem; color: #666; margin: 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

    # # Technology Stack with images
    # st.subheader("🛠️ Technology Stack")

    # # Display technology logos in a row
    # tech_logos = ["assets/Google_Gemini_logo.png", "assets/hacktiv8.htm"]
    # logo_cols = st.columns(len(tech_logos))

    # for i, logo_path in enumerate(tech_logos):
    #     with logo_cols[i]:
    #         try:
    #             st.image(logo_path, width=80, use_container_width=False)
    #         except:
    #             pass

    # tech_stack = {
    #     "Frontend": "Streamlit",
    #     "Backend": "Python, LangChain, LangGraph",
    #     "LLM": "Google Gemini (gemini-2.0-flash)",
    #     "Embeddings": "Cohere (embed-multilingual-v3.0)",
    #     "Vector Database": "FAISS",
    #     "Database": "SQLite",
    #     "Document Processing": "PyPDF2, python-docx, pandas, PIL"
    # }

    # tech_cols = st.columns(3)
    # for i, (tech, tool) in enumerate(tech_stack.items()):
    #     with tech_cols[i % 3]:
    #         st.markdown(f"""
    #         <div style="background: white; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; border-left: 4px solid #4CAF50;">
    #             <strong style="color: #2E7D32;">{tech}:</strong> {tool}
    #         </div>
    #         """, unsafe_allow_html=True)

    # # Supported Document Types
    # st.subheader("📚 Supported Document Types")

    # doc_types = [
    #     ("PDF", ".pdf", "Text extraction"),
    #     ("Word", ".docx, .doc", "Text extraction"),
    #     ("Text", ".txt, .md", "Direct reading"),
    #     ("Excel", ".xlsx, .xls, .csv", "DataFrame to text"),
    #     ("Images", ".jpg, .jpeg, .png, .bmp, .tiff", "OCR (requires Tesseract)")
    # ]

    # st.markdown("""
    # <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    #     <table style="width: 100%; border-collapse: collapse;">
    #         <thead>
    #             <tr style="background: #f5f5f5;">
    #                 <th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #ddd;">Format</th>
    #                 <th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #ddd;">Extension</th>
    #                 <th style="padding: 0.75rem; text-align: left; border-bottom: 2px solid #ddd;">Processing Method</th>
    #             </tr>
    #         </thead>
    #         <tbody>
    # """ + "".join([f"""
    #             <tr>
    #                 <td style="padding: 0.75rem; border-bottom: 1px solid #ddd;"><strong>{doc_type}</strong></td>
    #                 <td style="padding: 0.75rem; border-bottom: 1px solid #ddd;"><code>{extension}</code></td>
    #                 <td style="padding: 0.75rem; border-bottom: 1px solid #ddd;">{method}</td>
    #             </tr>
    # """ for doc_type, extension, method in doc_types]) + """
    #         </tbody>
    #     </table>
    # </div>
    # """, unsafe_allow_html=True)

    # Get Started Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started - Upload Document", type="primary", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

    # Additional Information
    with st.expander("📖 Learn More", expanded=False):
        st.markdown("""
        ### Usage Guide
        1. **Upload Documents**: Navigate to "📤 Upload Dokumen" and upload your files
        2. **View Documents**: Go to "📄 Dokumen Saya" to see all uploaded documents
        3. **Start Chatting**: Click "💬 Chat" on any document to start a conversation
        4. **View Statistics**: Check "📊 Statistik" for usage analytics and API configuration

        ### Troubleshooting
        - **API Issues**: Check your API keys in the Statistics page
        - **Document Processing**: Ensure file format is supported and file size is reasonable
        - **Performance**: Large documents may take longer to process

        For detailed troubleshooting, check the API Configuration section in the Statistics page.
        """)

def render_upload_page():
    """Render the document upload page"""
    # # Main Header
    # st.markdown("""
    # <div class="main-header">
    #     <h1>📤 Upload Document</h1>
    #     <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">
    #         Upload a document to start chatting with it using AI-powered RAG (Retrieval-Augmented Generation)
    #     </p>
    #     <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">🤖</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">AI-Powered</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">📚</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Document Analysis</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">💬</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Smart Chat</div>
    #         </div>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 1.5rem; margin: 2rem 0; border: 2px solid #e8f5e8; backdrop-filter: blur(10px);">
        <h3 style="color: #4CAF50; margin-bottom: 1rem;">📤 Upload Document</h3>
    </div>
    """, unsafe_allow_html=True)


    # Supported file types
    supported_types = [
        "📄 PDF files (.pdf)",
        "📝 Word documents (.docx, .doc)",
        "📃 Text files (.txt, .md)"    
    ]

    # st.markdown("""
    # <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 1.5rem; backdrop-filter: blur(10px);">
    #     <h3 style="color: white; margin-bottom: 1rem;">📋 Supported File Types</h3>
    #     <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
    # """ + "".join([f"<div style='background: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 10px;'>{file_type}</div>" for file_type in supported_types]) + """
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Document Upload",
        type=['pdf', 'docx', 'doc', 'txt', 'md']
    )

    if uploaded_file:
        # Display file info with enhanced design
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
                    border-radius: 15px; padding: 2rem; margin: 2rem 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1); backdrop-filter: blur(10px);">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">📁 File Information</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem;">
        """, unsafe_allow_html=True)

        # File metrics with enhanced styling
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #4CAF50;">File Name</div>
                <div style="font-size: 0.95rem; color: #666; margin-top: 0.5rem;">{uploaded_file.name[:30]}...</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            file_size_kb = uploaded_file.size / 1024
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💾</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #4CAF50;">File Size</div>
                <div style="font-size: 0.95rem; color: #666; margin-top: 0.5rem;">{file_size_kb:.1f} KB</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            file_type = uploaded_file.name.split('.')[-1].upper()
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏷️</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #4CAF50;">File Type</div>
                <div style="font-size: 0.95rem; color: #666; margin-top: 0.5rem;">{file_type}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Description input with enhanced styling
        st.markdown("""
        <div style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border: 2px solid #e8f5e8;">
            <h4 style="color: #4CAF50; margin-bottom: 1rem;">📝 Document Description (Optional)</h4>
        </div>
        """, unsafe_allow_html=True)

        description = st.text_area(
            "Document Description",
            placeholder="This document contains important information about...",
            height=100
        )

        # Upload button with enhanced styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing document... This may take a moment."):
                    try:
                        print(f"=== DOCUMENT PROCESSING STARTED ===")
                        print(f"File name: {uploaded_file.name}")
                        print(f"File size: {uploaded_file.size}")

                        # Update description in database
                        if description.strip():
                            # We'll add this after the document is processed
                            pass

                        # Process the document
                        print(f"Calling document_processor.process_uploaded_file...")
                        document_id, message = document_processor.process_uploaded_file(uploaded_file)
                        print(f"Processing result: document_id={document_id}, message={message}")

                        if document_id:
                            # Update description if provided
                            if description.strip():
                                with sqlite3.connect(Config.DB_PATH) as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        "UPDATE documents SET description = ? WHERE id = ?",
                                        (description.strip(), document_id)
                                    )
                                    conn.commit()

                            st.markdown(f"""
                            <div class="stSuccess">
                                <div style="display: flex; align-items: center; gap: 1rem;">
                                    <div style="font-size: 2rem;">✅</div>
                                    <div>
                                        <div style="font-weight: 600; font-size: 1.1rem;">Document Successfully Processed!</div>
                                        <div style="margin-top: 0.25rem;">{message}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            print(f"Document processed successfully, redirecting to chat")
                            print(f"Setting current_document_id to {document_id}")
                            print(f"Setting page to 'chat'")
                            print(f"About to call st.rerun() from upload success")
                            st.session_state.current_document_id = document_id
                            st.session_state.page = "chat"
                            st.rerun()
                        else:
                            st.markdown(f"""
                            <div class="stError">
                                <div style="display: flex; align-items: center; gap: 1rem;">
                                    <div style="font-size: 2rem;">❌</div>
                                    <div>
                                        <div style="font-weight: 600; font-size: 1.1rem;">Processing Failed</div>
                                        <div style="margin-top: 0.25rem;">{message}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        st.markdown(f"""
                        <div class="stError">
                            <div style="display: flex; align-items: center; gap: 1rem;">
                                <div style="font-size: 2rem;">⚠️</div>
                                <div>
                                    <div style="font-weight: 600; font-size: 1.1rem;">Error Processing Document</div>
                                    <div style="margin-top: 0.25rem;">{str(e)}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Display preview with enhanced styling
        if uploaded_file.type == 'text/plain' or uploaded_file.name.endswith('.txt'):
            st.markdown("""
            <div style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 1.5rem; margin: 2rem 0; border: 2px solid #e8f5e8;">
                <h4 style="color: #4CAF50; margin-bottom: 1rem;">📄 Document Preview</h4>
            </div>
            """, unsafe_allow_html=True)

            try:
                content = uploaded_file.read(500).decode('utf-8')
                st.text_area("Content Preview", content, height=200, disabled=True)
                uploaded_file.seek(0)  # Reset file pointer
            except:
                st.markdown("""
                <div style="background: rgba(255,255,255,0.9); border-radius: 10px; padding: 1rem; margin: 1rem 0;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="font-size: 1.5rem;">⚠️</div>
                        <div>Preview not available for this file type</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_documents_page():
    """Render the documents management page"""
    # # Main Header
    # st.markdown("""
    # <div class="main-header">
    #     <h1>📄 My Documents</h1>
    #     <p style="font-size: 1.2rem; margin: 1rem 0; opacity: 0.9;">
    #         Manage your uploaded documents and start new chat sessions
    #     </p>
    #     <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">📚</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Document Library</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">🔍</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Smart Search</div>
    #         </div>
    #         <div style="text-align: center;">
    #             <div style="font-size: 2rem;">💬</div>
    #             <div style="font-size: 0.9rem; margin-top: 0.5rem;">Quick Chat</div>
    #         </div>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

    # Search functionality with enhanced styling
    st.markdown("""
    <div style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 1.5rem; margin: 2rem 0; border: 2px solid #e8f5e8; backdrop-filter: blur(10px);">
        <h3 style="color: #4CAF50; margin-bottom: 1rem;">🔍 Search Documents</h3>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("Search Documents", placeholder="Search by filename or description...")

    # Get documents
    if search_query:
        documents = db.search_documents(search_query)
        print(f"Search results for '{search_query}': {len(documents)} documents")
    else:
        documents = db.get_documents()
        print(f"All documents: {len(documents)} documents")

    # Debug: Print all document IDs
    if documents:
        print("Document IDs in database:", [doc['id'] for doc in documents])
        print("Document filenames:", [doc['original_filename'] for doc in documents])

    if not documents:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 3rem; margin: 2rem 0; text-align: center; border: 2px solid #e8f5e8;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📭</div>
            <h3 style="color: #4CAF50; margin-bottom: 1rem;">No Documents Found</h3>
            <p style="color: #666; margin-bottom: 2rem;">Upload your first document to get started!</p>
            <div style="background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 50%, #CDDC39 100%); color: white; padding: 1rem 2rem; border-radius: 15px; display: inline-block; box-shadow: 0 8px 25px rgba(76, 175, 80, 0.25);">
                📤 Upload Your First Document
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Document count with enhanced styling
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.95); border-radius: 15px; padding: 1.5rem; margin: 2rem 0; border: 2px solid #e8f5e8;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: #4CAF50; margin-bottom: 0.5rem;">📚 Document Library</h3>
                <p style="color: #666; margin: 0;">Found {len(documents)} document{'s' if len(documents) != 1 else ''}</p>
            </div>
            <div style="background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 50%, #CDDC39 100%); color: white; padding: 1rem 2rem; border-radius: 15px; font-weight: 600; box-shadow: 0 8px 25px rgba(76, 175, 80, 0.25);">
                Total: {len(documents)} Documents
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Debug information section
    with st.expander("🔍 Debug Information", expanded=False):
        st.write("**Session State:**")
        st.write(f"- Current Document ID: {st.session_state.current_document_id}")
        st.write(f"- Current Session ID: {st.session_state.current_session_id}")
        st.write(f"- Current Page: {st.session_state.page}")
        st.write(f"- Chat Messages: {len(st.session_state.chat_messages)}")

        st.write("**Database Documents:**")
        if documents:
            for doc in documents:
                st.write(f"- Document {doc['id']}: {doc['original_filename']} (Processed: {doc['processed']})")
        else:
            st.write("- No documents found")

    # Display documents in a responsive grid
    for i in range(0, len(documents), 2):
        cols = st.columns(2)
        for j, doc in enumerate(documents[i:i+2]):
            with cols[j]:
                render_document_card(doc)


def render_document_card(document: Dict):
    """Render a single document card with enhanced design"""
    print(f"Rendering document card for document ID: {document['id']}, filename: {document['original_filename']}")

    # Status indicator and icon
    status_icon = "✅" if document['processed'] else "⏳"
    file_type_icons = {
        'pdf': '📄',
        'docx': '📝',
        'doc': '📝',
        'txt': '📃',
        'md': '📃',
        'xlsx': '📊',
        'xls': '📊',
        'csv': '📊',
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'png': '🖼️',
        'bmp': '🖼️',
        'tiff': '🖼️'
    }
    file_icon = file_type_icons.get(document['file_type'].lower(), '📄')

    # Card container with simplified design (no preview)
    with st.container():
        st.markdown(f"""
        <div class="document-card" style="
            background: linear-gradient(135deg, #ffffff 0%, #f8fffa 100%);
            border: 2px solid #e8f5e8;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 8px 32px rgba(76, 175, 80, 0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        ">
            <!-- File Icon and Name -->
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="font-size: 2rem;">{file_icon}</span>
                <span style="font-size: 1rem; font-weight: 600; color: #4CAF50;">{document['original_filename']}</span>
            </div>



        """, unsafe_allow_html=True)

        # Enhanced action buttons with better organization
        st.markdown(f"""
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e8f5e8;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.9rem; color: #666; font-weight: 500;">
                        {status_icon} Status: {'Siap Digunakan' if document['processed'] else 'Sedang Diproses'}
                    </span>
                    <span style="font-size: 0.8rem; color: #999;">
                        {document['file_size'] / 1024:.1f} KB
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Action buttons in organized rows
        if document['processed']:
            # Primary actions row - Chat and Detail
            col_chat, col_detail = st.columns([2, 1])

            with col_chat:
                if st.button(
                    "💬 Pilih Dokumen",
                    key=f"chat_{document['id']}",
                    help="Mulai percakapan dengan dokumen ini",
                    use_container_width=True
                ):
                    print(f"=== CHAT BUTTON CLICKED ===")
                    print(f"Document ID: {document['id']}")
                    print(f"Document Filename: {document['original_filename']}")
                    print(f"Document Processed: {document['processed']}")
                    print(f"Current session state before change:")
                    print(f"  - current_document_id: {st.session_state.current_document_id}")
                    print(f"  - current_session_id: {st.session_state.current_session_id}")
                    print(f"  - page: {st.session_state.page}")

                    # Update session state
                    st.session_state.current_document_id = document['id']
                    st.session_state.current_session_id = None
                    st.session_state.chat_messages = []  # Clear chat messages when switching documents
                    st.session_state.page = "chat"

                    print(f"Session state after change:")
                    print(f"  - current_document_id: {st.session_state.current_document_id}")
                    print(f"  - current_session_id: {st.session_state.current_session_id}")
                    print(f"  - page: {st.session_state.page}")
                    print(f"About to call st.rerun()")
                    st.rerun()

            with col_detail:
                if st.button(
                    "ℹ️ Detail",
                    key=f"info_{document['id']}",
                    help="Lihat informasi detail dokumen",
                    use_container_width=True
                ):
                    show_document_info(document['id'])

            # Danger actions row - Delete button (smaller, separate)
            with st.container():
                st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
                if st.button(
                    "🗑️ Hapus Dokumen",
                    key=f"delete_{document['id']}",
                    help="Hapus dokumen ini",
                    use_container_width=True
                ):
                    delete_document(document['id'])
        else:
            # When document is processing
            st.button(
                "⏳ Memproses...",
                key=f"processing_{document['id']}",
                disabled=True,
                use_container_width=True,
                help="Dokumen sedang diproses"
            )

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_document_info(document_id: int):
    """Show detailed information about a document"""
    doc_info = document_processor.get_document_info(document_id)
    stats = rag_pipeline.get_document_statistics(document_id)

    if doc_info:
        with st.expander(f"📊 Document Information", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Basic Info:**")
                st.write(f"- Filename: {doc_info['document']['original_filename']}")
                st.write(f"- Type: {doc_info['document']['file_type'].upper()}")
                st.write(f"- Size: {doc_info['document']['file_size'] / 1024:.1f} KB")
                st.write(f"- Chunks: {doc_info['chunks_count']}")

            with col2:
                st.write("**Processing Info:**")
                st.write(f"- Characters: {doc_info['total_characters']:,}")
                st.write(f"- Vector Store: {'✅' if doc_info['vector_store_exists'] else '❌'}")
                st.write(f"- Processed: {'✅' if doc_info['document']['processed'] else '❌'}")

                # Debug vector store info
                if doc_info['document']['vector_store_path']:
                    st.write(f"- Vector Path: {doc_info['document']['vector_store_path']}")

                # Test vector store loading
                try:
                    if vector_store_manager.load_vector_store(document_id):
                        st.write("✅ Vector Store: Loadable")
                    else:
                        st.write("❌ Vector Store: Not Loadable")
                except Exception as e:
                    st.write(f"❌ Vector Store Error: {str(e)}")

            # Preview
            st.write("**Content Preview:**")
            st.text_area("Document Preview", doc_info['preview'], height=100, disabled=True)

            # Statistics
            if stats:
                st.write("**Usage Statistics:**")
                st.write(f"- Chat Sessions: {stats['chat_sessions_count']}")
                st.write(f"- Total Messages: {stats['total_messages']}")

            # Recent chat sessions for this document
            doc_sessions = db.get_chat_sessions(document_id)
            if doc_sessions:
                st.write("**Recent Chat Sessions:**")
                for session in doc_sessions[:5]:  # Show last 5 sessions
                    message_count = len(db.get_chat_messages(session['id']))
                    st.write(f"- {session['session_name']} ({message_count} messages)")

                    if st.button(
                        f"Continue Chat",
                        key=f"doc_{document_id}_continue_session_{session['id']}",
                        help=f"Last activity: {session['last_activity']}"
                    ):
                        st.session_state.current_document_id = document_id
                        st.session_state.current_session_id = session['id']
                        st.session_state.chat_messages = db.get_chat_messages(session['id'])
                        st.session_state.page = "chat"
                        st.rerun()

            # Key insights
            with st.spinner("Extracting key insights..."):
                insights = rag_pipeline.extract_key_insights(document_id)
                if insights:
                    st.write("**Key Insights:**")
                    for insight in insights:
                        st.markdown(f"""
                        <div class="insight-box">
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)

def delete_document(document_id: int):
    """Delete a document"""
    if st.session_state.get(f"confirm_delete_{document_id}", False):
        # Perform deletion
        try:
            # Delete from database
            db.delete_document(document_id)
            # Delete vector store
            document_processor.delete_document_vector_store(document_id)
            # Delete processed file
            doc = db.get_document(document_id)
            if doc:
                file_path = f"uploads/{doc['filename']}"
                if os.path.exists(file_path):
                    os.remove(file_path)

            st.success("✅ Document deleted successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error deleting document: {str(e)}")
    else:
        # Show confirmation
        st.session_state[f"confirm_delete_{document_id}"] = True
        st.warning("⚠️ Are you sure you want to delete this document? Click Delete again to confirm.")
        st.rerun()

def delete_chat_session(session_id: int):
    """Delete a chat session"""
    if st.session_state.get(f"confirm_delete_session_{session_id}", False):
        # Perform deletion
        try:
            db.delete_chat_session(session_id)

            # Reset session state if this was the current session
            if st.session_state.get('current_session_id') == session_id:
                st.session_state.current_session_id = None
                st.session_state.chat_messages = []

            st.success("✅ Chat session deleted successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error deleting chat session: {str(e)}")
    else:
        # Show confirmation
        st.session_state[f"confirm_delete_session_{session_id}"] = True
        st.warning("⚠️ Are you sure you want to delete this chat session? Click Delete again to confirm.")
        st.rerun()

def render_chat_page():
    """Render the chat interface"""
    print(f"\n=== render_chat_page ===")
    print(f"Session state document ID: {st.session_state.current_document_id}")

    if not st.session_state.current_document_id:
        st.warning("⚠️ Please select a document to chat with. Go to 'My Documents' and choose a document.")
        return

    # Get document info with error handling
    try:
        print(f"Getting document info for ID: {st.session_state.current_document_id}")
        document = db.get_document(st.session_state.current_document_id)
        print(f"Document found: {document is not None}")
        if document:
            print(f"Document filename: {document['original_filename']}")
    except Exception as e:
        print(f"Error getting document: {str(e)}")
        st.error(f"❌ Error accessing document: {str(e)}")
        return

    if not document:
        st.error("❌ Document not found!")
        return

    # Clear chat messages if document ID doesn't match current messages document
    if 'last_document_id' not in st.session_state:
        st.session_state.last_document_id = st.session_state.current_document_id

    if st.session_state.last_document_id != st.session_state.current_document_id:
        st.session_state.chat_messages = []  # Clear messages when switching documents
        st.session_state.last_document_id = st.session_state.current_document_id
        st.session_state.current_session_id = None  # Reset session ID

    # Create two columns: Chat (left) and Log (right)
    col_chat, col_log = st.columns([2, 1])

    with col_chat:
        # Header with document info
        st.header(f"💬 Chat with {document['original_filename']}")
        st.markdown("Ask questions about your document and get AI-powered answers with references.")

        # Validation status
        validation_result = validate_document_vector_store(st.session_state.current_document_id)
        if validation_result['valid']:
            st.success(f"✅ Document Ready: {validation_result['chunk_count']} chunks loaded")
        else:
            st.error(f"❌ Document Error: {validation_result['error']}")

        # Session management
        with st.expander("📜 Chat Sessions", expanded=False):
            if st.button("🆕 New Chat Session", use_container_width=True):
                # Create new session
                try:
                    session_id = db.create_chat_session(
                        st.session_state.current_document_id,
                        f"Chat about {document['original_filename']}"
                    )
                    st.session_state.current_session_id = session_id
                    st.session_state.chat_messages = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating session: {str(e)}")

            # Show existing sessions
            try:
                document_sessions = db.get_chat_sessions(st.session_state.current_document_id)
                if document_sessions:
                    st.write("**Existing Sessions:**")
                    for session in document_sessions:
                        # Create columns for session button and delete button
                        col_session, col_delete = st.columns([4, 1])

                        with col_session:
                            if st.button(
                                f"💬 {session['session_name']}",
                                key=f"session_{session['id']}",
                                help=f"Created: {session['created_at']}",
                                use_container_width=True
                            ):
                                st.session_state.current_session_id = session['id']
                                st.session_state.chat_messages = db.get_chat_messages(session['id'])
                                st.rerun()

                        with col_delete:
                            if st.button(
                                "🗑️",
                                key=f"delete_session_{session['id']}",
                                help="Delete this chat session",
                                use_container_width=True
                            ):
                                delete_chat_session(session['id'])
                else:
                    st.write("No sessions yet. Create one above!")
            except Exception as e:
                st.error(f"Error loading sessions: {str(e)}")

    with col_log:
        st.subheader("📊 Monitoring Log")

        # Create a log area
        log_area = st.container()

        with log_area:
#             st.write("**🔧 Session State:**")
#             st.code(f"""
# Current Document ID: {st.session_state.current_document_id}
# Current Session ID: {st.session_state.current_session_id}
# Chat Messages: {len(st.session_state.chat_messages)}
# Current Page: {st.session_state.page}
# Last Document ID: {st.session_state.get('last_document_id', 'Not set')}
# """, language="text")

            st.write("**📋 Document Status:**")
            st.code(f"""
Document Name: {document['original_filename']}
Document ID: {document['id']}
Processed: {document['processed']}
File Type: {document['file_type']}
File Size: {document['file_size']} bytes
""", language="text")

            # Vector Store Status
            st.write("**🗂️ Vector Store Status:**")
            try:
                vs_info = vector_store_manager.get_vector_store_info(st.session_state.current_document_id)
                if vs_info:
                    st.code(f"""
Vector Store: ✅ Available
Chunks: {vs_info.get('num_documents', 0)}
Created: {vs_info.get('created_at', 'Unknown')}
""", language="text")
                else:
                    st.code("Vector Store: ❌ Not Available", language="text")
            except Exception as e:
                st.code(f"Vector Store Error: {str(e)}", language="text")

            # Real-time validation
            st.write("**🔍 Real-time Validation:**")
            validation_result = validate_document_vector_store(st.session_state.current_document_id)
            if validation_result['valid']:
                st.success("✅ All systems operational")
                st.code(f"""
Document Valid: ✅
Vector Store Loadable: ✅
Search Test: {validation_result.get('search_test_results', 0)} results
Chunks Available: {validation_result.get('chunk_count', 0)}
""", language="text")
            else:
                st.error("❌ System error detected")
                st.code(f"""
Error: {validation_result['error']}
Document Valid: ❌
""", language="text")

#             # Recent logs
#             st.write("**📝 Recent Activity:**")
#             with st.expander("Show Details", expanded=False):
#                 st.write(f"""
# - Page loaded: {datetime.now().strftime('%H:%M:%S')}
# - Document validated: {'✅' if validation_result['valid'] else '❌'}
# - Sessions available: {len(db.get_chat_sessions(st.session_state.current_document_id))}
# - Messages loaded: {len(st.session_state.chat_messages)}
# """)

    # Chat history in left column
    with col_chat:
        # Auto-session management
        try:
            document_sessions = db.get_chat_sessions(st.session_state.current_document_id)
            if not st.session_state.current_session_id and document_sessions:
                # Select the most recent session
                st.session_state.current_session_id = document_sessions[0]['id']
                st.session_state.chat_messages = db.get_chat_messages(document_sessions[0]['id'])
            elif not st.session_state.current_session_id:
                # Create new session if no sessions exist
                session_id = db.create_chat_session(
                    st.session_state.current_document_id,
                    f"Chat about {document['original_filename']}"
                )
                st.session_state.current_session_id = session_id
                st.session_state.chat_messages = []
        except Exception as e:
            st.error(f"Session management error: {str(e)}")

        # Chat history display
        chat_container = st.container()
        with chat_container:
            # Display chat messages
            for message in st.session_state.chat_messages:
                if message['message_type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Show references if available
                    if message.get('metadata', {}).get('references'):
                        with st.expander("📚 References"):
                            for i, ref in enumerate(message['metadata']['references'][:3]):
                                st.markdown(f"""
                                <div class="reference-box">
                                    <strong>Reference {i+1}:</strong> {ref['content'][:200]}...
                                    {f"<br><em>Page: {ref['metadata'].get('page', 'N/A')}</em>" if ref['metadata'].get('page') else ""}
                                </div>
                                """, unsafe_allow_html=True)

        # Chat input at the bottom
        user_input = st.chat_input("Ask a question about your document...", key="chat_input_main")

        if user_input:
            print(f"Chat input received: {user_input}")
            print(f"Current document ID: {st.session_state.current_document_id}")
            print(f"Current session ID: {st.session_state.current_session_id}")

            # Add user message to display
            st.session_state.chat_messages.append({
                'message_type': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })

            # Get AI response with error handling
            with st.spinner("🤔 Thinking..."):
                try:
                    print("Calling rag_pipeline.query_document...")
                    response = rag_pipeline.query_document(
                        st.session_state.current_document_id,
                        user_input,
                        st.session_state.current_session_id
                    )
                    print(f"RAG pipeline response: {response}")

                    if response and response.get('success', True):
                        # Add AI response to display
                        st.session_state.chat_messages.append({
                            'message_type': 'assistant',
                            'content': response['answer'],
                            'timestamp': datetime.now(),
                            'metadata': {'references': response.get('references', [])}
                        })
                    else:
                        # Handle error response
                        error_message = response.get('answer', 'Maaf, terjadi kesalahan saat memproses pertanyaan Anda.')
                        st.session_state.chat_messages.append({
                            'message_type': 'assistant',
                            'content': error_message,
                            'timestamp': datetime.now(),
                            'metadata': {'error': True}
                        })
                        st.error(f"❌ {error_message}")

                except Exception as e:
                    print(f"Error in chat processing: {str(e)}")
                    error_message = "Maaf, terjadi kesalahan sistem. Silakan coba lagi."
                    st.session_state.chat_messages.append({
                        'message_type': 'assistant',
                        'content': error_message,
                        'timestamp': datetime.now(),
                        'metadata': {'error': True}
                    })
                    st.error(f"❌ {error_message}")

            # Save messages to database
            try:
                if st.session_state.current_session_id:
                    # Save user message
                    db.add_chat_message(
                        st.session_state.current_session_id,
                        'user',
                        user_input
                    )
                    # Save assistant message if available
                    if st.session_state.chat_messages and st.session_state.chat_messages[-1]['message_type'] == 'assistant':
                        db.add_chat_message(
                            st.session_state.current_session_id,
                            'assistant',
                            st.session_state.chat_messages[-1]['content'],
                            st.session_state.chat_messages[-1].get('metadata', {})
                        )
            except Exception as e:
                print(f"Error saving messages to database: {str(e)}")

            st.rerun()

    # Sidebar with chat options
    with st.sidebar:
        st.subheader("💬 Chat Options")

        # New chat button
        if st.button("🆕 New Chat Session"):
            # Create new session
            session_id = db.create_chat_session(
                st.session_state.current_document_id,
                f"Chat about {document['original_filename']}"
            )
            st.session_state.current_session_id = session_id
            st.session_state.chat_messages = []
            st.rerun()

        # Document summary
        if st.button("📋 Document Summary"):
            with st.spinner("Generating summary..."):
                summary = rag_pipeline.summarize_document(st.session_state.current_document_id)
                st.info(summary)

        # Key insights
        if st.button("💡 Key Insights"):
            with st.spinner("Extracting insights..."):
                insights = rag_pipeline.extract_key_insights(st.session_state.current_document_id)
                for insight in insights:
                    st.markdown(f"""
                    <div class="insight-box">
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)

  
def render_statistics_page():
    """Render the statistics page"""
    st.header("📊 Statistics & Analytics")
    st.markdown("View detailed statistics about your documents and usage.")

    # API Connection Status
    st.subheader("🔌 API Connection Status")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Google Gemini API (LLM):**")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=Config.LLM_MODEL,
                google_api_key=Config.GOOGLE_API_KEY,
                transport="rest"
            )
            # Test with a simple query
            test_response = llm.invoke("test")
            st.success("✅ Connected")
            st.caption(f"Model: {Config.LLM_MODEL}")
        except Exception as e:
            st.error("❌ Connection Failed")
            st.caption(f"Error: {str(e)}")

    with col2:
        embedding_provider_name = "Cohere" if Config.EMBEDDING_PROVIDER == "cohere" else "Google"
        # st.write(f"**{embedding_provider_name} API (Embeddings):**")
        try:
            if Config.EMBEDDING_PROVIDER == "cohere":
                from langchain_cohere import CohereEmbeddings
                embeddings = CohereEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    cohere_api_key=Config.COHERE_API_KEY
                )
            else:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    google_api_key=Config.GOOGLE_API_KEY,
                    transport="rest"
                )
            # Test with a simple query
            test_embedding = embeddings.embed_query("test")
            # st.success("✅ Connected")
            # st.caption(f"Model: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            st.error("❌ Connection Failed")
            st.caption(f"Error: {str(e)}")

    # Vector Store Status
    st.subheader("🗂️ Vector Store Status")
    documents = db.get_documents()

    if documents:
        vector_store_stats = {
            'total': len(documents),
            'processed': sum(1 for doc in documents if doc['processed']),
            'with_vector_store': 0,
            'loadable': 0
        }

        for doc in documents:
            if doc.get('vector_store_path'):
                vector_store_stats['with_vector_store'] += 1
                try:
                    if vector_store_manager.load_vector_store(doc['id']):
                        vector_store_stats['loadable'] += 1
                except:
                    pass

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Documents", vector_store_stats['total'])
        with col2:
            st.metric("✅ Processed", vector_store_stats['processed'])
        with col3:
            st.metric("🗂️ Has Vector Store", vector_store_stats['with_vector_store'])
        with col4:
            st.metric("💾 Loadable", vector_store_stats['loadable'])
    else:
        st.info("📭 No documents found")

    # API Configuration Status
    st.subheader("🔧 API Configuration Status")

    api_col1, api_col2 = st.columns(2)

    with api_col1:
        st.write("**Google Gemini API (LLM)**")
        if not Config.GOOGLE_API_KEY:
            st.warning("⚠️ Google API Key not configured")
            with st.expander("📖 Setup Instructions"):
                st.markdown("""
                **To get Google API Key:**
                1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
                2. Sign in with your Google account
                3. Click "Create API Key"
                4. Copy the API key
                5. Add to your `.env` file: `GOOGLE_API_KEY=your_key_here`
                6. Restart the application
                """)
        else:
            st.success("✅ Google API Key configured")
            st.caption(f"Current model: {Config.LLM_MODEL}")

    with api_col2:
        # st.write("**API (Embeddings)**")
        if not Config.COHERE_API_KEY:
            st.warning("⚠️ API Key not configured")
            with st.expander("📖 Setup Instructions"):
                st.markdown("""
                **To get Cohere API Key:**
                1. Go to [Cohere Dashboard](https://dashboard.cohere.com/api-keys)
                2. Sign up or sign in to your account
                3. Click "Create API Key"
                4. Copy the API key
                5. Add to your `.env` file: `COHERE_API_KEY=your_key_here`
                6. Restart the application
                """)
        else:
            print("Embedding API Key configured")
            # st.success("")
            # st.caption(f"Current model: {Config.EMBEDDING_MODEL}")

    # Connection test button
    st.markdown("---")
    test_col1, test_col2, test_col3 = st.columns([1, 2, 1])
    with test_col2:
        if st.button("🧪 Test API Connections", use_container_width=True, help="Test both LLM and Embedding API connections"):
            with st.spinner("Testing connections..."):
                try:
                    from config import test_connection
                    results = test_connection()

                    st.markdown("### Connection Test Results")
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        llm_success, llm_message = results.get('llm', (False, 'Not tested'))
                        if llm_success:
                            st.success(f"✅ **LLM Connection**: {llm_message}")
                        else:
                            st.error(f"❌ **LLM Connection**: {llm_message}")

                    with result_col2:
                        embed_success, embed_message = results.get('embedding', (False, 'Not tested'))
                        if embed_success:
                            st.success(f"✅ **Embedding Connection**: {embed_message}")
                        else:
                            st.error(f"❌ **Embedding Connection**: {embed_message}")

                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")

    # Get all documents
    documents = db.get_documents()

    if not documents:
        st.info("📭 No documents available for statistics.")
        return

    # Overall statistics
    st.subheader("📈 Overall Statistics")

    total_docs = len(documents)
    processed_docs = sum(1 for doc in documents if doc['processed'])
    total_size = sum(doc['file_size'] for doc in documents)
    total_sessions = sum(len(db.get_chat_sessions(doc['id'])) for doc in documents)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Total Documents", total_docs)
    with col2:
        st.metric("✅ Processed", processed_docs)
    with col3:
        st.metric("💾 Total Size", f"{total_size / (1024*1024):.1f} MB")
    with col4:
        st.metric("💬 Chat Sessions", total_sessions)

    # Document details
    st.subheader("📄 Document Details")

    document_data = []
    for doc in documents:
        sessions = db.get_chat_sessions(doc['id'])
        total_messages = sum(len(db.get_chat_messages(session['id'])) for session in sessions)
        stats = rag_pipeline.get_document_statistics(doc['id'])

        document_data.append({
            'Document': doc['original_filename'],
            'Type': doc['file_type'].upper(),
            'Size (KB)': f"{doc['file_size'] / 1024:.1f}",
            'Processed': '✅' if doc['processed'] else '❌',
            'Chat Sessions': len(sessions),
            'Total Messages': total_messages,
            'Chunks': stats.get('total_chunks', 'N/A'),
            'Upload Date': doc['upload_time'].split(' ')[0]
        })

    # Display as table
    df = pd.DataFrame(document_data)
    st.dataframe(df, use_container_width=True)

    # Visualizations
    st.subheader("📊 Visualizations")

    # Document types distribution
    doc_types = {}
    for doc in documents:
        doc_type = doc['file_type'].upper()
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    if doc_types:
        st.write("**Document Types Distribution:**")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(doc_types)
        with col2:
            # Pie chart data
            chart_data = pd.DataFrame(list(doc_types.items()), columns=['Type', 'Count'])
            st.write(chart_data)

    # Most active documents
    doc_activity = {}
    for doc in documents:
        sessions = db.get_chat_sessions(doc['id'])
        total_messages = sum(len(db.get_chat_messages(session['id'])) for session in sessions)
        if total_messages > 0:
            doc_activity[doc['original_filename'][:30]] = total_messages

    if doc_activity:
        st.write("**Most Active Documents (by messages):**")
        # Sort by activity
        sorted_activity = dict(sorted(doc_activity.items(), key=lambda x: x[1], reverse=True)[:10])
        st.bar_chart(sorted_activity)

def main():
    """Main application function"""
    try:
        print(f"\n=== Main Application Start ===")
        print(f"Current page: {st.session_state.get('page', 'Not set')}")

        # Initialize session state
        initialize_session_state()

        # Validate configuration
        try:
            validate_config()
        except ValueError as e:
            st.error(f"❌ Configuration Error: {str(e)}")
            st.error("Please check your .env file and ensure GOOGLE_API_KEY is set.")
            return

        # Render sidebar
        render_sidebar()

        # Render current page with error handling
        print(f"Rendering page: {st.session_state.page}")
        if st.session_state.page == 'homepage':
            render_homepage()
        elif st.session_state.page == 'upload':
            render_upload_page()
        elif st.session_state.page == 'documents':
            render_documents_page()
        elif st.session_state.page == 'chat':
            try:
                render_chat_page()
            except Exception as e:
                print(f"Error in chat page: {str(e)}")
                import traceback
                traceback.print_exc()
                st.error(f"❌ Error loading chat page: {str(e)}")
                st.session_state.page = 'documents'  # Redirect to documents page
                st.rerun()
        elif st.session_state.page == 'statistics':
            render_statistics_page()
        else:
            st.error(f"❌ Unknown page: {st.session_state.page}")
            st.session_state.page = 'homepage'
            st.rerun()

    except Exception as e:
        print(f"Unhandled error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"❌ Unexpected error: {str(e)}")
        st.session_state.page = 'upload'
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 10px;'>"
        "🤖 Document Chatbot Heri Santoso | Powered by Google Gemini"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()