#!/usr/bin/env python3
"""
Document Chatbot with RAG
A Streamlit application for chatting with documents using RAG (Retrieval-Augmented Generation)
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import validate_config
from database import db

def main():
    """Entry point for the application"""
    try:
        # Validate configuration
        validate_config()

        # Initialize database
        db.init_database()

        # Import and run the main app
        from app import main as app_main
        app_main()

    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.error("Please check your configuration and try again.")

if __name__ == "__main__":
    main()