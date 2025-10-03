import os
import pickle
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from langchain.schema import Document as LangchainDocument
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import Config
from database import db

class VectorStoreManager:
    """Enhanced FAISS vector store manager with additional features"""

    def __init__(self):
        self.embeddings = None
        self.vector_stores = {}  # In-memory cache
        self._initialized = False
        # Set embedding dimension based on provider
        if Config.EMBEDDING_PROVIDER == "google":
            self.embedding_dimension = 768  # Google embeddings dimension
        elif Config.EMBEDDING_PROVIDER == "cohere":
            self.embedding_dimension = 1024  # Cohere embeddings dimension
        else:
            self.embedding_dimension = 768  # Default
            print(f"Warning: Unknown embedding provider {Config.EMBEDDING_PROVIDER}, using default dimension.")
        self._init_embeddings()

    def reinitialize_if_needed(self):
        """Reinitialize embeddings if configuration has changed"""
        if not self._initialized or self.embeddings is None:
            self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with error handling"""
        try:
            if Config.EMBEDDING_PROVIDER == "google":
                # Get API key from environment
                api_key = Config.GOOGLE_API_KEY
                if not api_key:
                    raise ValueError("Google API Key is required. Please set it in your .env file.")

                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    google_api_key=api_key,
                    transport="rest"
                )
                print("Google embeddings initialized successfully")
            elif Config.EMBEDDING_PROVIDER == "cohere":
                # Get API key from environment
                api_key = Config.COHERE_API_KEY
                if not api_key:
                    raise ValueError("Cohere API Key is required. Please set it in your .env file.")

                self.embeddings = CohereEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    cohere_api_key=api_key
                )
                print("Cohere embeddings initialized successfully")
            else:
                raise ValueError(f"Unsupported embedding provider: {Config.EMBEDDING_PROVIDER}. Supported providers: google, cohere.")
        except Exception as e:
            print(f"Failed to initialize embeddings ({Config.EMBEDDING_PROVIDER}): {str(e)}")
            self.embeddings = None
            self._initialized = False
        else:
            self._initialized = True

    def _check_embeddings_available(self):
        """Check if embeddings are available"""
        if not self.embeddings:
            try:
                self._init_embeddings()
            except Exception as e:
                provider_name = Config.EMBEDDING_PROVIDER.capitalize()
                raise Exception(f"{provider_name} embeddings not available: {str(e)}. Please check your internet connection and API key.")
        return True

    def _embed_with_retry(self, texts, max_retries=3):
        """Embed texts with retry logic"""
        for attempt in range(max_retries):
            try:
                self._check_embeddings_available()
                if isinstance(texts, str):
                    return self.embeddings.embed_query(texts)
                else:
                    return self.embeddings.embed_documents(texts)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to embed after {max_retries} attempts: {str(e)}")
                print(f"Embedding attempt {attempt + 1} failed, retrying... Error: {str(e)}")
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

    def create_vector_store(self, document_id: int, documents: List[LangchainDocument]) -> str:
        """Create a new FAISS vector store for documents"""
        if not documents:
            raise ValueError("No documents provided to create vector store")

        try:
            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Create embeddings with retry logic
            print(f"Creating embeddings for {len(texts)} document chunks...")
            embeddings = self._embed_with_retry(texts)
            print(f"Embeddings created with dimension: {len(embeddings[0]) if embeddings else 0}")

            # Create FAISS index
            index = faiss.IndexFlatL2(self.embedding_dimension)
            index.add(np.array(embeddings).astype('float32'))

            # Create vector store directory
            vector_store_path = f"{Config.VECTOR_STORE_PATH}/{document_id}"
            os.makedirs(vector_store_path, exist_ok=True)

            # Save FAISS index
            faiss.write_index(index, os.path.join(vector_store_path, "index.faiss"))

            # Save metadata
            vector_store_data = {
                'texts': texts,
                'metadatas': metadatas,
                'document_id': document_id,
                'embedding_model': Config.EMBEDDING_MODEL,
                'embedding_dimension': self.embedding_dimension
            }

            with open(os.path.join(vector_store_path, "store.pkl"), 'wb') as f:
                pickle.dump(vector_store_data, f)

            # Save additional info as JSON for easy access
            info = {
                'document_id': document_id,
                'num_documents': len(documents),
                'embedding_dimension': self.embedding_dimension,
                'embedding_model': Config.EMBEDDING_MODEL,
                'created_at': str(pd.Timestamp.now())
            }

            with open(os.path.join(vector_store_path, "info.json"), 'w') as f:
                json.dump(info, f, indent=2)

            # Cache in memory
            self.vector_stores[document_id] = {
                'index': index,
                'texts': texts,
                'metadatas': metadatas
            }

            return vector_store_path

        except Exception as e:
            # Clean up on failure
            if 'vector_store_path' in locals() and os.path.exists(vector_store_path):
                import shutil
                shutil.rmtree(vector_store_path)
            raise Exception(f"Failed to create vector store: {str(e)}")

    def load_vector_store(self, document_id: int) -> bool:
        """Load vector store from disk"""
        if document_id in self.vector_stores:
            return True

        vector_store_path = f"{Config.VECTOR_STORE_PATH}/{document_id}"
        if not os.path.exists(vector_store_path):
            return False

        try:
            # Load FAISS index
            index_path = os.path.join(vector_store_path, "index.faiss")
            store_path = os.path.join(vector_store_path, "store.pkl")

            if not os.path.exists(index_path) or not os.path.exists(store_path):
                return False

            index = faiss.read_index(index_path)

            with open(store_path, 'rb') as f:
                vector_store_data = pickle.load(f)

            self.vector_stores[document_id] = {
                'index': index,
                'texts': vector_store_data['texts'],
                'metadatas': vector_store_data['metadatas']
            }

            return True

        except Exception as e:
            print(f"Error loading vector store for document {document_id}: {str(e)}")
            return False

    def search(self, document_id: int, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Ensure embeddings are properly initialized with current API key
        self.reinitialize_if_needed()

        if not self.load_vector_store(document_id):
            return []

        try:
            vector_store = self.vector_stores[document_id]

            # Create query embedding with retry
            query_embedding = self._embed_with_retry(query)
            query_vector = np.array([query_embedding]).astype('float32')

            # Search
            distances, indices = vector_store['index'].search(query_vector, min(k, len(vector_store['texts'])))

            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(vector_store['texts']):
                    result = {
                        'content': vector_store['texts'][idx],
                        'metadata': vector_store['metadatas'][idx],
                        'score': float(dist),
                        'rank': i + 1
                    }
                    results.append(result)

            return results

        except Exception as e:
            error_msg = f"Error searching vector store: {str(e)}"
            print(error_msg)
            # Return more informative error for connection issues
            if "getaddrinfo failed" in str(e):
                raise Exception(f"Network connection error to Cohere API. Please check your internet connection and try again. Error: {str(e)}")
            return []

    def similarity_search_with_score(self, document_id: int, query: str, k: int = 4) -> List[Tuple[LangchainDocument, float]]:
        """Search with LangChain Document format and scores"""
        results = self.search(document_id, query, k)

        langchain_results = []
        for result in results:
            doc = LangchainDocument(
                page_content=result['content'],
                metadata=result['metadata']
            )
            langchain_results.append((doc, result['score']))

        return langchain_results

    def add_documents(self, document_id: int, documents: List[LangchainDocument]) -> bool:
        """Add new documents to existing vector store"""
        if not documents:
            return False

        if not self.load_vector_store(document_id):
            # Create new vector store if it doesn't exist
            self.create_vector_store(document_id, documents)
            return True

        try:
            vector_store = self.vector_stores[document_id]

            # Extract texts and metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Create embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Add to index
            vector_store['index'].add(np.array(embeddings).astype('float32'))

            # Update texts and metadatas
            vector_store['texts'].extend(texts)
            vector_store['metadatas'].extend(metadatas)

            # Save updated vector store
            vector_store_path = f"{Config.VECTOR_STORE_PATH}/{document_id}"

            # Save FAISS index
            faiss.write_index(vector_store['index'], os.path.join(vector_store_path, "index.faiss"))

            # Save updated metadata
            vector_store_data = {
                'texts': vector_store['texts'],
                'metadatas': vector_store['metadatas'],
                'document_id': document_id,
                'embedding_model': Config.EMBEDDING_MODEL,
                'embedding_dimension': self.embedding_dimension
            }

            with open(os.path.join(vector_store_path, "store.pkl"), 'wb') as f:
                pickle.dump(vector_store_data, f)

            # Update info
            info = {
                'document_id': document_id,
                'num_documents': len(vector_store['texts']),
                'embedding_dimension': self.embedding_dimension,
                'embedding_model': Config.EMBEDDING_MODEL,
                'updated_at': str(pd.Timestamp.now())
            }

            with open(os.path.join(vector_store_path, "info.json"), 'w') as f:
                json.dump(info, f, indent=2)

            return True

        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False

    def delete_vector_store(self, document_id: int) -> bool:
        """Delete vector store for a document"""
        try:
            # Remove from cache
            if document_id in self.vector_stores:
                del self.vector_stores[document_id]

            # Delete from disk
            vector_store_path = f"{Config.VECTOR_STORE_PATH}/{document_id}"
            if os.path.exists(vector_store_path):
                import shutil
                shutil.rmtree(vector_store_path)

            return True

        except Exception as e:
            print(f"Error deleting vector store: {str(e)}")
            return False

    def get_vector_store_info(self, document_id: int) -> Optional[Dict]:
        """Get information about a vector store"""
        vector_store_path = f"{Config.VECTOR_STORE_PATH}/{document_id}"
        info_path = os.path.join(vector_store_path, "info.json")

        if not os.path.exists(info_path):
            return None

        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading vector store info: {str(e)}")
            return None

    def list_vector_stores(self) -> List[Dict]:
        """List all available vector stores"""
        vector_stores = []

        if not os.path.exists(Config.VECTOR_STORE_PATH):
            return vector_stores

        for item in os.listdir(Config.VECTOR_STORE_PATH):
            item_path = os.path.join(Config.VECTOR_STORE_PATH, item)
            if os.path.isdir(item_path):
                info = self.get_vector_store_info(int(item))
                if info:
                    vector_stores.append(info)

        return vector_stores

    def rebuild_vector_store(self, document_id: int) -> bool:
        """Rebuild vector store from database chunks"""
        try:
            # Get document chunks from database
            chunks = db.get_document_chunks(document_id)
            if not chunks:
                return False

            # Create LangChain documents
            documents = []
            for chunk in chunks:
                doc = LangchainDocument(
                    page_content=chunk['content'],
                    metadata=chunk.get('metadata', {})
                )
                documents.append(doc)

            # Delete existing vector store
            self.delete_vector_store(document_id)

            # Create new vector store
            self.create_vector_store(document_id, documents)

            return True

        except Exception as e:
            print(f"Error rebuilding vector store: {str(e)}")
            return False

    def get_statistics(self, document_id: int) -> Optional[Dict]:
        """Get statistics for a vector store"""
        if not self.load_vector_store(document_id):
            return None

        vector_store = self.vector_stores[document_id]

        return {
            'document_id': document_id,
            'num_vectors': len(vector_store['texts']),
            'embedding_dimension': self.embedding_dimension,
            'index_type': 'IndexFlatL2',
            'memory_cached': True
        }

# Global vector store manager instance
vector_store_manager = VectorStoreManager()