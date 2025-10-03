import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import os

class Database:
    def __init__(self, db_path: str = "chatbot.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    vector_store_path TEXT
                )
            """)

            # Chat sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    session_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # Chat messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    message_type TEXT NOT NULL CHECK (message_type IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
                )
            """)

            # User memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE,
                    UNIQUE(session_id, memory_key)
                )
            """)

            # Document chunks table for better tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_upload_time ON documents(upload_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_document_id ON chat_sessions(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_memory_session_id ON user_memory(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)")

            conn.commit()

    def add_document(self, filename: str, original_filename: str, file_type: str,
                    file_size: int, description: str = None) -> int:
        """Add a new document to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (filename, original_filename, file_type, file_size, description)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, original_filename, file_type, file_size, description))
            return cursor.lastrowid

    def get_documents(self) -> List[Dict]:
        """Get all documents"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, original_filename, file_type, file_size,
                       upload_time, description, processed
                FROM documents
                ORDER BY upload_time DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_document(self, document_id: int) -> Optional[Dict]:
        """Get a specific document"""
        print(f"Database: Looking for document ID: {document_id}")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, original_filename, file_type, file_size,
                       upload_time, description, processed, vector_store_path
                FROM documents
                WHERE id = ?
            """, (document_id,))
            row = cursor.fetchone()
            result = dict(row) if row else None
            print(f"Database: Document {document_id} found: {result is not None}")
            if result:
                print(f"Database: Document filename: {result['original_filename']}")
            return result

    def update_document_processed(self, document_id: int, vector_store_path: str):
        """Mark document as processed and save vector store path"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE documents
                SET processed = TRUE, vector_store_path = ?
                WHERE id = ?
            """, (vector_store_path, document_id))

    def delete_document(self, document_id: int):
        """Delete a document and all related data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def create_chat_session(self, document_id: int, session_name: str = None) -> int:
        """Create a new chat session for a document"""
        if not session_name:
            document = self.get_document(document_id)
            session_name = f"Chat about {document['original_filename']}" if document else "New Chat"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_sessions (document_id, session_name)
                VALUES (?, ?)
            """, (document_id, session_name))
            return cursor.lastrowid

    def get_chat_sessions(self, document_id: int) -> List[Dict]:
        """Get all chat sessions for a document"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, session_name, created_at, last_activity
                FROM chat_sessions
                WHERE document_id = ?
                ORDER BY last_activity DESC
            """, (document_id,))
            return [dict(row) for row in cursor.fetchall()]

    def delete_chat_session(self, session_id: int):
        """Delete a chat session and all related messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

    def add_chat_message(self, session_id: int, message_type: str, content: str,
                        metadata: Dict = None):
        """Add a chat message"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute("""
                INSERT INTO chat_messages (session_id, message_type, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, message_type, content, metadata_json))

            # Update last activity
            cursor.execute("""
                UPDATE chat_sessions
                SET last_activity = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (session_id,))

    def get_chat_messages(self, session_id: int) -> List[Dict]:
        """Get all messages for a chat session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, message_type, content, metadata, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            results = []
            for row in cursor.fetchall():
                message = dict(row)
                if message['metadata']:
                    message['metadata'] = json.loads(message['metadata'])
                results.append(message)
            return results

    def save_memory(self, session_id: int, memory_key: str, memory_value: str):
        """Save or update user memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO user_memory (session_id, memory_key, memory_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (session_id, memory_key, memory_value))

    def get_memory(self, session_id: int) -> Dict[str, str]:
        """Get all memory for a session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT memory_key, memory_value
                FROM user_memory
                WHERE session_id = ?
            """, (session_id,))
            return {row['memory_key']: row['memory_value'] for row in cursor.fetchall()}

    def add_document_chunks(self, document_id: int, chunks: List[str], metadata: List[Dict] = None):
        """Add document chunks to track processing"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                chunk_metadata = json.dumps(metadata[i]) if metadata and i < len(metadata) else None
                cursor.execute("""
                    INSERT INTO document_chunks (document_id, chunk_index, content, metadata)
                    VALUES (?, ?, ?, ?)
                """, (document_id, i, chunk, chunk_metadata))

    def get_document_chunks(self, document_id: int) -> List[Dict]:
        """Get all chunks for a document"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_index, content, metadata
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
            """, (document_id,))
            results = []
            for row in cursor.fetchall():
                chunk = dict(row)
                if chunk['metadata']:
                    chunk['metadata'] = json.loads(chunk['metadata'])
                results.append(chunk)
            return results

    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by filename or description"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, original_filename, file_type, file_size,
                       upload_time, description, processed
                FROM documents
                WHERE original_filename LIKE ? OR description LIKE ?
                ORDER BY upload_time DESC
            """, (f"%{query}%", f"%{query}%"))
            return [dict(row) for row in cursor.fetchall()]

# Global database instance
db = Database()