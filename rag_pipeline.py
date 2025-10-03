import os
import sqlite3
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import pandas as pd

# LangChain components
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from config import Config
from database import db
from vector_store_manager import vector_store_manager

# Define custom state for RAG
class RAGState(TypedDict):
    messages: List
    context: str
    references: List[Dict]
    query: str

class RAGPipeline:
    """Enhanced RAG Pipeline with LangGraph integration and memory management"""

    def __init__(self):
        # Initialize LLM based on provider selection
        self._initialized = False
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize or reinitialize the LLM with Google Gemini"""
        # Get API key from environment
        api_key = Config.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("Google API Key is required. Please set it in your .env file.")

        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=api_key,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            transport="rest"
        )

        # Initialize memory components
        self.memory_saver = MemorySaver()
        # Using in-memory store as replacement for MemoryStore
        self.memory_store = {}

        # Create prompt templates
        self._create_prompts()
        self._initialized = True

    def reinitialize_if_needed(self):
        """Reinitialize LLM if configuration has changed"""
        if not self._initialized:
            self._initialize_llm()

    def _create_prompts(self):
        """Create prompt templates for RAG"""
        # System prompt for RAG
        self.rag_system_prompt = """Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konten dokumen yang diberikan.

Konteks dari dokumen:
{context}

Petunjuk:
1. Gunakan hanya konteks yang diberikan untuk menjawab pertanyaan
2. Jika konteks tidak mengandung jawabannya, katakan "Saya tidak memiliki informasi yang cukup dalam dokumen untuk menjawab pertanyaan ini"
3. Berikan detail dan kutipan spesifik dari dokumen jika memungkinkan
4. Akurat dan jujur
5. Jika Anda menyebutkan informasi dari dokumen, kutip sumbernya

Riwayat Chat:
{chat_history}

Pertanyaan Pengguna: {question}

Berikan jawaban yang komprehensif berdasarkan konteks dokumen di atas:"""

        # Prompt template for document-specific Q&A
        self.qa_prompt = PromptTemplate(
            template=self.rag_system_prompt,
            input_variables=["context", "question", "chat_history"]
        )

        # Conversation prompt
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Anda adalah asisten AI yang membantu. Anda memiliki akses ke konteks dokumen dan dapat mengingat informasi dari percakapan sebelumnya.

Konteks Dokumen:
{context}

Riwayat Chat Sebelumnya:
{chat_history}"""),
            ("human", "{question}")
        ])

    def create_document_qa_chain(self, document_id: int):
        """Create a QA chain for a specific document"""
        def retrieve_context(state: RAGState) -> Dict:
            """Retrieve relevant context from document"""
            query = state["messages"][-1].content
            print(f"\n--- retrieve_context ---")
            print(f"Query: {query}")

            # Search for relevant documents
            search_results = vector_store_manager.search(document_id, query, k=5)
            # print(f"Search results: {len(search_results)}")

            if search_results:
                context = "\n\n".join([f"Document {i+1}: {result['content']}"
                                     for i, result in enumerate(search_results)])

                # Add reference information
                references = []
                for i, result in enumerate(search_results):
                    ref = {
                        'rank': i + 1,
                        'content': result['content'][:200] + "...",
                        'metadata': result['metadata'],
                        'score': result['score']
                    }
                    references.append(ref)

                # print(f"Context generated: {context[:200]}...")
                # print(f"References: {len(references)}")
            else:
                context = "No relevant information found in the document."
                references = []
                print("No search results found")

            # Return state with context and references (keep original messages unchanged)
            return {
                "messages": state["messages"],  # Keep original messages
                "context": context,
                "references": references,
                "query": query
            }

        def generate_response(state: RAGState) -> Dict:
            """Generate response using LLM with retrieved context"""
            print(f"\n--- generate_response ---")
            print(f"State keys: {list(state.keys())}")
            print(f"Messages count: {len(state['messages'])}")

            # Get context, query, and references directly from state
            context = state.get("context", "")
            query = state.get("query", "")
            references = state.get("references", [])

            # If no query in state, get from last message
            if not query and state["messages"]:
                query = state["messages"][-1].content

            print(f"Extracted Context: {context[:200]}...")
            print(f"Extracted Query: {query}")
            print(f"Extracted References: {len(references)}")

            # Get chat history
            messages = state["messages"]
            chat_history = ""
            if len(messages) > 1:
                # Convert messages to text for chat history
                history_messages = []
                for msg in messages[:-1]:  # Exclude the last message (current query)
                    if hasattr(msg, 'content'):
                        if isinstance(msg, HumanMessage):
                            history_messages.append(f"User: {msg.content}")
                        elif isinstance(msg, AIMessage):
                            history_messages.append(f"Assistant: {msg.content}")

                if history_messages:
                    chat_history = "\n".join(history_messages)

            # Create prompt with context
            prompt = self.qa_prompt.format(
                context=context,
                question=query,
                chat_history=chat_history
            )

            print(f"Generated prompt length: {len(prompt)}")
            print(f"Prompt starts with: {prompt[:300]}...")

            # Generate response
            response = self.llm.invoke(prompt)
            print(f"LLM response: {response.content[:200]}...")

            # Format response with references
            if references:
                refs_text = "\n\n**References:**\n"
                for i, ref in enumerate(references[:3]):  # Show top 3 references
                    refs_text += f"\n{i+1}. {ref['content']}"
                    if ref['metadata'].get('page'):
                        refs_text += f" (Page {ref['metadata']['page']})"

                full_response = response.content + refs_text
            else:
                full_response = response.content

            return {
                "messages": [AIMessage(content=full_response)],
                "references": references
            }

        # Create the graph
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)

        # Add edges
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)

        # Compile with memory
        chain = workflow.compile(checkpointer=self.memory_saver)

        return chain

    def query_document(self, document_id: int, question: str, session_id: int,
                      thread_id: str = "default") -> Dict[str, Any]:
        """Query a document with RAG"""
        print(f"\n=== query_document ===")
        print(f"Document ID: {document_id}")
        print(f"Question: {question}")
        print(f"Session ID: {session_id}")

        # Ensure LLM is properly initialized with current API key
        self.reinitialize_if_needed()

        try:
            # Check if vector store is available first
            if not vector_store_manager.load_vector_store(document_id):
                error_msg = "Vector store tidak tersedia untuk dokumen ini. Silakan upload ulang dokumen."
                print(error_msg)
                db.add_chat_message(session_id, "assistant", error_msg)
                return {
                    "answer": error_msg,
                    "references": [],
                    "success": False,
                    "error": "Vector store not available"
                }

            # Test search functionality first
            try:
                test_search = vector_store_manager.search(document_id, "test", k=1)
                if not test_search:
                    error_msg = "Tidak dapat melakukan pencarian pada vector store. Dokumen mungkin kosong atau rusak."
                    print(error_msg)
                    db.add_chat_message(session_id, "assistant", error_msg)
                    return {
                        "answer": error_msg,
                        "references": [],
                        "success": False,
                        "error": "Search returned no results"
                    }
            except Exception as search_error:
                error_msg = f"Error testing vector store search: {str(search_error)}"
                print(error_msg)

                # Check if it's a connection error
                if "getaddrinfo failed" in str(search_error) or "Network connection error" in str(search_error):
                    error_msg = "❌ **Koneksi ke Embedding API Gagal**\n\nTidak dapat terhubung ke layanan embedding. Silakan:\n1. Periksa koneksi internet Anda\n2. Coba lagi beberapa saat\n3. Jika masalah berlanjut, hubungi administrator\n\nError detail: " + str(search_error)

                db.add_chat_message(session_id, "assistant", error_msg)
                return {
                    "answer": error_msg,
                    "references": [],
                    "success": False,
                    "error": str(search_error)
                }

            # Create QA chain for the document
            qa_chain = self.create_document_qa_chain(document_id)

            # Get chat history from database
            chat_messages = db.get_chat_messages(session_id)
            messages = []

            # Convert chat messages to LangChain format
            for msg in chat_messages:
                if msg['message_type'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                else:
                    messages.append(AIMessage(content=msg['content']))

            # Add current question
            messages.append(HumanMessage(content=question))

            # Create state with messages
            state = {"messages": messages}

            # Configure thread for memory
            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            # Run the chain with error handling
            try:
                print(f"Invoking LangGraph chain with thread_id: {thread_id}")
                result = qa_chain.invoke(state, config=config)
                print(f"Chain execution completed successfully")
            except Exception as chain_error:
                print(f"Error in LangGraph chain execution: {str(chain_error)}")
                import traceback
                traceback.print_exc()

                error_msg = f"Error dalam RAG pipeline: {str(chain_error)}"
                db.add_chat_message(session_id, "assistant", error_msg)
                return {
                    "answer": error_msg,
                    "references": [],
                    "success": False,
                    "error": f"Chain execution failed: {str(chain_error)}"
                }

            # Extract response and references
            try:
                if result["messages"]:
                    response_content = result["messages"][-1].content
                    print(f"Response extracted: {response_content[:100]}...")
                else:
                    response_content = "Maaf, tidak dapat menghasilkan respons. Silakan coba lagi."
                    print("No messages in result")
            except Exception as extract_error:
                print(f"Error extracting response: {str(extract_error)}")
                response_content = f"Error extracting response: {str(extract_error)}"

            references = result.get("references", [])

            # Save to database
            db.add_chat_message(session_id, "user", question)
            db.add_chat_message(
                session_id,
                "assistant",
                response_content,
                metadata={"references": references}
            )

            return {
                "answer": response_content,
                "references": references,
                "success": True
            }

        except Exception as e:
            error_msg = f"Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"
            print(error_msg)

            # Check if it's a connection error
            if "getaddrinfo failed" in str(e) or "Network connection error" in str(e):
                error_msg = "❌ **Koneksi ke Embedding API Gagal**\n\nTidak dapat terhubung ke layanan embedding. Silakan:\n1. Periksa koneksi internet Anda\n2. Coba lagi beberapa saat\n3. Jika masalah berlanjut, hubungi administrator\n\nError detail: " + str(e)

            # Save error to database
            db.add_chat_message(session_id, "assistant", error_msg)

            return {
                "answer": error_msg,
                "references": [] ,
                "success": False,
                "error": str(e)
            }

    def get_similar_documents(self, document_id: int, query: str, k: int = 5) -> List[Dict]:
        """Get similar documents without generating a response"""
        try:
            results = vector_store_manager.search(document_id, query, k)

            formatted_results = []
            for result in results:
                formatted_result = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'score': result['score'],
                    'rank': result['rank']
                }
                formatted_results.append(formatted_result)

            return formatted_results

        except Exception as e:
            print(f"Terjadi kesalahan mendapatkan dokumen serupa: {str(e)}")
            return []

    def extract_key_insights(self, document_id: int, num_insights: int = 5) -> List[str]:
        """Extract key insights from a document"""
        try:
            # Load vector store to get document content
            if not vector_store_manager.load_vector_store(document_id):
                return ["Tidak dapat memuat dokumen untuk analisis"]

            # Get document info
            document = db.get_document(document_id)
            if not document:
                return ["Dokumen tidak ditemukan"]

            # Create a prompt to extract insights
            insights_prompt = f"""
            Berdasarkan dokumen "{document['original_filename']}", silakan ekstrak {num_insights} wawasan kunci atau poin utama.

            Fokus pada:
            1. Topik atau tema utama
            2. Temuan atau kesimpulan penting
            3. Data atau statistik kunci yang disebutkan
            4. Rekomendasi atau takeaways signifikan

            Silakan berikan setiap wawasan sebagai poin terpisah yang ringkas.
            """

            # Get some sample content from the document
            sample_results = vector_store_manager.search(document_id, "main topics summary", k=10)
            if sample_results:
                context = "\n\n".join([result['content'] for result in sample_results[:5]])

                full_prompt = f"Document Content Sample:\n{context}\n\n{insights_prompt}"

                response = self.llm.invoke(full_prompt)
                insights_text = response.content

                # Split into individual insights
                insights = []
                for line in insights_text.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or
                               line.startswith('*') or line[0].isdigit() + '.' in line[:3]):
                        insights.append(line)
                    elif line and len(line) > 20:  # Substantial content
                        insights.append(f"• {line}")

                return insights[:num_insights] if insights else ["No specific insights could be extracted"]
            else:
                return ["Tidak ada konten yang tersedia untuk analisis"]

        except Exception as e:
            print(f"Terjadi kesalahan mengekstrak wawasan: {str(e)}")
            return [f"Terjadi kesalahan mengekstrak wawasan: {str(e)}"]

    def summarize_document(self, document_id: int, max_length: int = 300) -> str:
        """Generate a summary of the document"""
        try:
            document = db.get_document(document_id)
            if not document:
                return "Dokumen tidak ditemukan"

            # Get diverse content from the document
            search_queries = ["ringkasan utama", "poin kunci", "kesimpulan", "pendahuluan", "ikhtisar"]
            all_content = []

            for query in search_queries:
                results = vector_store_manager.search(document_id, query, k=3)
                for result in results:
                    if result['content'] not in all_content:
                        all_content.append(result['content'])

            if not all_content:
                return "Tidak ada konten yang tersedia untuk ringkasan"

            combined_content = "\n\n".join(all_content[:10])  # Limit content

            summary_prompt = f"""
            Silakan berikan ringkasan komprehensif dari konten dokumen berikut.
            Ringkasan harus sekitar {max_length} kata dan menyoroti poin-poin utama.

            Dokumen: {document['original_filename']}

            Konten:
            {combined_content}

            Ringkasan:
            """

            response = self.llm.invoke(summary_prompt)
            return response.content

        except Exception as e:
            print(f"Terjadi kesalahan membuat ringkasan: {str(e)}")
            return f"Terjadi kesalahan membuat ringkasan: {str(e)}"

    def chat_with_memory(self, message: str, session_id: int, document_id: int = None) -> Dict[str, Any]:
        """Chat with memory about a document or general conversation"""
        try:
            # Get user memory
            user_memory = db.get_memory(session_id)
            memory_context = ""
            if user_memory:
                memory_context = "\nUser Information:\n" + "\n".join([f"- {k}: {v}" for k, v in user_memory.items()])

            # Get document context if provided
            document_context = ""
            if document_id:
                doc_results = vector_store_manager.search(document_id, message, k=3)
                if doc_results:
                    document_context = "\nDocument Context:\n" + "\n".join([result['content'] for result in doc_results])

            # Get chat history
            chat_messages = db.get_chat_messages(session_id)
            chat_history = ""
            if chat_messages:
                recent_messages = chat_messages[-5:]  # Last 5 messages
                chat_history = "\nRecent Conversation:\n" + "\n".join([
                    f"{msg['message_type']}: {msg['content']}"
                    for msg in recent_messages
                ])

            # Create comprehensive prompt
            full_prompt = f"""Anda adalah asisten AI yang membantu dengan kemampuan memori. Anda mengingat percakapan dan informasi pengguna sebelumnya.

{memory_context}

{document_context}

{chat_history}

Pesan saat ini: {message}

Silakan merespons dengan membantu, mengacu pada konteks sebelumnya jika relevan. Jika Anda mempelajari informasi baru tentang pengguna, pertimbangkan untuk percakapan masa depan."""

            response = self.llm.invoke(full_prompt)

            # Save to database
            db.add_chat_message(session_id, "user", message)
            db.add_chat_message(session_id, "assistant", response.content)

            return {
                "response": response.content,
                "success": True
            }

        except Exception as e:
            error_msg = f"Terjadi kesalahan dalam chat: {str(e)}"
            print(error_msg)
            return {
                "response": error_msg,
                "success": False,
                "error": str(e)
            }

    def save_user_memory(self, session_id: int, key: str, value: str) -> bool:
        """Save information to user memory"""
        try:
            db.save_memory(session_id, key, value)
            return True
        except Exception as e:
            print(f"Terjadi kesalahan menyimpan memori: {str(e)}")
            return False

    def get_document_statistics(self, document_id: int) -> Dict:
        """Get statistics about document usage and content"""
        try:
            # Get document info
            document = db.get_document(document_id)
            if not document:
                return {}

            # Get vector store statistics
            vector_stats = vector_store_manager.get_statistics(document_id)

            # Get chat sessions for this document
            sessions = db.get_chat_sessions(document_id)

            # Count total messages
            total_messages = 0
            for session in sessions:
                messages = db.get_chat_messages(session['id'])
                total_messages += len(messages)

            # Get document chunks info
            chunks = db.get_document_chunks(document_id)

            stats = {
                'document': document,
                'vector_statistics': vector_stats or {},
                'chat_sessions_count': len(sessions),
                'total_messages': total_messages,
                'total_chunks': len(chunks),
                'total_characters': sum(len(chunk['content']) for chunk in chunks),
                'file_size_mb': round(document['file_size'] / (1024 * 1024), 2) if document['file_size'] else 0
            }

            return stats

        except Exception as e:
            print(f"Terjadi kesalahan mendapatkan statistik dokumen: {str(e)}")
            return {}

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()