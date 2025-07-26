"""
Insurance Document Query System - Streamlit App
Complete pipeline for parsing documents, indexing, and querying with LLM evaluation.
"""
import streamlit as st
import os
import json
import time
from typing import List, Dict, Any
import pandas as pd

# Import our modules
from src.parse_documents import load_and_parse_documents
from src.chunk_documents import chunk_documents
from src.embed_and_index import index_chunks_in_pinecone, clear_pinecone_index, get_index_stats
from src.query_processor import QueryProcessor
from src.performance_monitor import PerformanceMonitor, estimate_indexing_time

# Page configuration
st.set_page_config(
    page_title="Insurance Document Query System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏥 Insurance Document Query System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        pinecone_api_key = st.text_input("Pinecone API Key", type="password", 
                                       value=st.secrets.get("PINECONE_API_KEY", ""))
        gemini_api_key = st.text_input("Gemini API Key", type="password",
                                     value=st.secrets.get("GEMINI_API_KEY", ""))
        
        # Index settings
        index_name = st.text_input("Pinecone Index Name", value="policy-index")
        
        st.markdown("---")
        
        # Document management
        st.header("📄 Document Management")
        
        # Check if documents are parsed
        if 'documents_parsed' not in st.session_state:
            st.session_state.documents_parsed = False
        
        if 'documents_indexed' not in st.session_state:
            st.session_state.documents_indexed = False
        
        # Parse documents button
        if st.button("🔍 Parse Documents", help="Parse PDF documents from docs/ folder"):
            parse_documents()
        
        # Index documents button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Index Documents", help="Create vector embeddings and index in Pinecone"):
                if not pinecone_api_key:
                    st.error("Please provide Pinecone API Key")
                else:
                    index_documents(pinecone_api_key, index_name)
        
        with col2:
            if st.button("🔄 Smart Re-index", help="Clear index first, then re-index to avoid duplicates"):
                if not pinecone_api_key:
                    st.error("Please provide Pinecone API Key")
                else:
                    smart_reindex(pinecone_api_key, index_name)
        
        # Index management section
        st.markdown("---")
        st.header("🗂️ Index Management")
        
        # Show current index stats
        if pinecone_api_key:
            if st.button("📊 Check Index Stats", help="View current Pinecone index statistics"):
                check_index_stats(pinecone_api_key, index_name)
        
        # Clear index button with warning
        if st.button("🗑️ Clear Index", help="⚠️ Delete all vectors from Pinecone index"):
            if not pinecone_api_key:
                st.error("Please provide Pinecone API Key")
            else:
                clear_index_with_confirmation(pinecone_api_key, index_name)
        
        # Status indicators with more details
        st.markdown("### Status")
        col1, col2 = st.columns(2)
        with col1:
            status_icon = "✅" if st.session_state.documents_parsed else "❌"
            st.markdown(f"{status_icon} **Documents Parsed:** {st.session_state.documents_parsed}")
            
        with col2:
            status_icon = "✅" if st.session_state.documents_indexed else "❌"
            st.markdown(f"{status_icon} **Documents Indexed:** {st.session_state.documents_indexed}")
        
        # Show indexing statistics if available
        if st.session_state.documents_indexed and 'total_chunks' in st.session_state:
            st.markdown("### Indexing Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", st.session_state.total_chunks)
            with col2:
                if 'indexed_count' in st.session_state:
                    st.metric("Vectors Indexed", st.session_state.indexed_count)
            
            # Show last indexing performance
            if 'last_indexing_metrics' in st.session_state:
                metrics = st.session_state.last_indexing_metrics
                if metrics and 'duration_seconds' in metrics:
                    st.metric("Last Indexing Time", f"{metrics['duration_seconds']:.1f}s")
            
            # Quick index health check
            if pinecone_api_key:
                try:
                    stats = get_index_stats(pinecone_api_key, index_name)
                    if stats['exists']:
                        total_vectors = stats['total_vector_count']
                        expected_chunks = st.session_state.get('total_chunks', 0)
                        
                        if total_vectors > expected_chunks * 1.5:
                            st.warning(f"⚠️ **Duplicate Detection**: {total_vectors:,} vectors in index (expected ~{expected_chunks}). Consider using 'Smart Re-index'.")
                except:
                    pass
                    
        # Performance tips
        with st.expander("⚡ Performance Tips"):
            st.markdown("""
            **Optimizations Applied:**
            1. ✅ **Batch Processing**: Embeddings processed in batches of 32
            2. ✅ **Model Caching**: Embedding model cached between operations
            3. ✅ **Progress Tracking**: Real-time progress with time estimates
            4. ✅ **Optimized Chunking**: Better text segmentation with 100-char overlap
            5. ✅ **Memory Monitoring**: Track memory usage and provide recommendations
            6. ✅ **Error Handling**: Robust error recovery with detailed debugging
            7. ✅ **Deterministic IDs**: Re-indexing same content won't create duplicates
            
            **Expected Performance:**
            - Small docs (5 PDFs): ~30 seconds
            - Medium docs (10-20 PDFs): 1-3 minutes  
            - Large docs (50+ PDFs): 5-15 minutes
            
            **⚠️ Storage Management:**
            - Each indexing run adds vectors to Pinecone
            - Use "📊 Check Index Stats" to monitor vector count
            - Use "🔄 Smart Re-index" to avoid duplicates
            - Use "🗑️ Clear Index" if you have too many vectors
            
            **If still slow, check:**
            - Internet connection to Pinecone (AWS us-east-1)
            - Available system memory (need ~2GB free)
            - Document complexity (lots of tables/images take longer)
            - Close other heavy applications
            """)
        
        # Quick diagnostics
        with st.expander("🔧 Quick Diagnostics"):
            if st.button("Check System Resources"):
                try:
                    from src.performance_monitor import PerformanceMonitor
                    monitor = PerformanceMonitor()
                    system_info = monitor.get_system_info()
                    
                    if not system_info.get('psutil_available', False):
                        st.warning("⚠️ Performance monitoring limited - psutil not available")
                        if 'message' in system_info:
                            st.info(system_info['message'])
                        if 'error' in system_info:
                            st.error(f"Error: {system_info['error']}")
                    else:
                        # Color code based on usage
                        cpu_color = "🔴" if system_info['cpu_percent'] > 80 else "🟡" if system_info['cpu_percent'] > 60 else "🟢"
                        mem_color = "🔴" if system_info['memory_percent'] > 80 else "🟡" if system_info['memory_percent'] > 60 else "🟢"
                        
                        st.write(f"{cpu_color} **CPU Usage:** {system_info['cpu_percent']:.1f}%")
                        st.write(f"{mem_color} **Memory Usage:** {system_info['memory_percent']:.1f}%")
                        st.write(f"💾 **Available Memory:** {system_info['available_memory_gb']:.1f} GB")
                        
                        if system_info['memory_percent'] > 80:
                            st.warning("⚠️ High memory usage! Close other applications before indexing.")
                        if system_info['cpu_percent'] > 80:
                            st.warning("⚠️ High CPU usage! Consider waiting or reducing batch size.")
                        
                except Exception as e:
                    st.error(f"Could not check system resources: {e}")
                    st.info("💡 Try installing psutil for better performance monitoring: `pip install psutil`")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Query Processing")
        
        # Sample queries
        st.markdown("### Sample Queries")
        sample_queries = [
            "46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
            "30F, heart surgery, Mumbai, 1 year policy",
            "25M, eye surgery, Delhi, 6 months policy",
            "Is dental treatment covered for a 35-year-old in Bangalore?",
            "What is the waiting period for maternity benefits?"
        ]
        
        selected_sample = st.selectbox("Choose a sample query:", 
                                     [""] + sample_queries)
        
        # Query input
        query = st.text_area("Enter your query:", 
                           value=selected_sample,
                           height=100,
                           placeholder="e.g., 46M, knee surgery, Pune, 3-month policy")
        
        # Process query button
        if st.button("🚀 Process Query", type="primary"):
            if not query.strip():
                st.error("Please enter a query")
            elif not pinecone_api_key or not gemini_api_key:
                st.error("Please provide both Pinecone and Gemini API keys")
            elif not st.session_state.documents_indexed:
                st.error("Please index documents first")
            else:
                process_query(query, pinecone_api_key, gemini_api_key, index_name)
    
    with col2:
        st.header("📊 System Overview")
        
        # Document statistics
        if st.session_state.documents_parsed and 'parsed_docs' in st.session_state:
            docs = st.session_state.parsed_docs
            st.metric("Total Documents", len(docs))
            
            total_paragraphs = sum(len(doc.get('parsed_output', {}).get('paragraphs', [])) for doc in docs)
            total_tables = sum(len(doc.get('parsed_output', {}).get('tables', [])) for doc in docs)
            
            st.metric("Total Paragraphs", total_paragraphs)
            st.metric("Total Tables", total_tables)
            
            # Document list
            st.markdown("### Documents")
            for doc in docs:
                with st.expander(doc['document_name']):
                    parsed = doc.get('parsed_output', {})
                    if 'error' in parsed:
                        st.error(f"Error: {parsed['error']}")
                    else:
                        st.write(f"Paragraphs: {len(parsed.get('paragraphs', []))}")
                        st.write(f"Tables: {len(parsed.get('tables', []))}")
                        st.write(f"Method: {parsed.get('method', 'Unknown')}")

def parse_documents():
    """Parse all PDF documents from docs/ folder."""
    with st.spinner("Parsing documents..."):
        try:
            # Find PDF files
            docs_dir = 'docs'
            if not os.path.exists(docs_dir):
                st.error(f"Directory '{docs_dir}' not found!")
                return
            
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                st.error(f"No PDF files found in '{docs_dir}' directory!")
                return
            
            # Parse documents
            pdf_paths = [os.path.join(docs_dir, f) for f in pdf_files]
            parsed_docs = load_and_parse_documents(pdf_paths)
            
            # Store in session state
            st.session_state.parsed_docs = parsed_docs
            st.session_state.documents_parsed = True
            
            st.success(f"Successfully parsed {len(parsed_docs)} documents!")
            
        except Exception as e:
            st.error(f"Error parsing documents: {str(e)}")

def smart_reindex(pinecone_api_key: str, index_name: str):
    """Smart re-indexing: clear index first, then index documents to avoid duplicates."""
    if not st.session_state.documents_parsed:
        st.error("Please parse documents first!")
        return
    
    # Get current index stats
    stats = get_index_stats(pinecone_api_key, index_name)
    
    if stats['exists'] and stats['total_vector_count'] > 0:
        st.info(f"🔄 **Smart Re-indexing:** Found {stats['total_vector_count']:,} existing vectors. Clearing index first...")
        
        # Clear the index
        with st.spinner("Clearing existing vectors..."):
            try:
                cleared_count = clear_pinecone_index(pinecone_api_key, index_name)
                st.success(f"✅ Cleared {cleared_count:,} vectors")
            except Exception as e:
                st.error(f"Failed to clear index: {str(e)}")
                return
    
    # Now proceed with normal indexing
    st.info("🚀 **Proceeding with fresh indexing...**")
    index_documents(pinecone_api_key, index_name)

def index_documents(pinecone_api_key: str, index_name: str):
    """Optimized indexing of parsed documents in Pinecone with progress tracking."""
    if not st.session_state.documents_parsed:
        st.error("Please parse documents first!")
        return
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Create progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(message: str, progress: float):
        """Callback function to update progress."""
        progress_bar.progress(int(progress))
        status_text.text(message)
        monitor.update_memory_peak("indexing")
    
    try:
        with monitor.monitor_operation("indexing"):
            # Prepare documents for chunking
            docs_for_chunking = []
            for doc in st.session_state.parsed_docs:
                parsed = doc.get('parsed_output', {})
                if 'error' not in parsed:
                    # Combine paragraphs and tables
                    paragraphs = parsed.get('paragraphs', [])
                    tables = parsed.get('tables', [])
                    
                    # Create full text with better formatting
                    full_text = '\n\n'.join(paragraphs)
                    if tables:
                        full_text += '\n\n=== TABLES ===\n\n' + '\n\n'.join(tables)
                    
                    docs_for_chunking.append({
                        'document_name': doc['document_name'],
                        'parsed_text': full_text
                    })
            
            update_progress("Chunking documents...", 0)
            
            # Chunk documents with progress callback
            chunks = chunk_documents(docs_for_chunking, progress_callback=update_progress)
            
            # Show time estimate before starting indexing
            avg_chunk_length = sum(len(chunk['text']) for chunk in chunks) // len(chunks) if chunks else 0
            time_estimate = estimate_indexing_time(len(chunks), avg_chunk_length)
            
            update_progress(f"Indexing {len(chunks)} chunks (Est. {time_estimate['total_time_minutes']:.1f} min)...", 30)
            
            # Index in Pinecone with progress callback
            indexed_count = index_chunks_in_pinecone(
                chunks, 
                pinecone_api_key, 
                "us-east-1", 
                index_name,
                progress_callback=update_progress
            )
        
        # Store results in session state
        st.session_state.documents_indexed = True
        st.session_state.total_chunks = len(chunks)
        st.session_state.indexed_count = indexed_count
        st.session_state.last_indexing_metrics = monitor.metrics.get("indexing", {})
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success message with details
        metrics_display = monitor.format_metrics("indexing")
        st.success(f"""
        ✅ **Indexing Complete!**
        - **Documents processed:** {len(docs_for_chunking)}
        - **Chunks created:** {len(chunks)}
        - **Vectors indexed:** {indexed_count}
        - **Index name:** {index_name}
        
        {metrics_display}
        """)
        
        # Show performance recommendations
        recommendations = monitor.get_recommendations("indexing")
        if recommendations:
            with st.expander("🚀 Performance Recommendations"):
                for rec in recommendations:
                    st.info(rec)
        
        # Show chunking and indexing statistics
        with st.expander("📊 Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Chunking Stats:**")
                chunk_stats = {}
                for chunk in chunks:
                    doc_name = chunk['document_name']
                    if doc_name not in chunk_stats:
                        chunk_stats[doc_name] = {'count': 0, 'avg_length': 0, 'total_length': 0}
                    chunk_stats[doc_name]['count'] += 1
                    chunk_stats[doc_name]['total_length'] += len(chunk['text'])
                
                for doc_name, stats in chunk_stats.items():
                    stats['avg_length'] = stats['total_length'] // stats['count']
                    st.write(f"**{doc_name}:** {stats['count']} chunks (avg {stats['avg_length']} chars)")
            
            with col2:
                st.markdown("**System Info:**")
                system_info = monitor.get_system_info()
                if system_info.get('psutil_available', False):
                    st.write(f"CPU Usage: {system_info['cpu_percent']:.1f}%")
                    st.write(f"Memory Usage: {system_info['memory_percent']:.1f}%")
                    st.write(f"Available Memory: {system_info['available_memory_gb']:.1f} GB")
                else:
                    st.write("System monitoring unavailable")
                    st.write("(psutil not installed)")
            
    except Exception as e:
        # Clean up progress indicators on error
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ **Indexing failed:** {str(e)}")
        
        # Show more detailed error information
        with st.expander("🔍 Error Details"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())
            
            # Show system info that might help debug
            try:
                system_info = monitor.get_system_info()
                st.json(system_info)
            except:
                pass

def process_query(query: str, pinecone_api_key: str, gemini_api_key: str, index_name: str):
    """Process a natural language query."""
    with st.spinner("Processing query..."):
        try:
            # Initialize query processor
            processor = QueryProcessor(pinecone_api_key, gemini_api_key, index_name)
            
            # Process query
            result = processor.process_query(query)
            
            # Display results
            display_results(result)
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

def display_results(result: Dict[str, Any]):
    """Display query processing results."""
    
    st.markdown("## 📋 Query Results")
    
    # Query and status
    st.markdown(f"**Query:** {result['query']}")
    st.markdown(f"**Status:** {result['status']}")
    
    # Show API status if available
    if 'api_status' in result:
        api_status = result['api_status']
        if not api_status['gemini_available']:
            with st.expander("⚠️ LLM Status Information", expanded=True):
                if api_status['quota_exceeded']:
                    st.error("🚨 **Gemini API Quota Exceeded**")
                    st.info("The system is using enhanced fallback methods for analysis.")
                else:
                    st.warning("⚠️ **LLM Not Available**")
                    st.info(f"Reason: {api_status.get('fallback_reason', 'Unknown')}")
                
                st.markdown("**Recommendations:**")
                for rec in api_status.get('recommendations', []):
                    st.write(f"- {rec}")
        else:
            st.success(f"✅ Using {api_status['model_name']} for optimal analysis")
    
    if result['status'] == 'error':
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Decision", "🔍 Entities", "📄 Retrieved Context", "📊 Details"])
    
    with tab1:
        st.header("Decision & Evaluation")
        evaluation = result.get('evaluation', {})
        
        # Show note about fallback methods if present
        if 'note' in evaluation:
            st.info(evaluation['note'])
        
        # Decision badge
        decision = evaluation.get('decision', 'unknown')
        confidence = evaluation.get('confidence', 0.0)
        
        if decision == 'approved':
            st.success(f"✅ **APPROVED** (Confidence: {confidence:.1%})")
        elif decision == 'rejected':
            st.error(f"❌ **REJECTED** (Confidence: {confidence:.1%})")
        elif decision == 'needs_review':
            st.warning(f"⚠️ **NEEDS REVIEW** (Confidence: {confidence:.1%})")
        else:
            st.info(f"ℹ️ **{decision.upper()}** (Confidence: {confidence:.1%})")
        
        # Amount
        amount = evaluation.get('amount')
        if amount:
            st.metric("Amount", f"₹{amount:,}")
        
        # Justification
        st.markdown("### Justification")
        st.write(evaluation.get('justification', 'No justification provided'))
        
        # Reasoning
        st.markdown("### Reasoning")
        st.write(evaluation.get('reasoning', 'No reasoning provided'))
        
        # Relevant clauses
        clauses = evaluation.get('relevant_clauses', [])
        if clauses:
            st.markdown("### 📜 Relevant Clauses")
            for i, clause in enumerate(clauses, 1):
                st.write(f"{i}. {clause}")
    
    with tab2:
        st.header("Extracted Entities")
        entities = result.get('entities', {})
        
        if entities:
            # Display as metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if entities.get('age'):
                    st.metric("Age", f"{entities['age']} years")
                if entities.get('gender'):
                    st.metric("Gender", entities['gender'])
            
            with col2:
                if entities.get('procedure'):
                    st.metric("Procedure", entities['procedure'])
                if entities.get('location'):
                    st.metric("Location", entities['location'])
            
            with col3:
                if entities.get('policy_duration'):
                    st.metric("Policy Duration", entities['policy_duration'])
                if entities.get('amount'):
                    st.metric("Amount", f"₹{entities['amount']:,}")
            
            # Display all entities as JSON
            st.markdown("### Raw Entities")
            st.json(entities)
        else:
            st.info("No entities extracted")
    
    with tab3:
        st.header("Retrieved Context")
        search_results = result.get('search_results', [])
        
        if search_results:
            for i, res in enumerate(search_results, 1):
                with st.expander(f"Result {i} - {res.get('document_name', 'Unknown')} (Score: {res.get('score', 0):.3f})"):
                    st.markdown(f"**Document:** {res.get('document_name', 'Unknown')}")
                    st.markdown(f"**Page:** {res.get('page_number', 'Unknown')}")
                    st.markdown(f"**Similarity Score:** {res.get('score', 0):.3f}")
                    st.markdown("**Content:**")
                    st.write(res.get('text', 'No content'))
        else:
            st.info("No search results found")
    
    with tab4:
        st.header("Technical Details")
        
        # Full result as JSON
        st.markdown("### Full Response")
        st.json(result)

def check_index_stats(pinecone_api_key: str, index_name: str):
    """Check and display Pinecone index statistics."""
    with st.spinner("Checking index statistics..."):
        try:
            stats = get_index_stats(pinecone_api_key, index_name)
            
            if not stats['exists']:
                st.warning(f"Index '{index_name}' does not exist yet.")
                if 'error' in stats:
                    st.error(f"Error: {stats['error']}")
                return
            
            # Display stats in a nice format
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Vectors", f"{stats['total_vector_count']:,}")
            
            with col2:
                st.metric("Dimensions", stats['dimension'])
            
            with col3:
                st.metric("Index Fullness", f"{stats['index_fullness']:.1%}")
            
            # Storage usage estimation
            vector_count = stats['total_vector_count']
            estimated_size_mb = (vector_count * stats['dimension'] * 4) / (1024 * 1024)  # 4 bytes per float32
            
            st.info(f"""
            📊 **Index Analysis:**
            - **Storage Usage**: ~{estimated_size_mb:.1f} MB
            - **Vector Count**: {vector_count:,} vectors
            - **Status**: {'Healthy' if stats['index_fullness'] < 0.9 else 'Nearly Full'}
            """)
            
            # Warning for high vector count
            if vector_count > 1000:
                st.warning(f"⚠️ **High vector count detected!** You have {vector_count:,} vectors. This may indicate duplicate indexing.")
                
                # Calculate how many documents this represents
                if st.session_state.get('total_chunks'):
                    expected_chunks = st.session_state.total_chunks
                    duplicate_ratio = vector_count / expected_chunks if expected_chunks > 0 else 0
                    if duplicate_ratio > 1.5:
                        st.error(f"🔴 **Duplicate indexing detected!** You have ~{duplicate_ratio:.1f}x more vectors than expected. Consider clearing the index.")
            
        except Exception as e:
            st.error(f"Failed to check index stats: {str(e)}")

def clear_index_with_confirmation(pinecone_api_key: str, index_name: str):
    """Clear Pinecone index with user confirmation."""
    
    # First check if index exists and get stats
    stats = get_index_stats(pinecone_api_key, index_name)
    
    if not stats['exists']:
        st.info(f"Index '{index_name}' does not exist or is already empty.")
        return
    
    vector_count = stats['total_vector_count']
    
    if vector_count == 0:
        st.info("Index is already empty.")
        return
    
    # Show warning and require confirmation
    st.warning(f"""
    ⚠️ **WARNING: This will permanently delete ALL {vector_count:,} vectors from your Pinecone index!**
    
    This action cannot be undone. You will need to re-index your documents after clearing.
    """)
    
    # Double confirmation
    confirm_text = st.text_input(
        f"Type 'DELETE {vector_count}' to confirm:",
        placeholder=f"DELETE {vector_count}"
    )
    
    if st.button("🗑️ CONFIRM DELETE", type="secondary"):
        if confirm_text == f'DELETE {vector_count}':
            with st.spinner(f"Clearing {vector_count:,} vectors from index..."):
                try:
                    cleared_count = clear_pinecone_index(pinecone_api_key, index_name)
                    
                    # Reset session state
                    st.session_state.documents_indexed = False
                    if 'total_chunks' in st.session_state:
                        del st.session_state.total_chunks
                    if 'indexed_count' in st.session_state:
                        del st.session_state.indexed_count
                    
                    st.success(f"""
                    ✅ **Index Cleared Successfully!**
                    - **Deleted**: {cleared_count:,} vectors
                    - **Status**: Index is now empty
                    - **Next Step**: You can now re-index your documents
                    """)
                    
                    # Show storage savings
                    estimated_size_mb = (cleared_count * 384 * 4) / (1024 * 1024)  # 384 dims, 4 bytes per float
                    st.info(f"💾 **Storage freed**: ~{estimated_size_mb:.1f} MB")
                    
                except Exception as e:
                    st.error(f"Failed to clear index: {str(e)}")
        else:
            st.error("Please type the exact confirmation text to proceed.")

if __name__ == "__main__":
    main()
