import streamlit as st
import os
from src.parse_documents import load_and_parse_documents
from src.chunk_documents import chunk_documents_old_format
from src.embed_and_index import index_chunks_in_pinecone
from src.query_processor import QueryProcessor

def parse_documents():
    """Parse all PDF documents from docs/ folder."""
    try:
        # Find PDF files
        docs_dir = 'docs'
        if not os.path.exists(docs_dir):
            return {"success": False, "error": "Directory 'docs' not found!"}
        
        pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            return {"success": False, "error": "No PDF files found in 'docs' directory!"}
        
        # Parse documents
        pdf_paths = [os.path.join(docs_dir, f) for f in pdf_files]
        parsed_docs = load_and_parse_documents(pdf_paths)
        
        if not parsed_docs:
            return {"success": False, "error": "Failed to parse documents!"}
        
        # Process into chunks
        chunks = chunk_documents_old_format(parsed_docs)
        
        return {
            "success": True,
            "document_count": len(parsed_docs),
            "total_chunks": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def embed_and_index():
    """Embed and index the chunks."""
    try:
        if 'parsed_chunks' not in st.session_state:
            return {"success": False, "error": "No chunks found. Please parse documents first."}
        
        chunks = st.session_state.parsed_chunks
        
        # Get API keys from secrets or environment
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            return {"success": False, "error": "Pinecone API key not found!"}
        
        # Embed and index (pinecone_env is ignored in the function)
        result = index_chunks_in_pinecone(chunks, pinecone_api_key, "us-east-1")
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def semantic_search_with_reranking(query: str):
    """Wrapper function for complete query processing with entities and reasoning."""
    try:
        # Get API keys from secrets or environment
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not pinecone_api_key:
            raise Exception("Pinecone API key not found!")
        
        if not gemini_api_key:
            raise Exception("Gemini API key not found!")
        
        # Create QueryProcessor instance
        processor = QueryProcessor(pinecone_api_key, gemini_api_key)
        
        # Use the complete query processing pipeline
        result = processor.process_query(query)
        
        return result
        
    except Exception as e:
        raise Exception(f"Search error: {str(e)}")

def main():
    st.set_page_config(
        page_title="Policy Document Q&A System",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Policy Document Q&A System")
    st.markdown("---")
    
    # Sidebar for parsing and indexing controls
    with st.sidebar:
        st.header("📋 Document Processing")
        
        # Check if documents are parsed
        if 'documents_parsed' not in st.session_state:
            st.session_state.documents_parsed = False
        
        if 'documents_indexed' not in st.session_state:
            st.session_state.documents_indexed = False
        
        # Parse Documents Section
        if st.button("📄 Parse Documents", use_container_width=True):
            with st.spinner("Parsing documents..."):
                result = parse_documents()
                if result["success"]:
                    st.session_state.documents_parsed = True
                    st.session_state.total_chunks = result["total_chunks"]
                    st.session_state.parsed_chunks = result["chunks"]
                    st.success(f"✅ Successfully parsed {result['document_count']} documents into {result['total_chunks']} chunks!")
                else:
                    st.error(f"❌ Error parsing documents: {result['error']}")
        
        # Index Documents Section
        if st.button("🚀 Index Documents", use_container_width=True):
            if not st.session_state.documents_parsed:
                st.warning("⚠️ Please parse documents first!")
            else:
                with st.spinner("Embedding and indexing documents..."):
                    result = embed_and_index()
                    if result["success"]:
                        st.session_state.documents_indexed = True
                        st.session_state.indexed_count = result.get("indexed_count", 0)
                        st.success(f"✅ Successfully indexed {st.session_state.indexed_count} document chunks!")
                    else:
                        st.error(f"❌ Error indexing documents: {result['error']}")
        
        # Status indicators
        st.markdown("---")
        st.subheader("📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'total_chunks' in st.session_state:
                st.metric("Total Chunks", st.session_state.total_chunks)
        with col2:
            if 'indexed_count' in st.session_state:
                st.metric("Vectors Indexed", st.session_state.indexed_count)
    
    # Main query interface
    if st.session_state.documents_indexed:
        st.header("🔍 Ask a Question")
        
        # Sample queries section
        st.markdown("### 💡 Sample Queries")
        sample_queries = [
            "46M, knee surgery, Pune, 3-month policy",
            "30F, heart surgery, Mumbai, 6-month policy", 
            "25M, eye surgery, Delhi, 1-year policy",
            "40F, dental treatment, Bangalore, 2-month policy",
            "35F, maternity benefits, Chennai, 8-month policy"
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_query = st.selectbox("Choose a sample query:", [""] + sample_queries)
        with col2:
            st.markdown("**Expected Response:**")
            st.info("Yes/No + brief justification")
        
        query = st.text_input(
            "Enter your question:",
            value=selected_query,
            placeholder="e.g., 46M, knee surgery, Pune, 3-month policy",
            key="query_input"
        )
        
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                with st.spinner("Searching for answers..."):
                    try:
                        # Use the complete query processing pipeline
                        result = semantic_search_with_reranking(query)
                        
                        if result["status"] == "success":
                            # Display the main answer/decision
                            evaluation = result["evaluation"]
                            
                            # Show concise answer first
                            if evaluation.get("decision"):
                                decision_color = "success" if evaluation["decision"].lower() in ["yes", "covered", "eligible"] else "info"
                                st.markdown(f"""
                                ### 🎯 **Answer:** 
                                """)
                                if decision_color == "success":
                                    st.success(f"✅ {evaluation['justification']}")
                                else:
                                    st.info(f"ℹ️ {evaluation['justification']}")
                            
                            # Show extracted entities
                            if result["entities"]:
                                with st.expander("🔍 Extracted Information", expanded=False):
                                    entities = result["entities"]
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        if entities.get("age"):
                                            st.metric("Age", f"{entities['age']} years")
                                        if entities.get("gender"):
                                            st.metric("Gender", entities["gender"])
                                    
                                    with col2:
                                        if entities.get("procedure"):
                                            st.write(f"**Procedure:** {entities['procedure']}")
                                        if entities.get("location"):
                                            st.write(f"**Location:** {entities['location']}")
                                    
                                    with col3:
                                        if entities.get("policy_duration"):
                                            st.write(f"**Policy Duration:** {entities['policy_duration']}")
                                        if entities.get("amount"):
                                            st.write(f"**Amount:** ₹{entities['amount']:,}")
                            
                            # Show detailed analysis
                            with st.expander("📋 Detailed Analysis", expanded=False):
                                if evaluation.get("reasoning"):
                                    st.write("**Reasoning:**")
                                    st.write(evaluation["reasoning"])
                                
                                if evaluation.get("relevant_clauses"):
                                    st.write("**Relevant Policy Clauses:**")
                                    for clause in evaluation["relevant_clauses"]:
                                        st.write(f"• {clause}")
                                
                                if evaluation.get("confidence"):
                                    st.metric("Confidence Score", f"{evaluation['confidence']:.1%}")
                            
                            # Show search results
                            search_results = result["search_results"]
                            if search_results:
                                with st.expander(f"📄 Supporting Documents ({len(search_results)} sources)", expanded=False):
                                    for i, search_result in enumerate(search_results, 1):
                                        score = search_result.get('cross_score', search_result.get('vector_score', 0))
                                        with st.container():
                                            st.markdown(f"**Source {i}** - Relevance: {score:.3f}")
                                            st.write(search_result['text'])
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.caption(f"📄 Document: {search_result.get('document_name', 'Unknown')}")
                                            with col2:
                                                if search_result.get('context_expanded'):
                                                    st.caption("📖 Context: ✅ Expanded")
                                            st.markdown("---")
                        
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during search: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question!")
    else:
        st.info("📋 Please parse and index documents first using the sidebar controls.")

if __name__ == "__main__":
    main()
