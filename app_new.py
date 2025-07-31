import streamlit as st
import os
import json
from src.pipeline import streamlit_single_click_pipeline_sync, query_documents_sync

def query_documents():
    """Query the processed documents using RAG pipeline."""
    try:
        # Get API keys from secrets or environment
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not pinecone_api_key:
            st.error("❌ Pinecone API key not found! Please set PINECONE_API_KEY in secrets or environment.")
            return
            
        if not gemini_api_key:
            st.warning("⚠️ Gemini API key not found! Using fallback mode with limited accuracy.")
            gemini_api_key = None  # Use None for fallback mode
        
        # Show API status
        if gemini_api_key:
            st.success("✅ LLM Ready: Advanced analysis available")
        else:
            st.warning("⚠️ LLM Unavailable: Using fallback mode with reduced accuracy")
        
        # Sample queries
        with st.expander("💡 Sample Queries", expanded=False):
            st.markdown("""
            **Coverage Questions:**
            - "What is covered under accidental death benefit?"
            - "Are pre-existing conditions covered?"
            - "What is the waiting period for maturity benefits?"
            
            **Claim Questions:**
            - "How do I file a claim for medical expenses?"
            - "What documents are needed for death claim?"
            - "What is the claim settlement process?"
            
            **Policy Questions:**
            - "What is the premium payment term?"
            - "Can I surrender my policy early?"
            - "What are the tax benefits available?"
            """)
        
        # Main query input
        user_query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is covered under accidental death benefit?",
            height=100
        )
        
        if st.button("🔍 Search Documents", type="primary", disabled=not user_query.strip()):
            if user_query.strip():
                with st.spinner("🔍 Searching documents and analyzing..."):
                    # Process query using pipeline function
                    result = query_documents_sync(
                        query=user_query.strip(),
                        pinecone_api_key=pinecone_api_key,
                        gemini_api_key=gemini_api_key,
                        index_name="policy-index"
                    )
                    
                    if result.get("success"):
                        # Display results
                        st.markdown("### 📊 Search Results")
                        
                        # Show evaluation
                        evaluation = result.get("evaluation", {})
                        decision = evaluation.get("decision", "unknown")
                        confidence = evaluation.get("confidence", 0.0)
                        
                        # Color-coded decision
                        if decision == "covered":
                            st.success(f"✅ **COVERED** (Confidence: {confidence:.1%})")
                        elif decision == "not_covered":
                            st.error(f"❌ **NOT COVERED** (Confidence: {confidence:.1%})")
                        elif decision == "partial":
                            st.warning(f"⚠️ **PARTIALLY COVERED** (Confidence: {confidence:.1%})")
                        else:
                            st.info(f"🤔 **UNCLEAR** (Confidence: {confidence:.1%})")
                        
                        # Show justification
                        justification = evaluation.get("justification", "No explanation available")
                        st.markdown(f"**Explanation:** {justification}")
                        
                        # Show relevant clauses
                        relevant_clauses = evaluation.get("relevant_clauses", [])
                        if relevant_clauses:
                            st.markdown("### 📄 Relevant Policy Clauses")
                            for i, clause in enumerate(relevant_clauses, 1):
                                with st.expander(f"Clause {i}: {clause.get('section', 'Unknown Section')}"):
                                    st.markdown(f"**Content:** {clause.get('content', 'No content available')}")
                                    if clause.get('page'):
                                        st.markdown(f"**Source:** Page {clause['page']}")
                        
                        # Show search results
                        search_results = result.get("search_results", [])
                        if search_results:
                            st.markdown("### 🔍 Source Documents")
                            for i, chunk in enumerate(search_results, 1):
                                score = chunk.get('score', 0.0)
                                source = chunk.get('source_document', 'Unknown')
                                content = chunk.get('text', 'No content available')
                                
                                with st.expander(f"Result {i}: {source} (Score: {score:.3f})"):
                                    st.markdown(content[:1000] + "..." if len(content) > 1000 else content)
                        
                        # Show extracted entities
                        entities = result.get("entities", {})
                        if entities and any(entities.values()):
                            with st.expander("🏷️ Extracted Information", expanded=False):
                                st.json(entities)
                        
                        # Show technical details
                        with st.expander("🔧 Technical Details", expanded=False):
                            api_status = result.get("api_status", {})
                            tech_info = {
                                "Search Method": evaluation.get("search_method", "unknown"),
                                "Chunks Used": evaluation.get("final_chunks_used", 0),
                                "LLM Available": api_status.get('gemini_available', False),
                                "Model": api_status.get('model_name', 'N/A')
                            }
                            st.json(tech_info)
                    
                    else:
                        st.error("❌ Query processing failed!")
                        st.error(f"Error: {result.get('error', 'Unknown error')}")
                        
    except Exception as e:
        st.error(f"❌ Query system error: {str(e)}")
        st.exception(e)


def single_click_pipeline():
    """Single-click pipeline for complete document processing."""
    try:
        # Get API keys from secrets or environment
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        
        if not pinecone_api_key:
            return {"success": False, "error": "Pinecone API key not found!"}
        
        # Run the complete pipeline
        result = streamlit_single_click_pipeline_sync(
            pinecone_api_key=pinecone_api_key,
            force_reprocess=False
        )
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    st.set_page_config(
        page_title="Insurance Policy RAG System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Insurance Policy RAG System")
    st.markdown("**AI-Powered Insurance Policy Analysis** • Ask questions and get instant answers with source citations")
    st.markdown("---")
    
    # Sidebar for document processing
    with st.sidebar:
        st.header("📄 Document Processing")
        st.markdown("Process PDF documents to build the knowledge base")
        
        # Show current document status
        docs_dir = "docs"
        if os.path.exists(docs_dir):
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            if pdf_files:
                st.success(f"📁 {len(pdf_files)} PDF files found")
                with st.expander("📋 Files in docs/", expanded=False):
                    for pdf in pdf_files:
                        st.markdown(f"• {pdf}")
            else:
                st.warning("📂 No PDF files found")
                st.info("Place PDF files in the `docs/` folder")
        else:
            st.error("📂 docs/ directory not found")
        
        st.markdown("---")
        
        # Processing button
        if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
            with st.spinner("🔄 Processing documents..."):
                result = single_click_pipeline()
                
                if result.get("success", False):
                    # Safely get statistics and timing with defaults
                    stats = result.get("statistics", {})
                    timing = result.get("timing", {})
                    
                    st.success("✅ Processing Complete!")
                    
                    # Display compact statistics
                    st.metric("Files Processed", stats.get("processed_files", 0))
                    st.metric("Chunks Created", stats.get("total_chunks", 0)) 
                    st.metric("Vectors Indexed", stats.get("indexed_vectors", 0))
                    
                    # Show timing info
                    total_time = timing.get('total_time', 'N/A')
                    st.info(f"⏱️ Total: {total_time}")
                    
                    # Show skipped files info
                    skipped_files = stats.get("skipped_files", 0)
                    if skipped_files > 0:
                        st.info(f"📋 Skipped {skipped_files} processed files")
                        
                else:
                    st.error("❌ Processing Failed")
                    error_msg = result.get('error', 'Unknown error occurred')
                    step = result.get('step', 'unknown')
                    st.error(f"Error in {step}: {error_msg}")
                    
                    # Show additional debug information if available
                    if 'user_message' in result:
                        st.info(result['user_message'])
        
        # Force reprocess option
        st.markdown("---")
        if st.button("🔄 Force Reprocess All", use_container_width=True):
            with st.spinner("🔄 Reprocessing all documents..."):
                # Get API keys from secrets or environment
                pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
                
                if pinecone_api_key:
                    result = streamlit_single_click_pipeline_sync(
                        pinecone_api_key=pinecone_api_key,
                        force_reprocess=True
                    )
                    
                    if result.get("success", False):
                        st.success("✅ Reprocessing Complete!")
                        stats = result.get("statistics", {})
                        st.metric("Total Processed", stats.get("processed_files", 0))
                    else:
                        st.error("❌ Reprocessing Failed")
                        st.error(result.get('error', 'Unknown error'))
                else:
                    st.error("❌ API key not found")
    
    # Main content area for queries
    st.markdown("### 🔍 Ask Questions About Your Insurance Policies")
    
    # Query interface (now taking full width)
    query_documents()

if __name__ == "__main__":
    main()
