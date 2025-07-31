import streamlit as st
import os
from src.pipeline import streamlit_single_click_pipeline_sync

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
        page_title="Document Processing Pipeline",
        page_icon="📄",
        layout="centered"
    )
    
    st.title("📄 Document Processing Pipeline")
    st.markdown("Complete processing: Parse → Chunk → Embed → Index")
    st.markdown("---")
    
    # Single-Click Pipeline Section
    st.markdown("### Optimized Single-Click Pipeline")
    st.markdown("CPU-only • Fast • No GPU required")
    
    if st.button("Process All Documents", use_container_width=True, type="primary"):
        with st.spinner("Running complete pipeline..."):
            result = single_click_pipeline()
            
            if result["success"]:
                stats = result["statistics"]
                timing = result["timing"]
                
                st.success("Pipeline Complete!")
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Processed", stats["processed_files"])
                with col2:
                    st.metric("Chunks Created", stats["total_chunks"])
                with col3:
                    st.metric("Vectors Indexed", stats["indexed_vectors"])
                
                # Show timing info
                st.info(f"Total Time: {timing['total_time']} | Speed: {stats['processing_speed']}")
                
                if stats["skipped_files"] > 0:
                    st.info(f"Skipped {stats['skipped_files']} already processed files.")
                    
            else:
                st.error("Pipeline Failed")
                st.error(f"Error in {result.get('step', 'unknown')} step: {result['error']}")

if __name__ == "__main__":
    main()
