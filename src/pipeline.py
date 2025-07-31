"""
Single-click pipeline for complete document processing.
Combines parsing, chunking, embedding, and indexing into one streamlined operation.
"""

import os
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .parse_documents import load_and_parse_documents
from .chunk_documents_optimized import chunk_documents_optimized
from .embed_and_index import index_chunks_in_pinecone
from .document_registry import DocumentRegistry


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    chunk_size: int = 800
    chunk_overlap: int = 150
    use_semantic: bool = True
    save_parsed_text: bool = False
    index_name: str = "policy-index"
    pinecone_env: str = "us-east-1"  # Kept for compatibility
    output_dir: str = "results"


class DocumentPipeline:
    """Complete document processing pipeline."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.registry = DocumentRegistry()
    
    async def process_all_documents(self, 
                            docs_dir: str = "docs", 
                            pinecone_api_key: Optional[str] = None,
                            force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Complete single-click pipeline for all documents.
        
        Args:
            docs_dir: Directory containing PDF documents
            pinecone_api_key: Pinecone API key for indexing
            force_reprocess: Whether to reprocess already processed documents
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate inputs
            if not os.path.exists(docs_dir):
                return {
                    "success": False,
                    "error": f"Directory '{docs_dir}' not found!",
                    "step": "validation"
                }
            
            if not pinecone_api_key:
                return {
                    "success": False,
                    "error": "Pinecone API key is required!",
                    "step": "validation"
                }
            
            # Step 2: Find PDF files
            pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                return {
                    "success": False,
                    "error": f"No PDF files found in '{docs_dir}' directory!",
                    "step": "file_discovery"
                }
            
            # Step 3: Filter files based on registry (unless force reprocess)
            files_to_process = []
            skipped_files = []
            
            if force_reprocess:
                files_to_process = [os.path.join(docs_dir, f) for f in pdf_files]
            else:
                # Check registry status
                doc_status = self.registry.get_document_status(docs_dir)
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(docs_dir, pdf_file)
                    status = doc_status.get(pdf_file, 'new')
                    
                    if status in ['new', 'changed']:
                        files_to_process.append(pdf_path)
                    else:
                        skipped_files.append(pdf_file)
            
            if not files_to_process and not force_reprocess:
                return {
                    "success": True,
                    "message": "All documents already processed. Use force_reprocess=True to reprocess.",
                    "skipped_files": skipped_files,
                    "step": "registry_check",
                    "processing_time": time.time() - start_time
                }
            
            # Step 4: Parse documents (async)
            parsing_start = time.time()
            parsed_docs = await self._parse_documents_async(files_to_process)
            
            if not parsed_docs:
                return {
                    "success": False,
                    "error": "Failed to parse any documents!",
                    "step": "parsing"
                }
            
            parsing_time = time.time() - parsing_start
            
            # Step 5: Chunk documents using the existing chunk_documents function (async)
            chunking_start = time.time()
            
            all_chunks = await self._chunk_documents_async(parsed_docs)
            
            chunking_time = time.time() - chunking_start
            
            if not all_chunks:
                return {
                    "success": False,
                    "error": "Failed to create any chunks!",
                    "step": "chunking"
                }
            
            # Step 6: Generate embeddings and index (async)
            indexing_start = time.time()
            
            indexing_result = await self._index_documents_async(all_chunks, pinecone_api_key)
            
            indexing_time = time.time() - indexing_start
            
            if not indexing_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Indexing failed: {indexing_result.get('error', 'Unknown error')}",
                    "step": "indexing"
                }
            
            # Step 7: Update registry for processed files (async)
            await self._update_registry_async(files_to_process, all_chunks)
            
            # Calculate statistics
            total_time = time.time() - start_time
            total_chars = sum(len(doc.get('content', '')) for doc in parsed_docs)
            
            return {
                "success": True,
                "statistics": {
                    "total_files": len(pdf_files),
                    "processed_files": len(files_to_process),
                    "skipped_files": len(skipped_files),
                    "total_documents": len(parsed_docs),
                    "total_chunks": len(all_chunks),
                    "indexed_vectors": indexing_result.get("indexed_count", len(all_chunks)),
                    "total_characters": total_chars,
                    "processing_speed": f"{total_chars / total_time:.0f} chars/sec",
                },
                "timing": {
                    "parsing_time": f"{parsing_time:.2f}s",
                    "chunking_time": f"{chunking_time:.2f}s", 
                    "indexing_time": f"{indexing_time:.2f}s",
                    "total_time": f"{total_time:.2f}s"
                },
                "files": {
                    "processed": [os.path.basename(f) for f in files_to_process],
                    "skipped": skipped_files
                },
                "step": "completed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Pipeline error: {str(e)}",
                "step": "unknown",
                "processing_time": time.time() - start_time
            }

    async def _parse_documents_async(self, files_to_process: List[str]) -> List[Dict]:
        """Async wrapper for document parsing."""
        # Run parsing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_and_parse_documents, files_to_process)
    
    async def _chunk_documents_async(self, parsed_docs: List[Dict]) -> List[Dict]:
        """Async wrapper for document chunking."""
        # Transform parsed_docs format to match what chunk_documents expects
        transformed_docs = []
        
        for doc in parsed_docs:
            doc_name = doc.get('document_name', 'unknown')
            parsed_output = doc.get('parsed_output', {})
            
            # Extract content from parsed_output
            if isinstance(parsed_output, dict):
                # Check for different possible content fields
                content = (parsed_output.get('content', '') or 
                          parsed_output.get('text', '') or 
                          parsed_output.get('cleaned_text', ''))
                ordered_content = parsed_output.get('ordered_content', [])
            else:
                content = str(parsed_output) if parsed_output else ''
                ordered_content = []
            
            # Create the format expected by chunk_documents
            transformed_doc = {
                'document_name': doc_name,
                'content': content,
                'ordered_content': ordered_content
            }
            
            transformed_docs.append(transformed_doc)
        
        # Run chunking in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: chunk_documents_optimized(
                parsed_content=transformed_docs,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                save_parsed_text=self.config.save_parsed_text,
                output_dir=self.config.output_dir
            )
        )
    
    async def _index_documents_async(self, all_chunks: List[Dict], pinecone_api_key: str) -> Dict:
        """Async wrapper for document indexing."""
        # Run indexing in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: index_chunks_in_pinecone(
                chunks=all_chunks,
                pinecone_api_key=pinecone_api_key,
                pinecone_env=self.config.pinecone_env,
                index_name=self.config.index_name
            )
        )
    
    async def _update_registry_async(self, files_to_process: List[str], all_chunks: List[Dict]):
        """Async wrapper for registry updates."""
        loop = asyncio.get_event_loop()
        
        def update_registry():
            for file_path in files_to_process:
                filename = os.path.basename(file_path)
                # Count chunks for this specific document
                doc_chunks = [c for c in all_chunks if c.get('source_document', '').endswith(filename)]
                self.registry.mark_document_indexed(filename, file_path, len(doc_chunks))
        
        await loop.run_in_executor(None, update_registry)


# Convenience function for single-click operation
async def process_all_documents_pipeline(docs_dir: str = "docs",
                                 pinecone_api_key: Optional[str] = None,
                                 force_reprocess: bool = False,
                                 config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Single-click function to process all documents through the complete pipeline.
    
    Args:
        docs_dir: Directory containing PDF documents (default: "docs")
        pinecone_api_key: Pinecone API key for vector indexing
        force_reprocess: Whether to reprocess already processed documents
        config: Pipeline configuration (uses defaults if None)
        
    Returns:
        Dictionary with processing results and detailed statistics
        
    Example:
        >>> result = await process_all_documents_pipeline(
        ...     pinecone_api_key="your-key-here",
        ...     force_reprocess=False
        ... )
        >>> if result["success"]:
        ...     print(f"✅ Processed {result['statistics']['total_chunks']} chunks")
        ... else:
        ...     print(f"❌ Error: {result['error']}")
    """
    pipeline = DocumentPipeline(config)
    return await pipeline.process_all_documents(docs_dir, pinecone_api_key, force_reprocess)


# Alternative streamlined function for Streamlit
async def streamlit_single_click_pipeline(pinecone_api_key: Optional[str] = None,
                                  force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Streamlit-optimized single-click pipeline function.
    Uses sensible defaults and provides user-friendly output.
    
    Args:
        pinecone_api_key: Pinecone API key
        force_reprocess: Whether to reprocess all documents
        
    Returns:
        Dictionary with success status and user-friendly messages
    """
    # Use optimized config for Streamlit
    config = PipelineConfig(
        chunk_size=800,  # Optimal for retrieval
        chunk_overlap=150,
        use_semantic=True,
        save_parsed_text=False,
        index_name="policy-index",
        output_dir="results"
    )
    
    result = await process_all_documents_pipeline(
        docs_dir="docs",
        pinecone_api_key=pinecone_api_key,
        force_reprocess=force_reprocess,
        config=config
    )
    
    # Add user-friendly messages
    if result["success"]:
        stats = result["statistics"]
        timing = result["timing"]
        
        result["user_message"] = f"""
🎉 **Pipeline Complete!**

📊 **Processing Summary:**
- Processed {stats['processed_files']} PDF files
- Created {stats['total_chunks']} semantic chunks
- Indexed {stats['indexed_vectors']} vectors in Pinecone
- Processing speed: {stats['processing_speed']}

⏱️ **Timing:**
- Parsing: {timing['parsing_time']}
- Chunking: {timing['chunking_time']}
- Indexing: {timing['indexing_time']}
- **Total: {timing['total_time']}**

✅ Your document search system is ready!
        """.strip()
        
        if stats['skipped_files'] > 0:
            result["user_message"] += f"\n\n📋 Skipped {stats['skipped_files']} already processed files."
    
    else:
        result["user_message"] = f"❌ **Pipeline Failed**\n\nError in {result.get('step', 'unknown')} step:\n{result['error']}"
    
    return result


# Synchronous wrapper for Streamlit compatibility
def streamlit_single_click_pipeline_sync(pinecone_api_key: Optional[str] = None,
                                        force_reprocess: bool = False) -> Dict[str, Any]:
    """
    Synchronous wrapper for Streamlit compatibility.
    Runs the async pipeline in a new event loop.
    
    Args:
        pinecone_api_key: Pinecone API key
        force_reprocess: Whether to reprocess all documents
        
    Returns:
        Dictionary with success status and user-friendly messages
    """
    try:
        # Create new event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async pipeline
        result = loop.run_until_complete(
            streamlit_single_click_pipeline(pinecone_api_key, force_reprocess)
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Pipeline execution error: {str(e)}",
            "step": "async_wrapper",
            "user_message": f"❌ **Pipeline Failed**\n\nAsync execution error:\n{str(e)}"
        }
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except:
            pass
