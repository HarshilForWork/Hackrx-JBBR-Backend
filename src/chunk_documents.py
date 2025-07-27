"""
Module: chunk_documents.py
Functionality: Advanced semantic chunking using LangChain with MiniLM embeddings.
"""
from typing import List, Dict, Optional, Callable
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Cache for the semantic splitter to avoid reloading
_semantic_splitter_cache = None

def get_semantic_splitter():
    """Get cached LangChain semantic splitter with MiniLM embeddings."""
    global _semantic_splitter_cache
    if _semantic_splitter_cache is None:
        # Initialize HuggingFace embeddings with the same model we use for indexing
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create semantic chunker
        _semantic_splitter_cache = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # Split at semantic breaks
            breakpoint_threshold_amount=95,  # Only split at top 5% of semantic boundaries
        )
    
    return _semantic_splitter_cache

def get_fallback_splitter():
    """Get fallback recursive character splitter for when semantic splitting fails."""
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
        is_separator_regex=False,
    )

def chunk_documents(parsed_docs: List[Dict], progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Advanced semantic chunking using LangChain with MiniLM embeddings.
    
    Args:
        parsed_docs: List of parsed document dictionaries
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of chunk dictionaries: {text, document_name, page_number, chunk_id}
    """
    all_chunks = []
    total_docs = len(parsed_docs)
    
    if progress_callback:
        progress_callback("Initializing LangChain semantic splitter...", 0)
    
    # Get the semantic splitter (cached)
    try:
        semantic_splitter = get_semantic_splitter()
        fallback_splitter = get_fallback_splitter()
        splitter_type = "semantic"
        
        if progress_callback:
            progress_callback("✅ Semantic splitter loaded successfully", 5)
            
    except Exception as e:
        print(f"⚠️ Warning: Could not load semantic splitter ({e}). Using fallback.")
        semantic_splitter = None
        fallback_splitter = get_fallback_splitter()
        splitter_type = "fallback"
        
        if progress_callback:
            progress_callback("⚠️ Using fallback splitter", 5)
    
    for doc_idx, doc in enumerate(parsed_docs):
        doc_name = doc['document_name']
        parsed_output = doc['parsed_output']
        
        # Check for parsing errors
        if 'error' in parsed_output:
            print(f"⚠️ Skipping {doc_name}: {parsed_output['error']}")
            continue
        
        # Combine paragraphs and tables into text
        text_parts = []
        
        # Add paragraphs
        if 'paragraphs' in parsed_output:
            text_parts.extend(parsed_output['paragraphs'])
        
        # Add tables
        if 'tables' in parsed_output:
            for table in parsed_output['tables']:
                text_parts.append(f"\n[TABLE]\n{table}\n[/TABLE]\n")
        
        # Combine all text
        text = '\n\n'.join(text_parts)
        
        if not text.strip():
            print(f"⚠️ No text content found in {doc_name}")
            continue
        
        # Update progress
        if progress_callback:
            progress = 5 + ((doc_idx / total_docs) * 90)
            progress_callback(f"Processing {doc_name} with {splitter_type} chunking...", progress)
        
        try:
            # Try semantic chunking first
            if semantic_splitter is not None:
                try:
                    chunks = semantic_splitter.split_text(text)
                    chunking_method = "LangChain Semantic (MiniLM)"
                except Exception as e:
                    print(f"⚠️ Semantic chunking failed for {doc_name}: {e}")
                    chunks = fallback_splitter.split_text(text)
                    chunking_method = "LangChain Recursive (Fallback)"
            else:
                # Use fallback splitter
                chunks = fallback_splitter.split_text(text)
                chunking_method = "LangChain Recursive"
            
            # Process chunks
            for idx, chunk_text in enumerate(chunks):
                # Skip very short chunks
                if len(chunk_text.strip()) < 50:
                    continue
                
                # Create deterministic chunk ID based on content
                content_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()[:8]
                chunk_id = f"{doc_name}_{idx}_{content_hash}"
                
                chunk_dict = {
                    'text': chunk_text.strip(),
                    'document_name': doc_name,
                    'page_number': 1,  # Could be improved with actual page detection
                    'chunk_id': chunk_id,
                    'chunking_method': chunking_method,
                    'chunk_index': idx,
                    'total_chars': len(chunk_text)
                }
                all_chunks.append(chunk_dict)
                
        except Exception as e:
            print(f"❌ Error chunking document {doc_name}: {e}")
            # Continue with next document
            continue
    
    if progress_callback:
        progress_callback(f"✅ Chunking complete! Created {len(all_chunks)} semantic chunks", 100)
    
    # Print summary
    semantic_count = sum(1 for c in all_chunks if 'Semantic' in c.get('chunking_method', ''))
    fallback_count = len(all_chunks) - semantic_count
    
    print(f"""
📊 Chunking Summary:
- Total chunks: {len(all_chunks)}
- Semantic chunks: {semantic_count}
- Fallback chunks: {fallback_count}
- Average chunk size: {sum(c['total_chars'] for c in all_chunks) // len(all_chunks) if all_chunks else 0} chars
- Documents processed: {total_docs}
    """)
    
    return all_chunks 