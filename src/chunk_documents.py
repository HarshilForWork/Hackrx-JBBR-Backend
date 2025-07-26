"""
Module: chunk_documents.py
Functionality: Optimized semantic chunking of parsed documents.
"""
from typing import List, Dict, Optional, Callable
import re
import hashlib

def split_text_semantically(text: str, max_chunk_length: int = 800, overlap: int = 50) -> List[str]:
    """
    Splits text into semantically coherent chunks using sentence boundaries and a max length.
    
    Args:
        text: Input text to chunk
        max_chunk_length: Maximum characters per chunk
        overlap: Character overlap between chunks for context preservation
    """
    # Split by sentences for better semantic coherence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ''
    
    for sentence in sentences:
        # If adding this sentence would exceed max length
        if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + ' ' + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += (' ' if current_chunk else '') + sentence
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks (less than 50 characters)
    return [c for c in chunks if len(c.strip()) >= 50]

def chunk_documents(parsed_docs: List[Dict], progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Optimized function to split parsed documents into semantic chunks with metadata.
    
    Args:
        parsed_docs: List of parsed document dictionaries
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of chunk dictionaries: {text, document_name, page_number, chunk_id}
    """
    all_chunks = []
    total_docs = len(parsed_docs)
    
    if progress_callback:
        progress_callback("Starting document chunking...", 0)
    
    for doc_idx, doc in enumerate(parsed_docs):
        doc_name = doc['document_name']
        text = doc['parsed_text']
        
        # Update progress
        if progress_callback:
            progress = (doc_idx / total_docs) * 100
            progress_callback(f"Chunking document: {doc_name}", progress)
        
        # Improved chunking with better parameters
        semantic_chunks = split_text_semantically(text, max_chunk_length=800, overlap=100)
        
        for idx, chunk in enumerate(semantic_chunks):
            # Create deterministic chunk ID based on content
            content_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()[:8]
            chunk_id = f"{doc_name}_{idx}_{content_hash}"
            
            chunk_dict = {
                'text': chunk,
                'document_name': doc_name,
                'page_number': 1,  # Could be improved with actual page detection
                'chunk_id': chunk_id
            }
            all_chunks.append(chunk_dict)
    
    if progress_callback:
        progress_callback(f"Chunking complete! Created {len(all_chunks)} chunks", 100)
    
    print(f"Created {len(all_chunks)} chunks from {total_docs} documents")
    return all_chunks 