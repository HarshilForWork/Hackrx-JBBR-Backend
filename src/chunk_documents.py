"""
Module: chunk_documents.py
Functionality: Lightweight and reliable text chunking without semantic dependencies.
"""
from typing import List, Dict, Optional, Callable, Any
import hashlib
import time
import os
import re

def get_simple_splitter(chunk_size: int = 1000, chunk_overlap: int = 150):
    """Get a simple but reliable text splitter without external dependencies."""
    return SimpleChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

class SimpleChunker:
    """
    A simple, reliable text chunker that doesn't require ML models.
    Uses intelligent splitting based on document structure.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Define separators in order of preference (semantic to structural)
        self.separators = [
            "\n\n\n",      # Multiple line breaks (section separators)
            "\n\n",        # Double line breaks (paragraph separators)
            "\n",          # Single line breaks
            ". ",          # Sentence ends
            "! ",          # Exclamation sentences
            "? ",          # Question sentences
            "; ",          # Semi-colons
            ", ",          # Commas
            " ",           # Spaces
            ""             # Character level (last resort)
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using intelligent separators.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Start with the full text
        chunks = [text]
        
        # Apply separators in order of preference
        for separator in self.separators:
            new_chunks = []
            
            for chunk in chunks:
                if len(chunk) <= self.chunk_size:
                    # Chunk is already small enough
                    new_chunks.append(chunk)
                else:
                    # Split this chunk further
                    split_chunks = self._split_by_separator(chunk, separator)
                    new_chunks.extend(split_chunks)
            
            chunks = new_chunks
            
            # Check if all chunks are now small enough
            if all(len(chunk) <= self.chunk_size for chunk in chunks):
                break
        
        # Add overlap between chunks for better context
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        # Remove empty chunks and strip whitespace
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by a specific separator while respecting chunk size."""
        if not separator:
            # Character-level splitting (last resort)
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Add separator back to the part (except for the last part)
            if part != parts[-1]:
                part_with_sep = part + separator
            else:
                part_with_sep = part
            
            # Check if adding this part would exceed chunk size
            if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                current_chunk += part_with_sep
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk)
                
                # Start new chunk with this part
                if len(part_with_sep) <= self.chunk_size:
                    current_chunk = part_with_sep
                else:
                    # Part is too large, needs further splitting
                    chunks.append(part_with_sep)
                    current_chunk = ""
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks for better context."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no previous overlap needed
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                
                # Get last N characters from previous chunk as overlap
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
                
                # Find a good breaking point in the overlap (prefer word boundaries)
                overlap_text = self._find_word_boundary(overlap_text, from_end=True)
                
                # Combine overlap with current chunk
                combined = overlap_text + " " + chunk
                overlapped_chunks.append(combined)
        
        return overlapped_chunks
    
    def _find_word_boundary(self, text: str, from_end: bool = False) -> str:
        """Find a good word boundary to break text."""
        if not text:
            return text
        
        # Try to break at word boundaries
        if from_end:
            # Find last space or punctuation
            for i in range(len(text) - 1, -1, -1):
                if text[i] in [' ', '\n', '.', '!', '?', ';', ',']:
                    return text[i + 1:]
        else:
            # Find first space or punctuation
            for i in range(len(text)):
                if text[i] in [' ', '\n', '.', '!', '?', ';', ',']:
                    return text[:i]
        
        # If no good boundary found, return as is
        return text

def generate_chunk_id(doc_name: str, content: str, chunk_index: int) -> str:
    """Generate a unique chunk ID based on document name, content hash, and index."""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    return f"{doc_name}_{chunk_index}_{content_hash}"

def save_parsed_text_file_ordered(doc_name: str, ordered_content: List[Dict], output_dir: str):
    """Save ordered parsed text to file for inspection."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove file extension and add suffix
    base_name = doc_name.rsplit('.', 1)[0] if '.' in doc_name else doc_name
    output_path = os.path.join(output_dir, f"{base_name}_parsed_ORDERED.txt")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"=== ORDERED PARSED CONTENT: {doc_name} ===\n")
        f.write(f"Total items: {len(ordered_content)}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, item in enumerate(ordered_content, 1):
            content_type = item.get('type', 'text').upper()
            page_num = item.get('page', 1)
            item_content = item.get('content', '')
            
            f.write(f"[ITEM {i}] {content_type} (Page {page_num})\n")
            f.write(f"{item_content}\n")
            f.write("-" * 40 + "\n\n")
    
    print(f"📄 Saved ordered parsed text: {output_path}")

def chunk_documents(
    parsed_content: List[Dict[str, Any]], 
    use_semantic: bool = False,  # Changed default to False
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    save_parsed_text: bool = False,
    output_dir: str = "output"
) -> List[Dict[str, Any]]:
    """
    Chunk documents using lightweight and reliable text splitter.
    
    Args:
        parsed_content: List of parsed document dictionaries
        use_semantic: Deprecated - kept for compatibility (ignored)
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        save_parsed_text: Whether to save parsed text files
        output_dir: Directory to save parsed text files
        
    Returns:
        List of chunked document dictionaries
    """
    if not parsed_content:
        return []
    
    print(f"🔄 Starting document chunking with lightweight reliable splitter...")
    
    chunked_docs = []
    
    # Always use the simple splitter for reliability
    print("� Using SimpleChunker (lightweight, reliable, no ML dependencies)...")
    splitter = get_simple_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for doc_data in parsed_content:
        doc_name = doc_data.get('document_name', 'unknown')
        content = doc_data.get('content', '')
        
        if not content.strip():
            print(f"⚠️ Skipping empty document: {doc_name}")
            continue
            
        print(f"📄 Chunking document: {doc_name} ({len(content):,} characters)")
        
        try:
            # Handle ordered content if available
            if 'ordered_content' in doc_data and doc_data['ordered_content']:
                # Process ordered content maintaining sequence - handle both formats
                ordered_text = ""
                for item in doc_data['ordered_content']:
                    if isinstance(item, dict):
                        # New format: {'content': 'text', 'type': 'text', 'page': 1}
                        content_type = item.get('type', 'text')
                        item_content = item.get('content', '')
                        page_num = item.get('page', 1)
                        
                        if content_type == 'table':
                            ordered_text += f"\n\n[TABLE on Page {page_num}]\n{item_content}\n\n"
                        else:
                            ordered_text += f"{item_content}\n"
                    elif isinstance(item, str):
                        # Current format: just strings
                        ordered_text += f"{item}\n"
                    else:
                        # Fallback: convert to string
                        ordered_text += f"{str(item)}\n"
                
                # Save ordered text file if requested
                if save_parsed_text:
                    # Convert strings to dict format for saving function
                    ordered_content_for_save = []
                    for i, item in enumerate(doc_data['ordered_content']):
                        if isinstance(item, dict):
                            ordered_content_for_save.append(item)
                        else:
                            ordered_content_for_save.append({
                                'type': 'text',
                                'content': str(item),
                                'page': 1
                            })
                    save_parsed_text_file_ordered(doc_name, ordered_content_for_save, output_dir)
                
                content_to_split = ordered_text.strip()
            else:
                content_to_split = content
            
            # Split the content
            chunks = splitter.split_text(content_to_split)
            
            print(f"✅ Created {len(chunks)} chunks for {doc_name}")
            
            # Create chunk dictionaries
            for i, chunk in enumerate(chunks):
                chunk_id = generate_chunk_id(doc_name, chunk, i)
                
                chunk_dict = {
                    'chunk_id': chunk_id,
                    'document_name': doc_name,
                    'chunk_index': i,
                    'content': chunk.strip(),
                    'metadata': {
                        'source': doc_name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk),
                        'splitter_type': 'simple_reliable'
                    }
                }
                
                # Add original metadata if available
                if 'metadata' in doc_data:
                    chunk_dict['metadata'].update(doc_data['metadata'])
                
                chunked_docs.append(chunk_dict)
                
        except Exception as e:
            print(f"❌ Error chunking document {doc_name}: {e}")
            continue
    
    print(f"✅ Document chunking complete! Generated {len(chunked_docs)} total chunks")
    return chunked_docs

# Backwards compatibility function for old API
def chunk_documents_old_format(parsed_docs: List[Dict], progress_callback: Optional[Callable] = None) -> List[Dict]:
    """
    Backwards compatibility wrapper for old format.
    Converts old format to new format and uses lightweight reliable chunking.
    """
    if progress_callback:
        progress_callback("🔄 Converting to new format for reliable chunking...", 0)
    
    # Convert old format to new format
    new_format_docs = []
    for doc in parsed_docs:
        if 'parsed_output' in doc and isinstance(doc['parsed_output'], dict):
            parsed_output = doc['parsed_output']
            
            # Handle ordered content if available
            if 'ordered_content' in parsed_output and parsed_output['ordered_content']:
                # FIXED: Handle both string list and dict list formats
                ordered_content = parsed_output['ordered_content']
                content_parts = []
                
                for item in ordered_content:
                    if isinstance(item, dict):
                        # New format: {'content': 'text', 'type': 'text', 'page': 1}
                        content_parts.append(str(item.get('content', '')))
                    elif isinstance(item, str):
                        # Current format: just strings
                        content_parts.append(item)
                    else:
                        # Fallback: convert to string
                        content_parts.append(str(item))
                
                new_doc = {
                    'document_name': doc['document_name'],
                    'content': '\n\n'.join(content_parts),
                    'ordered_content': parsed_output['ordered_content']
                }
            else:
                # Fallback for older structures (if any)
                text_parts = []
                if 'paragraphs' in parsed_output:
                    text_parts.extend(parsed_output.get('paragraphs', []))
                if 'tables' in parsed_output:
                    text_parts.extend(parsed_output.get('tables', []))
                
                new_doc = {
                    'document_name': doc['document_name'],
                    'content': '\n\n'.join(map(str, text_parts))
                }
            
            new_format_docs.append(new_doc)
    
    if progress_callback:
        progress_callback("� Using SimpleChunker (lightweight, reliable)...", 20)
    
    # Use new chunking function with simple reliable splitting
    chunked_docs = chunk_documents(new_format_docs, use_semantic=False)
    
    if progress_callback:
        progress_callback("✅ Converting back to old format...", 90)
    
    # Convert back to old format for compatibility
    old_format_chunks = []
    for chunk in chunked_docs:
        old_chunk = {
            'text': chunk['content'],
            'document_name': chunk['document_name'],
            'page_number': 1,  # Default page number
            'chunk_id': chunk['chunk_id'],
            'chunking_method': 'Semantic (MiniLM)',
            'chunk_index': chunk['chunk_index'],
            'total_chars': len(chunk['content'])
        }
        old_format_chunks.append(old_chunk)
    
    if progress_callback:
        progress_callback(f"🎉 Semantic chunking complete! Created {len(old_format_chunks)} chunks", 100)
    
    return old_format_chunks