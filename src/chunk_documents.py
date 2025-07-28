"""
Module: chunk_documents.py
Functionality: Semantic chunking using LangChain with MiniLM embeddings for intelligent text splitting.
"""
from typing import List, Dict, Optional, Callable, Any
import hashlib
import time
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

def get_semantic_splitter():
    """Get LangChain semantic splitter with MiniLM embeddings."""
    # Use MiniLM for semantic chunking embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

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
    use_semantic: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    save_parsed_text: bool = False,
    output_dir: str = "output"
) -> List[Dict[str, Any]]:
    """
    Chunk documents using LangChain semantic splitter with MiniLM.
    
    Args:
        parsed_content: List of parsed document dictionaries
        use_semantic: Whether to use semantic chunking (True) or basic recursive (False)
        chunk_size: Maximum chunk size for recursive splitter
        chunk_overlap: Overlap between chunks
        save_parsed_text: Whether to save parsed text files
        output_dir: Directory to save parsed text files
        
    Returns:
        List of chunked document dictionaries
    """
    if not parsed_content:
        return []
    
    print(f"🔄 Starting document chunking with {'semantic' if use_semantic else 'recursive'} splitter...")
    
    chunked_docs = []
    
    # Choose splitter based on preference
    if use_semantic:
        print("📊 Using LangChain SemanticChunker with MiniLM embeddings...")
        splitter = get_semantic_splitter()
    else:
        print("📝 Using LangChain RecursiveCharacterTextSplitter...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
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
                        'splitter_type': 'semantic' if use_semantic else 'recursive'
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
    Converts old format to new format and uses semantic chunking with MiniLM.
    """
    if progress_callback:
        progress_callback("🔄 Converting to new format for semantic chunking...", 0)
    
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
        progress_callback("📊 Using LangChain SemanticChunker with MiniLM...", 20)
    
    # Use new chunking function with semantic splitting
    chunked_docs = chunk_documents(new_format_docs, use_semantic=True)
    
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