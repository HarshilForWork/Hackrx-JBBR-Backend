"""
Module: chunk_documents.py
Functionality: Semantic chunking of parsed documents using sentence-transformers embeddings.
"""
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import re
import uuid

# Load a sentence transformer model (for semantic chunking guidance)
model = SentenceTransformer('all-MiniLM-L6-v2')


def split_text_semantically(text: str, max_chunk_length: int = 500) -> List[str]:
    """
    Splits text into semantically coherent chunks using sentence boundaries and a max length.
    For prototype: split by paragraphs, then merge if too short, or split if too long.
    """
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ''
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_length:
            current_chunk += para + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + '\n'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c.strip()]


def chunk_documents(parsed_docs: List[Dict]) -> List[Dict]:
    """
    For each parsed document, split into semantic chunks and assign metadata.
    Returns a flattened list of chunk dicts: {text, document_name, page_number, chunk_id}
    """
    all_chunks = []
    for doc in parsed_docs:
        doc_name = doc['document_name']
        text = doc['parsed_text']
        # For prototype, estimate page_number by splitting text by pages (if possible)
        # Here, we just assign page_number=1 for all chunks (improve if needed)
        semantic_chunks = split_text_semantically(text)
        for idx, chunk in enumerate(semantic_chunks):
            chunk_dict = {
                'text': chunk,
                'document_name': doc_name,
                'page_number': 1,  # Placeholder, could be improved
                'chunk_id': str(uuid.uuid4())
            }
            all_chunks.append(chunk_dict)
    return all_chunks 