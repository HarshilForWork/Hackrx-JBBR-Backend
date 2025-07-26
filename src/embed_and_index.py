"""
Module: embed_and_index.py
Functionality: Optimized embedding generation and vector indexing.
"""
from typing import List, Dict, Optional, Callable
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

# Global model cache to avoid reloading
_model_cache = None

def get_embedding_model():
    """Get cached embedding model to avoid reloading."""
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer('all-MiniLM-L6-v2')
    return _model_cache

def clear_pinecone_index(pinecone_api_key: str, index_name: str = 'policy-index'):
    """
    Clear all vectors from a Pinecone index.
    
    Args:
        pinecone_api_key: Pinecone API key
        index_name: Name of the Pinecone index to clear
        
    Returns:
        int: Number of vectors that were in the index before clearing
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    if index_name not in pc.list_indexes().names():
        return 0
    
    index = pc.Index(index_name)
    
    # Get current stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    
    # Delete all vectors
    index.delete(delete_all=True)
    
    return total_vectors

def get_index_stats(pinecone_api_key: str, index_name: str = 'policy-index'):
    """
    Get statistics about a Pinecone index.
    
    Args:
        pinecone_api_key: Pinecone API key
        index_name: Name of the Pinecone index
        
    Returns:
        dict: Index statistics
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        
        if index_name not in pc.list_indexes().names():
            return {'exists': False, 'total_vector_count': 0}
        
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        return {
            'exists': True,
            'total_vector_count': stats.get('total_vector_count', 0),
            'dimension': stats.get('dimension', 0),
            'index_fullness': stats.get('index_fullness', 0.0),
            'namespaces': stats.get('namespaces', {})
        }
    except Exception as e:
        return {'exists': False, 'error': str(e), 'total_vector_count': 0}

def index_chunks_in_pinecone(chunks: List[Dict], pinecone_api_key: str, pinecone_env: str, 
                           index_name: str = 'policy-index', progress_callback: Optional[Callable] = None):
    """
    Optimized function to generate embeddings and upsert to Pinecone with metadata.
    
    Args:
        chunks: List of chunk dictionaries
        pinecone_api_key: Pinecone API key
        pinecone_env: Pinecone environment (kept for compatibility)
        index_name: Name of the Pinecone index
        progress_callback: Optional callback function for progress updates
    """
    if progress_callback:
        progress_callback("Initializing Pinecone...", 0)
    
    # Initialize Pinecone client (new API)
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if index exists, else create
    if index_name not in pc.list_indexes().names():
        if progress_callback:
            progress_callback("Creating Pinecone index...", 5)
        pc.create_index(
            name=index_name,
            dimension=384,  # 384 for MiniLM
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        # Wait for index to be ready
        time.sleep(10)
    
    index = pc.Index(index_name)
    
    if progress_callback:
        progress_callback("Loading embedding model...", 10)
    
    # Use cached model
    model = get_embedding_model()
    
    if progress_callback:
        progress_callback("Generating embeddings...", 15)
    
    # OPTIMIZATION 1: Batch encode all texts at once instead of one by one
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    
    if progress_callback:
        progress_callback("Preparing vectors for indexing...", 60)
    
    # OPTIMIZATION 2: Prepare vectors efficiently
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = {
            'text': chunk['text'][:1000],  # Truncate text to avoid metadata size limits
            'document_name': chunk['document_name'],
            'page_number': chunk['page_number'],
            'chunk_id': chunk['chunk_id']
        }
        vectors.append((chunk['chunk_id'], embedding.tolist(), meta))
    
    if progress_callback:
        progress_callback("Upserting to Pinecone...", 70)
    
    # OPTIMIZATION 3: Upsert in optimal batches with progress tracking
    batch_size = 100  # Pinecone recommends batches of 100
    total_batches = len(vectors) // batch_size + (1 if len(vectors) % batch_size > 0 else 0)
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            
            # Update progress
            if progress_callback:
                batch_num = i // batch_size + 1
                progress = 70 + (batch_num / total_batches) * 25
                progress_callback(f"Indexed batch {batch_num}/{total_batches}", progress)
                
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
            raise
    
    if progress_callback:
        progress_callback("Indexing complete!", 100)
    
    print(f"Successfully indexed {len(chunks)} chunks into Pinecone index '{index_name}'.")
    return len(chunks) 