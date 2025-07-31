"""
Module: embed_and_index.py
Functionality: NVIDIA-hosted embedding generation using NIM API and vector indexing.
"""
from typing import List, Dict, Optional, Callable, Any
import requests
import time
import os
import json
from pinecone import Pinecone, ServerlessSpec
from .document_registry import DocumentRegistry

def get_nvidia_api_key() -> str:
    """Get NVIDIA API key from environment variables."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is required. Get your key from https://build.nvidia.com/")
    return api_key

def generate_embeddings_nvidia(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using NVIDIA Llama Text Embed v2 model via NIM API.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors (4096 dimensions)
    """
    api_key = get_nvidia_api_key()
    
    url = "https://ai.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    all_embeddings = []
    
    # Process in batches to handle API limits
    batch_size = 20  # Conservative batch size for large embeddings
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        payload = {
            "input": batch_texts,
            "model": "nvidia/llama-text-embed-v2",
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract embeddings from response
            batch_embeddings = []
            for item in result.get("data", []):
                embedding = item.get("embedding")
                if embedding:
                    batch_embeddings.append(embedding)
            
            all_embeddings.extend(batch_embeddings)
            
            print(f"✅ Generated {len(batch_embeddings)} embeddings for batch {i//batch_size + 1}")
            
            # Rate limiting - be respectful to the API
            if i + batch_size < len(texts):
                time.sleep(0.1)
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"❌ NVIDIA API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Error processing NVIDIA embeddings: {e}")
    
    if len(all_embeddings) != len(texts):
        raise RuntimeError(f"❌ Embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}")
    
    print(f"✅ Generated {len(all_embeddings)} embeddings using NVIDIA Llama Text Embed v2 (4096 dimensions)")
    return all_embeddings

def generate_query_embedding_nvidia(query: str) -> List[float]:
    """
    Generate a single query embedding using NVIDIA Llama Text Embed v2 model.
    
    Args:
        query: Query text to embed
        
    Returns:
        Single embedding vector (4096 dimensions)
    """
    api_key = get_nvidia_api_key()
    
    url = "https://ai.api.nvidia.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": [query],
        "model": "nvidia/llama-text-embed-v2",
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract embedding from response
        data = result.get("data", [])
        if not data:
            raise RuntimeError("❌ No embedding data returned from NVIDIA API")
        
        embedding = data[0].get("embedding")
        if not embedding:
            raise RuntimeError("❌ No embedding found in NVIDIA API response")
        
        print(f"✅ Generated query embedding using NVIDIA Llama Text Embed v2 (4096 dimensions)")
        return embedding
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"❌ NVIDIA API request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ Error processing NVIDIA query embedding: {e}")

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

def delete_duplicate_vectors(pinecone_api_key: str, index_name: str = 'policy-index', dry_run: bool = True):
    """
    Delete duplicate vectors from Pinecone index based on content hash.
    
    Args:
        pinecone_api_key: Pinecone API key
        index_name: Name of the Pinecone index
        dry_run: If True, only report duplicates without deleting
        
    Returns:
        dict: Results of duplicate detection/deletion
    """
    pc = Pinecone(api_key=pinecone_api_key)
    
    if index_name not in pc.list_indexes().names():
        return {'error': f'Index {index_name} not found'}
    
    index = pc.Index(index_name)
    
    # Get all vectors (this might be slow for large indexes)
    print("🔍 Scanning index for duplicates...")
    
    # Query all vectors by fetching with empty filter
    all_vectors = {}
    content_hashes = {}
    duplicates = []
    
    try:
        # Get all vector IDs first
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        
        if total_vectors == 0:
            return {'message': 'No vectors in index', 'duplicates_found': 0}
        
        print(f"📊 Found {total_vectors} vectors in index")
        
        # Fetch vectors in batches to find duplicates
        # Note: This is a simplified approach - for large indexes, you'd need pagination
        query_response = index.query(
            vector=[0.0] * 4096,  # Dummy vector for NVIDIA Llama Text Embed v2 (4096 dimensions)
            top_k=min(10000, total_vectors),  # Limit to avoid memory issues
            include_metadata=True
        )
        
        matches = getattr(query_response, 'matches', [])
        for match in matches:
            vector_id = match.id
            metadata = match.metadata or {}
            content_hash = metadata.get('content_hash', '')
            
            if content_hash:
                if content_hash in content_hashes:
                    # Found duplicate
                    duplicates.append({
                        'duplicate_id': vector_id,
                        'original_id': content_hashes[content_hash],
                        'content_hash': content_hash,
                        'document_name': metadata.get('document_name', 'unknown')
                    })
                else:
                    content_hashes[content_hash] = vector_id
        
        print(f"🔍 Found {len(duplicates)} duplicate vectors")
        
        if not dry_run and duplicates:
            print("🗑️ Deleting duplicate vectors...")
            duplicate_ids = [dup['duplicate_id'] for dup in duplicates]
            
            # Delete in batches
            batch_size = 100
            deleted_count = 0
            
            for i in range(0, len(duplicate_ids), batch_size):
                batch = duplicate_ids[i:i + batch_size]
                index.delete(ids=batch)
                deleted_count += len(batch)
                print(f"Deleted {deleted_count}/{len(duplicate_ids)} duplicates...")
            
            return {
                'duplicates_found': len(duplicates),
                'duplicates_deleted': deleted_count,
                'remaining_vectors': total_vectors - deleted_count,
                'action': 'deleted'
            }
        else:
            return {
                'duplicates_found': len(duplicates),
                'duplicates_deleted': 0,
                'total_vectors': total_vectors,
                'action': 'dry_run' if dry_run else 'none_deleted',
                'duplicate_details': duplicates[:10]  # Show first 10 for review
            }
            
    except Exception as e:
        return {'error': f'Error processing duplicates: {str(e)}'}

def reindex_documents(pinecone_api_key: str, documents_to_reindex: List[str], 
                     index_name: str = 'policy-index'):
    """
    Remove and re-add specific documents to the index.
    
    Args:
        pinecone_api_key: Pinecone API key
        documents_to_reindex: List of document names to reindex
        index_name: Name of the Pinecone index
        
    Returns:
        dict: Results of reindexing operation
    """
    pc = Pinecone(api_key=pinecone_api_key)
    
    if index_name not in pc.list_indexes().names():
        return {'error': f'Index {index_name} not found'}
    
    index = pc.Index(index_name)
    
    deleted_vectors = []
    
    for doc_name in documents_to_reindex:
        print(f"🗑️ Removing existing vectors for document: {doc_name}")
        
        # Query vectors for this document
        query_response = index.query(
            vector=[0.0] * 4096,  # Dummy vector with correct dimensions for Llama Text Embed v2
            filter={'document_name': doc_name},
            top_k=10000,  # Get all chunks for this document
            include_metadata=True
        )
        
        matches = getattr(query_response, 'matches', [])
        if matches:
            vector_ids = [match.id for match in matches]
            index.delete(ids=vector_ids)
            deleted_vectors.extend(vector_ids)
            print(f"Deleted {len(vector_ids)} vectors for {doc_name}")
    
    return {
        'documents_processed': len(documents_to_reindex),
        'vectors_deleted': len(deleted_vectors),
        'message': f'Deleted {len(deleted_vectors)} vectors. Re-run indexing to add fresh vectors.'
    }

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
            progress_callback("Creating Pinecone index for NVIDIA embeddings...", 5)
        pc.create_index(
            name=index_name,
            dimension=4096,  # 4096 for NVIDIA Llama Text Embed v2
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
        progress_callback("Generating embeddings with NVIDIA NV-Embed-QA...", 15)
    
    # Generate embeddings using NVIDIA API ONLY
    texts = [chunk['text'] for chunk in chunks]
    embeddings = generate_embeddings_nvidia(texts)
    
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
        
        # Convert embedding to list safely (handle both numpy arrays and already-converted lists)
        if isinstance(embedding, list):
            embedding_list = embedding
        elif hasattr(embedding, 'tolist'):
            embedding_list = embedding.tolist()  # type: ignore
        else:
            # Last resort conversion
            embedding_list = list(map(float, embedding))
        
        vectors.append((chunk['chunk_id'], embedding_list, meta))
    
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
    return {"success": True, "indexed_count": len(chunks)}

def smart_index_documents(docs_folder: str, pinecone_api_key: str, index_name: str = 'policy-index', 
                         progress_callback: Optional[Callable] = None, save_parsed_text: bool = False) -> Dict[str, Any]:
    """
    Smart indexing - only processes new or changed documents
    
    Args:
        docs_folder: Folder containing PDF documents
        pinecone_api_key: Pinecone API key
        index_name: Name of the Pinecone index
        progress_callback: Optional callback for progress updates
        save_parsed_text: Whether to save parsed content to text files for inspection
    """
    registry = DocumentRegistry()
    
    # Check document status
    status = registry.get_document_status(docs_folder)
    files_to_process = registry.get_files_to_process(docs_folder)
    
    # Count status
    status_counts = {
        'indexed': len([f for f, s in status.items() if s == 'indexed']),
        'new': len([f for f, s in status.items() if s == 'new']),
        'changed': len([f for f, s in status.items() if s == 'changed']),
        'missing': len([f for f, s in status.items() if s == 'missing'])
    }
    
    if progress_callback:
        progress_callback(f"📊 Status: {status_counts['indexed']} indexed, {status_counts['new']} new, {status_counts['changed']} changed", 10)
    
    if not files_to_process:
        if progress_callback:
            progress_callback("🎉 All documents are already indexed and up-to-date!", 100)
        return {
            "status": "up_to_date",
            "processed_files": 0,
            "skipped_files": status_counts['indexed'],
            "total_time": 0,
            "status_counts": status_counts
        }
    
    # Process only new/changed files
    start_time = time.time()
    processed_files = []
    
    # Import here to avoid circular imports
    from .chunk_documents import chunk_documents
    
    total_files = len(files_to_process)
    
    for i, filename in enumerate(files_to_process):
        file_path = os.path.join(docs_folder, filename)
        
        if progress_callback:
            progress_callback(f"🔄 Processing {filename} ({i+1}/{total_files})...", 20 + (i / total_files) * 60)
        
        try:
            # Parse single document
            from .parse_documents import load_and_parse_from_folder
            parsed_docs = load_and_parse_from_folder(docs_folder, file_filter=[filename], save_parsed_text=save_parsed_text)
            
            if parsed_docs:
                # Chunk document
                chunks = chunk_documents(parsed_docs)
                
                # Index chunks
                result = index_chunks_in_pinecone(chunks, pinecone_api_key, index_name)
                
                if result.get('success', False):
                    # Mark as indexed
                    registry.mark_document_indexed(filename, file_path, len(chunks))
                    processed_files.append(filename)
                    
                    if progress_callback:
                        progress_callback(f"✅ {filename}: {len(chunks)} chunks indexed", 20 + ((i+1) / total_files) * 60)
                else:
                    if progress_callback:
                        progress_callback(f"❌ Failed to index {filename}", 20 + ((i+1) / total_files) * 60)
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"❌ Error processing {filename}: {str(e)}", 20 + ((i+1) / total_files) * 60)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if progress_callback:
        progress_callback(f"🎉 Smart indexing complete! Processed {len(processed_files)} files in {processing_time:.1f}s", 100)
    
    return {
        "status": "completed",
        "processed_files": len(processed_files),
        "skipped_files": status_counts['indexed'],
        "total_time": processing_time,
        "files_processed": processed_files,
        "status_counts": status_counts
    }

def force_reindex_all(docs_folder: str, pinecone_api_key: str, index_name: str = 'policy-index', 
                     progress_callback: Optional[Callable] = None, save_parsed_text: bool = False) -> Dict[str, Any]:
    """
    Force reindex all documents (clears registry and processes everything)
    
    Args:
        docs_folder: Folder containing PDF documents
        pinecone_api_key: Pinecone API key
        index_name: Name of the Pinecone index
        progress_callback: Optional callback for progress updates
        save_parsed_text: Whether to save parsed content to text files for inspection
    """
    registry = DocumentRegistry()
    
    if progress_callback:
        progress_callback("🔄 Force re-indexing: clearing registry and index...", 5)
    
    # Clear registry
    registry.clear_registry()
    
    # Clear Pinecone index
    try:
        clear_result = clear_pinecone_index(pinecone_api_key, index_name)
        if progress_callback:
            progress_callback(f"🗑️ Cleared {clear_result} vectors from index", 10)
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Failed to clear index: {str(e)}", 10)
        return {"status": "failed", "error": f"Could not clear index: {str(e)}"}
    
    # Process all documents
    return smart_index_documents(docs_folder, pinecone_api_key, index_name, progress_callback, save_parsed_text) 