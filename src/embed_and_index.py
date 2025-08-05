"""
Module: embed_and_index.py
Functionality: Advanced embedding generation using Pinecone's text embeddings and vector indexing with smart document management.
"""
from typing import List, Dict, Optional, Callable, Any
import requests
import time
import os
from pinecone import Pinecone, ServerlessSpec
from .document_registry import DocumentRegistry

# Global Pinecone client cache
_pinecone_client = None

def get_pinecone_client(api_key: str):
    """Get or initialize Pinecone client."""
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=api_key)
        print("✅ Initialized Pinecone client for embeddings")
    return _pinecone_client

def generate_embeddings_batch(texts: List[str], pinecone_api_key: str, model: str = "multilingual-e5-large") -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Pinecone's embedding service.
    
    Args:
        texts: List of texts to embed
        pinecone_api_key: Pinecone API key
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    try:
        pc = get_pinecone_client(pinecone_api_key)
        
        # Use Pinecone's embedding service
        response = pc.inference.embed(
            model=model,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        # Extract embeddings from response
        embeddings = [item['values'] for item in response['data']]
        print(f"✅ Generated {len(embeddings)} embeddings using Pinecone {model}")
        return embeddings
        
    except Exception as e:
        print(f"❌ Pinecone batch embedding error: {e}")
        # Fallback to sentence transformers
        return generate_embeddings_fallback(texts)
    """
    Generate embeddings using Pinecone's embedding service.
    
    Args:
        texts: List of texts to embed
        api_key: Pinecone API key
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    try:
        pc = get_pinecone_client(api_key)
        
        # Use Pinecone's embedding service
        response = pc.inference.embed(
            model=model,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        # Extract embeddings from response
        embeddings = [item['values'] for item in response['data']]
        print(f"✅ Generated {len(embeddings)} embeddings using Pinecone {model}")
        return embeddings
        
    except Exception as e:
        print(f"❌ Pinecone embedding error: {e}")
        # Fallback to sentence transformers
        return generate_embeddings_fallback(texts)

def generate_embeddings_fallback(texts: List[str]) -> List[List[float]]:
    """
    Fallback embedding generation using SentenceTransformers.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_raw = model.encode(texts)
        embeddings = [emb.tolist() for emb in embeddings_raw]
        print(f"✅ Generated {len(embeddings)} embeddings using fallback SentenceTransformer")
        return embeddings
    except Exception as e:
        print(f"❌ Fallback embedding error: {e}")
        # Return zero vectors as last resort
        return [[0.0] * 384 for _ in texts]

def generate_query_embedding_pinecone(query: str, api_key: str, model: str = "multilingual-e5-large") -> List[float]:
    """
    Generate a single query embedding using Pinecone's embedding service.
    
    Args:
        query: Query text to embed
        api_key: Pinecone API key
        model: Embedding model to use
        
    Returns:
        Query embedding vector
    """
    try:
        pc = get_pinecone_client(api_key)
        
        # Use Pinecone's embedding service for query
        response = pc.inference.embed(
            model=model,
            inputs=[query],
            parameters={"input_type": "query", "truncate": "END"}
        )
        
        # Extract embedding from response
        # Try different possible response formats
        if isinstance(response, list) and len(response) > 0:
            embedding = response[0]['values']
        elif hasattr(response, 'data') and len(response.data) > 0:
            embedding = response.data[0]['values']
        elif isinstance(response, dict) and 'data' in response:
            embedding = response['data'][0]['values']
        else:
            # If we can't parse the response, raise an error
            raise ValueError(f"Unexpected response format: {type(response)}")
        
        print(f"✅ Generated query embedding using Pinecone {model} ({len(embedding)} dimensions)")
        return embedding
        
    except Exception as e:
        print(f"❌ Pinecone query embedding error: {e}")
        # Fallback to sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            model_st = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model_st.encode([query])[0].tolist()
            print(f"✅ Generated query embedding using fallback SentenceTransformer ({len(embedding)} dimensions)")
            return embedding
        except Exception as e2:
            print(f"❌ Fallback query embedding error: {e2}")
            return [0.0] * 384
        
        # Convert to list safely
        if hasattr(embeddings_raw, 'tolist'):
            embeddings = embeddings_raw.tolist()
        else:
            embeddings = [list(map(float, emb)) for emb in embeddings_raw]
        
        print(f"✅ Generated {len(embeddings)} embeddings using Llama Text Embed v2")
        return embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        # Fallback to sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_raw = model.encode(texts)
        
        # Convert to list safely
        if hasattr(embeddings_raw, 'tolist'):
            return embeddings_raw.tolist()
        else:
            return [list(map(float, emb)) for emb in embeddings_raw]

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
            vector=[0.0] * 384,  # Dummy vector for metadata-only query
            top_k=min(10000, total_vectors),  # Limit to avoid memory issues
            include_metadata=True
        )
        
        for match in query_response.matches:
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
            vector=[0.0] * 384,  # Dummy vector
            filter={'document_name': doc_name},
            top_k=10000,  # Get all chunks for this document
            include_metadata=True
        )
        
        if query_response.matches:
            vector_ids = [match.id for match in query_response.matches]
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

def generate_embeddings_pinecone(texts: List[str], api_key: str) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Pinecone inference API.
    
    Args:
        texts: List of text strings to embed
        api_key: Pinecone API key
        
    Returns:
        List of embedding vectors
    """
    try:
        pc = Pinecone(api_key=api_key)
        
        # Use Pinecone inference API with multilingual-e5-large model
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        # Extract the embedding vectors
        return [record['values'] for record in embeddings]
        
    except Exception as e:
        print(f"❌ Error generating embeddings with Pinecone inference: {e}")
        # Return zero vectors as fallback (1024 dimensions for multilingual-e5-large)
        return [[0.0] * 1024 for _ in texts]

def check_or_create_pinecone_index(pinecone_api_key: str, index_name: str = 'policy-index', 
                                  required_dimension: int = 1024, progress_callback: Optional[Callable] = None) -> bool:
    """
    Check if index exists with correct dimensions, delete and recreate if needed.
    
    Args:
        pinecone_api_key: Pinecone API key
        index_name: Name of the Pinecone index
        required_dimension: Required embedding dimension (1024 for multilingual-e5-large)
        progress_callback: Optional callback for progress updates
        
    Returns:
        bool: True if index is ready, False if failed
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes:
            # Check dimension
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            current_dimension = stats.get('dimension', 0)
            
            if current_dimension != required_dimension:
                print(f"⚠️ Index '{index_name}' has {current_dimension} dimensions, but we need {required_dimension}")
                print(f"🗑️ Deleting existing index to recreate with correct dimensions...")
                
                if progress_callback:
                    progress_callback(f"Deleting old index ({current_dimension}D)...", 10)
                
                # Delete the existing index
                pc.delete_index(index_name)
                
                # Wait for deletion to complete
                print("⏳ Waiting for index deletion to complete...")
                time.sleep(15)
                
                # Create new index with correct dimensions
                print(f"🏗️ Creating new index with {required_dimension} dimensions...")
                if progress_callback:
                    progress_callback(f"Creating new index ({required_dimension}D)...", 20)
                
                pc.create_index(
                    name=index_name,
                    dimension=required_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                print("⏳ Waiting for new index to be ready...")
                time.sleep(20)
                
                print(f"✅ Successfully recreated index '{index_name}' with {required_dimension} dimensions")
                return True
            else:
                print(f"✅ Index '{index_name}' already exists with correct {required_dimension} dimensions")
                return True
        else:
            # Create new index
            print(f"🏗️ Creating new index '{index_name}' with {required_dimension} dimensions...")
            if progress_callback:
                progress_callback(f"Creating new index ({required_dimension}D)...", 15)
            
            pc.create_index(
                name=index_name,
                dimension=required_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(15)
            
            print(f"✅ Successfully created index '{index_name}' with {required_dimension} dimensions")
            return True
            
    except Exception as e:
        print(f"❌ Error managing Pinecone index: {e}")
        if progress_callback:
            progress_callback(f"Index creation failed: {e}", -1)
        return False

def index_chunks_in_pinecone(chunks: List[Dict], pinecone_api_key: str, pinecone_env: str, 
                           index_name: str = 'policy-index', progress_callback: Optional[Callable] = None):
    """
    Optimized function to generate embeddings and upsert to Pinecone with metadata.
    Uses Pinecone inference API with proper dimension handling.
    
    Args:
        chunks: List of chunk dictionaries
        pinecone_api_key: Pinecone API key
        pinecone_env: Pinecone environment (kept for compatibility)
        index_name: Name of the Pinecone index
        progress_callback: Optional callback function for progress updates
    """
    if progress_callback:
        progress_callback("Initializing Pinecone...", 0)
    
    # Check or create index with correct dimensions (1024 for multilingual-e5-large)
    if not check_or_create_pinecone_index(pinecone_api_key, index_name, 1024, progress_callback):
        print("❌ Failed to create or verify Pinecone index")
        if progress_callback:
            progress_callback("Failed to create index", -1)
        return False
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    
    if progress_callback:
        progress_callback("Generating embeddings with Pinecone inference...", 15)
    
    # Generate embeddings using Pinecone inference API
    texts = [chunk['content'] for chunk in chunks]
    
    try:
        embeddings = generate_embeddings_pinecone(texts, pinecone_api_key)
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        if progress_callback:
            progress_callback(f"Error generating embeddings: {e}", -1)
        return False
    
    if progress_callback:
        progress_callback("Preparing vectors for indexing...", 60)
    
    # OPTIMIZATION 2: Prepare vectors efficiently
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = {
            'text': chunk['content'][:1000],  # Truncate text to avoid metadata size limits
            'document_name': chunk['document_name'],
            'page_number': chunk.get('page_number', 0),  # Use get() with default since this field might not exist
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
    from .chunk_documents_optimized import chunk_documents_optimized
    
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
                # Transform to expected format for optimized chunking
                transformed_docs = []
                for doc in parsed_docs:
                    doc_name = doc.get('document_name', 'unknown')
                    parsed_output = doc.get('parsed_output', {})
                    content = (parsed_output.get('content', '') or 
                              parsed_output.get('text', '') or 
                              parsed_output.get('cleaned_text', ''))
                    
                    transformed_doc = {
                        'document_name': doc_name,
                        'content': content,
                        'ordered_content': parsed_output.get('ordered_content', [])
                    }
                    transformed_docs.append(transformed_doc)
                
                # Chunk document with optimized chunking
                chunks = chunk_documents_optimized(transformed_docs)
                
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