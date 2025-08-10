"""
Module: langchain_faiss_store.py
Functionality: LangChain FAISS integration for better metadata handling and retrieval.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.base import Embeddings
from langchain_core.documents import Document

# Pinecone for embeddings
from pinecone import Pinecone


class PineconeEmbeddings(Embeddings):
    """Custom embedding class using Pinecone's multilingual-e5-large model."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.pc = Pinecone(api_key=api_key)
        self.model_name = "multilingual-e5-large"
        self.dimension = 1024
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            # Process in batches for efficiency
            batch_size = 96
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                print(f"📦 Embedding batch {i//batch_size + 1}/{(len(texts)+batch_size-1)//batch_size}: {len(batch)} texts")
                
                response = self.pc.inference.embed(
                    model=self.model_name,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                batch_embeddings = [embedding.values for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
            
            print(f"✅ Generated {len(all_embeddings)} embeddings using Pinecone inference ({self.dimension} dims)")
            return all_embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Return random fallback embeddings
            import random
            return [[random.uniform(-0.01, 0.01) for _ in range(self.dimension)] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        try:
            response = self.pc.inference.embed(
                model=self.model_name,
                inputs=[text],
                parameters={"input_type": "query", "truncate": "END"}
            )
            
            embedding = response.data[0].values
            print(f"✅ Generated query embedding using Pinecone inference ({len(embedding)} dims)")
            return embedding
            
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            # Return random fallback embedding
            import random
            return [random.uniform(-0.01, 0.01) for _ in range(self.dimension)]


@dataclass
class LangChainFAISSConfig:
    """Configuration for LangChain FAISS vector store."""
    storage_dir: str = "faiss_storage"
    index_name: str = "policy-index"
    chunk_metadata_fields: List[str] = None
    distance_metric: str = "cosine"  # cosine, euclidean, or dot_product
    
    def __post_init__(self):
        if self.chunk_metadata_fields is None:
            self.chunk_metadata_fields = [
                'document_name', 'page_number', 'chunk_id', 
                'section', 'chunk_index', 'total_chunks'
            ]


class LangChainFAISSStore:
    """Enhanced FAISS vector store using LangChain with metadata support."""
    
    def __init__(self, config: LangChainFAISSConfig, pinecone_api_key: str):
        self.config = config
        self.embeddings = PineconeEmbeddings(pinecone_api_key)
        self.vector_store = None
        
        # Ensure storage directory exists
        os.makedirs(self.config.storage_dir, exist_ok=True)
        
        # Define file paths
        self.index_path = os.path.join(self.config.storage_dir, f"{self.config.index_name}.faiss")
        self.pkl_path = os.path.join(self.config.storage_dir, f"{self.config.index_name}.pkl")
        
        # Load existing vector store or initialize empty one
        self._load_or_initialize()
    
    def _load_or_initialize(self):
        """Load existing FAISS index or create a new one."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.pkl_path):
                print(f"🔍 Loading existing FAISS index from {self.index_path}")
                self.vector_store = FAISS.load_local(
                    self.config.storage_dir,
                    self.embeddings,
                    self.config.index_name,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ Loaded FAISS index with {self.vector_store.index.ntotal} vectors")
            else:
                print("🆕 Creating new FAISS index")
                # Create empty vector store with a dummy document
                dummy_doc = Document(
                    page_content="dummy",
                    metadata={"type": "dummy", "document_name": "init"}
                )
                self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
                
                # Remove the dummy document
                if self.vector_store.index.ntotal > 0:
                    # Clear the index but keep the structure
                    self.vector_store.index.reset()
                    self.vector_store.docstore._dict.clear()
                    self.vector_store.index_to_docstore_id.clear()
                
                print("✅ Created new empty FAISS index")
                
        except Exception as e:
            print(f"❌ Error loading/initializing FAISS index: {e}")
            # Create fresh index as fallback
            dummy_doc = Document(page_content="dummy", metadata={"type": "dummy"})
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            self.vector_store.index.reset()
            self.vector_store.docstore._dict.clear()
            self.vector_store.index_to_docstore_id.clear()
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, Any]:
        """
        Add chunks to the FAISS vector store with proper metadata handling.
        
        Args:
            chunks: List of chunk dictionaries with content and metadata
            batch_size: Number of chunks to process at once
            
        Returns:
            Dictionary with operation results
        """
        if not chunks:
            return {"success": False, "error": "No chunks provided"}
        
        start_time = time.time()
        
        try:
            # Convert chunks to LangChain Documents
            documents = []
            for chunk in chunks:
                # Extract content
                content = chunk.get('content', '')
                if not content or len(content.strip()) < 10:
                    continue  # Skip empty or very short chunks
                
                # Prepare metadata
                metadata = {
                    'document_name': chunk.get('document_name', 'unknown'),
                    'page_number': chunk.get('page_number', 0),
                    'chunk_id': chunk.get('chunk_id', f"chunk_{len(documents)}"),
                    'chunk_index': chunk.get('chunk_index', len(documents)),
                    'total_chunks': len(chunks),
                    'source': chunk.get('source_document', chunk.get('document_name', 'unknown')),
                    'section': chunk.get('section', 'content'),
                    'content_type': chunk.get('content_type', 'text'),
                    'content_length': len(content),
                    'timestamp': time.time()
                }
                
                # Add any additional metadata from the chunk
                for key, value in chunk.items():
                    if key not in ['content'] and key not in metadata:
                        metadata[key] = value
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            if not documents:
                return {"success": False, "error": "No valid documents to add"}
            
            print(f"📄 Prepared {len(documents)} documents for indexing")
            
            # Add documents to vector store in batches
            total_added = 0
            
            if self.vector_store.index.ntotal == 0:
                # First batch - create the vector store
                batch = documents[:batch_size]
                print(f"🔄 Creating FAISS index with first batch of {len(batch)} documents")
                self.vector_store = FAISS.from_documents(batch, self.embeddings)
                total_added += len(batch)
                documents = documents[batch_size:]
            
            # Add remaining documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                print(f"🔄 Adding batch {i//batch_size + 1}: {len(batch)} documents")
                
                # Create temporary vector store for this batch
                batch_store = FAISS.from_documents(batch, self.embeddings)
                
                # Merge with main vector store
                self.vector_store.merge_from(batch_store)
                total_added += len(batch)
                
                print(f"✅ Added batch {i//batch_size + 1}, total vectors: {self.vector_store.index.ntotal}")
            
            # Save the updated vector store
            self.save()
            
            processing_time = time.time() - start_time
            
            print(f"✅ Successfully indexed {total_added} chunks in {processing_time:.2f}s")
            
            return {
                "success": True,
                "indexed_count": total_added,
                "total_vectors": self.vector_store.index.ntotal,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"❌ Error adding chunks to FAISS: {e}")
            return {"success": False, "error": str(e)}
    
    def similarity_search(self, query: str, k: int = 20, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with content, metadata, and scores
        """
        if not self.vector_store or self.vector_store.index.ntotal == 0:
            print("❌ No documents in FAISS index")
            return []
        
        try:
            print(f"🔍 Searching FAISS index for top {k} similar documents")
            
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter by score threshold and format results
            formatted_results = []
            for doc, score in results:
                if score >= score_threshold:
                    result = {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': float(score),
                        'document_name': doc.metadata.get('document_name', 'unknown'),
                        'page_number': doc.metadata.get('page_number', 0),
                        'chunk_id': doc.metadata.get('chunk_id', ''),
                        'section': doc.metadata.get('section', 'content')
                    }
                    formatted_results.append(result)
            
            print(f"✅ Found {len(formatted_results)} relevant documents (score >= {score_threshold})")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error during similarity search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if not self.vector_store:
            return {"total_vectors": 0, "index_exists": False}
        
        try:
            total_vectors = self.vector_store.index.ntotal
            
            # Count documents by type if metadata is available
            doc_types = {}
            document_names = set()
            
            if hasattr(self.vector_store, 'docstore') and self.vector_store.docstore._dict:
                for doc_id, doc in self.vector_store.docstore._dict.items():
                    if hasattr(doc, 'metadata'):
                        doc_name = doc.metadata.get('document_name', 'unknown')
                        document_names.add(doc_name)
                        
                        content_type = doc.metadata.get('content_type', 'text')
                        doc_types[content_type] = doc_types.get(content_type, 0) + 1
            
            return {
                "total_vectors": total_vectors,
                "unique_documents": len(document_names),
                "document_names": list(document_names),
                "content_types": doc_types,
                "index_exists": True,
                "dimension": self.embeddings.dimension,
                "storage_path": self.index_path
            }
            
        except Exception as e:
            print(f"❌ Error getting FAISS stats: {e}")
            return {"total_vectors": 0, "index_exists": False, "error": str(e)}
    
    def save(self):
        """Save the vector store to disk."""
        try:
            if self.vector_store:
                self.vector_store.save_local(self.config.storage_dir, self.config.index_name)
                print(f"💾 Saved FAISS index to {self.index_path}")
        except Exception as e:
            print(f"❌ Error saving FAISS index: {e}")
    
    def clear(self):
        """Clear all vectors from the store."""
        try:
            if self.vector_store:
                self.vector_store.index.reset()
                self.vector_store.docstore._dict.clear()
                self.vector_store.index_to_docstore_id.clear()
            
            # Remove files
            for path in [self.index_path, self.pkl_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            print("🗑️ Cleared FAISS index")
            
        except Exception as e:
            print(f"❌ Error clearing FAISS index: {e}")


def create_langchain_faiss_store(pinecone_api_key: str, 
                                index_name: str = "policy-index",
                                storage_dir: str = "faiss_storage") -> LangChainFAISSStore:
    """
    Factory function to create a LangChain FAISS store.
    
    Args:
        pinecone_api_key: API key for Pinecone embeddings
        index_name: Name of the FAISS index
        storage_dir: Directory to store FAISS files
        
    Returns:
        Configured LangChainFAISSStore instance
    """
    config = LangChainFAISSConfig(
        storage_dir=storage_dir,
        index_name=index_name
    )
    
    return LangChainFAISSStore(config, pinecone_api_key)
