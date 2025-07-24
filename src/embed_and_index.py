"""
Module: embed_and_index.py
Functionality: Embedding generation and vector indexing using sentence-transformers and Pinecone.
"""
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os


def index_chunks_in_pinecone(chunks: List[Dict], pinecone_api_key: str, pinecone_env: str, index_name: str = 'policy-index'):
    """
    For each chunk, generate embedding and upsert to Pinecone with metadata.
    """
    # Initialize Pinecone client (new API)
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if index exists, else create
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # 384 for MiniLM
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare and upsert vectors
    vectors = []
    for chunk in chunks:
        embedding = model.encode(chunk['text']).tolist()
        meta = {
            'document_name': chunk['document_name'],
            'page_number': chunk['page_number'],
            'chunk_id': chunk['chunk_id']
        }
        vectors.append((chunk['chunk_id'], embedding, meta))
    # Upsert in batches (Pinecone recommends batches of 100)
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i+100]
        ids = [v[0] for v in batch]
        embeds = [v[1] for v in batch]
        metas = [v[2] for v in batch]
        index.upsert(vectors=[(ids[j], embeds[j], metas[j]) for j in range(len(batch))])
    print(f"Indexed {len(chunks)} chunks into Pinecone index '{index_name}'.") 