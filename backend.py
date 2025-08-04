# backend.py
from fastapi import FastAPI, File, UploadFile, Form, Request, Header, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import requests
from dotenv import load_dotenv
from src.pipeline import process_all_documents_pipeline, query_documents_sync
from pydantic import BaseModel
from typing import List, Optional
import time
from sentence_transformers import SentenceTransformer
import asyncio

load_dotenv()
app = FastAPI(title="HackRx Insurance API", description="API for querying insurance PDFs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = SentenceTransformer('all-MiniLM-L6-v2')

# Hardcoded API token - keep it simple
API_TOKEN = "552a90e441d8b2a0c195b5425dd982e0e71292568a08d2facf1ebc9434c1bcd0"

# Security scheme for Bearer token authentication
security = HTTPBearer()

class QueryPDFRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]  # List of questions to answer

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the token provided in the authorization header."""
    if credentials.scheme != "Bearer" or credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.post("/hackrx/run")
async def query_pdf(input: QueryPDFRequest, token: str = Depends(verify_token)):
    total_start_time = time.time()
    timings = {}
    pdf_url = input.documents  # Changed from pdf_url to documents
    queries = input.questions  # Changed from queries to questions
    if not pdf_url or not queries or not isinstance(queries, list):
        return JSONResponse({"error": "documents URL and questions (list) are required"}, status_code=400)

    # Download PDF
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        try:
            t0 = time.time()
            r = requests.get(pdf_url)
            r.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            timings["download"] = time.time() - t0
        except Exception as e:
            return JSONResponse({"error": f"Failed to download PDF: {str(e)}"}, status_code=400)

        # Embed queries and process PDF concurrently
        async def embed_single_query(idx, query):
            t0 = time.time()
            # Encode one query at a time
            embedding = model.encode(query, show_progress_bar=False)
            embedding_time = time.time() - t0
            return idx, embedding, embedding_time

        async def process_pdf():
            pinecone_key = os.getenv("PINECONE_API_KEY")
            t0 = time.time()
            result = await process_all_documents_pipeline(
                docs_dir=tmpdir,
                pinecone_api_key=pinecone_key,
                force_reprocess=True
            )
            timings["process_and_index"] = time.time() - t0
            return result, pinecone_key
            
        # Create embedding tasks for all queries
        embedding_tasks = []
        for idx, query in enumerate(queries):
            task = embed_single_query(idx, query)
            embedding_tasks.append(task)
            
        # Process PDF in parallel with embeddings
        pdf_task = asyncio.create_task(process_pdf())
        
        # Gather embedding results
        embedding_results = await asyncio.gather(*embedding_tasks)
        
        # Process embedding results and maintain order
        query_embeddings = [None] * len(queries)
        query_embedding_times = [0] * len(queries)
        for idx, embedding, time_taken in embedding_results:
            query_embeddings[idx] = embedding
            query_embedding_times[idx] = time_taken
            
        # Wait for PDF processing to complete
        result, pinecone_key = await pdf_task
        
        # Store embedding timing information
        timings["query_embedding_individual"] = query_embedding_times
        timings["query_embedding"] = sum(query_embedding_times)

        if not result.get("success"):
            return JSONResponse({"error": result.get("error", "Pipeline failed")}, status_code=500)

        # Process all queries together and ensure answers are in original order
        gemini_key = os.getenv("GEMINI_API_KEY")
        answers = [None] * len(queries)  # Pre-allocate list to maintain order
        query_times = [0] * len(queries)  # Pre-allocate timing list
        
        # Create a QueryProcessor once for all queries
        t0 = time.time()
        from src.query_processor import QueryProcessor
        processor = QueryProcessor(
            pinecone_api_key=pinecone_key,
            gemini_api_key=gemini_key,
            index_name="policy-index"
        )
        timings["processor_init"] = time.time() - t0
        
        # Process queries concurrently with asyncio
        async def process_single_query(idx, query, embedding):
            t_query_start = time.time()
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            # Process individual query using processor
            verdict = processor.process_query(
                query=query,
                query_embedding=embedding
            )
            
            # Extract answer and store at correct index to maintain order
            answer = verdict.get("evaluation", {}).get("answer")
            answers[idx] = answer
            query_times[idx] = time.time() - t_query_start
        
        # Create tasks for all queries
        query_tasks = []
        for idx, query in enumerate(queries):
            task = process_single_query(idx, query, query_embeddings[idx])
            query_tasks.append(task)
        
        # Run all query tasks concurrently
        await asyncio.gather(*query_tasks)
            
        timings["queries"] = query_times
        
        # Clean up Pinecone index after all queries are processed
        try:
            t0 = time.time()
            if processor.index:
                # Delete all vectors from the index
                processor.index.delete(delete_all=True)
                print("✅ Successfully deleted all vectors from Pinecone index")
            timings["cleanup_index"] = time.time() - t0
        except Exception as e:
            print(f"❌ Error cleaning up Pinecone index: {e}")
            timings["cleanup_index"] = 0
            
        timings["total_execution_time"] = time.time() - total_start_time

        # Create response with individual query times
        query_timing_details = []
        for idx, (query, time_taken) in enumerate(zip(queries, query_times)):
            query_timing_details.append({
                "query_index": idx,
                "query": query[:50] + "..." if len(query) > 50 else query,  # Truncate long queries
                "time_seconds": time_taken
            })

        # Calculate combined time for all queries
        total_query_time = sum(query_times)

        # Collect all timing metrics for each step
        all_timings = {
            "download_pdf": timings.get("download", 0),
            "query_embedding": {
                "total": timings.get("query_embedding", 0),
                "individual": timings.get("query_embedding_individual", [])
            },
            "pdf_processing_and_indexing": timings.get("process_and_index", 0),
            "query_processor_initialization": timings.get("processor_init", 0),
            "query_processing": {
                "total": total_query_time,
                "average": sum(query_times) / len(query_times) if query_times else 0,
                "individual": query_times
            },
            "cleanup_index": timings.get("cleanup_index", 0),
            "total_execution_time": timings.get("total_execution_time", 0)
        }

        # Return answers and detailed timing information
        return JSONResponse({
            "answers": answers,
            "timings": all_timings,
            "cleanup_status": "Vectors deleted from Pinecone index"
        })
     
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

# To run: uvicorn backend:app --reload