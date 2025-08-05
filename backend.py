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
from src.embed_and_index import generate_query_embedding_pinecone
from pinecone import Pinecone
from pydantic import BaseModel
from typing import List, Optional
import time
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

# Security scheme for Bearer token authentication
security = HTTPBearer()

# Hardcoded API token - keep it simple
API_TOKEN = "552a90e441d8b2a0c195b5425dd982e0e71292568a08d2facf1ebc9434c1bcd0"

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

        # Process PDF in a separate task
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
        
        # Create PDF processing task
        pdf_task = asyncio.create_task(process_pdf())
        
        # Embed all queries in a batch
        t0 = time.time()
        pinecone_key = os.getenv("PINECONE_API_KEY")

        # Process all queries in a single batch using Pinecone
        try:
            # Create Pinecone client directly instead of using get_pinecone_client
            pc = Pinecone(api_key=pinecone_key)
            model_name = "multilingual-e5-large"
            
            # Make the API call directly to ensure proper formatting
            response = pc.inference.embed(
                model=model_name,
                inputs=queries,
                parameters={"input_type": "query", "truncate": "END"}
            )
            
            # Process the response based on its structure
            if isinstance(response, dict) and 'data' in response:
                # Standard response format
                all_embeddings = [item['values'] for item in response['data']]
            elif isinstance(response, list):
                # Alternative response format
                all_embeddings = [item['values'] for item in response]
            elif hasattr(response, 'data'):
                # EmbeddingsList object format
                all_embeddings = [item['values'] for item in response.data]
            else:
                # Last resort: try to extract data directly from the response object
                try:
                    # Try to convert the response to a dict
                    response_dict = response.__dict__
                    if 'data' in response_dict:
                        all_embeddings = [item['values'] for item in response_dict['data']]
                    else:
                        # If we can't figure out the format, just use individual embedding
                        raise ValueError(f"Cannot extract embeddings from response")
                except:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                
            query_embedding_time = time.time() - t0
            
            # Store embedding timing information
            query_embedding_times = [query_embedding_time / len(queries)] * len(queries)
            timings["query_embedding_individual"] = query_embedding_times
            timings["query_embedding"] = query_embedding_time
            
            print(f"✅ Successfully batch-embedded {len(all_embeddings)} queries with {model_name}")
            
        except Exception as e:
            print(f"❌ Error in batch embedding: {e}")
            print(f"Response type: {type(response) if 'response' in locals() else 'Unknown'}")
            if 'response' in locals():
                print(f"Response attributes: {dir(response)}")
                if hasattr(response, 'data'):
                    print(f"Response.data type: {type(response.data)}")
                    if hasattr(response.data, '__len__'):
                        print(f"Response.data length: {len(response.data)}")
                        if len(response.data) > 0:
                            print(f"First item type: {type(response.data[0])}")
            
            # We'll use individual embedding as fallback since that's more reliable
            all_embeddings = []
            total_embedding_time = 0
            for query in queries:
                t_embed = time.time()
                embedding = generate_query_embedding_pinecone(query, pinecone_key)
                embed_time = time.time() - t_embed
                total_embedding_time += embed_time
                all_embeddings.append(embedding)
            
            # Update timing information for fallback case
            query_embedding_times = [total_embedding_time / len(queries)] * len(queries)
            timings["query_embedding_individual"] = query_embedding_times
            timings["query_embedding"] = total_embedding_time
        
        # Wait for PDF processing to complete
        result, pinecone_key = await pdf_task

        if not result.get("success"):
            return JSONResponse({"error": result.get("error", "Pipeline failed")}, status_code=500)

        # Process all queries together and ensure answers are in original order
        gemini_key = os.getenv("GEMINI_API_KEY")
        answers = [None] * len(queries)  # Pre-allocate list to maintain order
        query_times = [0] * len(queries)  # Pre-allocate timing list
        
        # Process queries using the pipeline's query_documents_sync function
        t0 = time.time()
        from src.pipeline import query_documents_sync
        from src.query_processor import QueryProcessor
        
        # Initialize the QueryProcessor for cleanup later
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
            
            # Process individual query using query_documents_sync from pipeline
            result = query_documents_sync(
                query=query,
                pinecone_api_key=pinecone_key,
                gemini_api_key=gemini_key,
                index_name="policy-index",
                query_embedding=embedding
            )
            
            # Extract important information from the result
            answer = result.get("evaluation", {}).get("answer")
            search_results = result.get("search_results", [])
            evaluation = result.get("evaluation", {})
            
            # Create a comprehensive response for this query, preserving the original structure
            query_response = {
                "search_results": search_results,
                "evaluation": evaluation,  # Keep the full evaluation object intact
                "api_status": result.get("api_status", {}),
                "status": result.get("status", "success"),
                "success": result.get("success", True),
                "query": query
            }
            
            # Store the comprehensive response
            answers[idx] = query_response
            
            query_times[idx] = time.time() - t_query_start
        
        # Create tasks for all queries
        query_tasks = []
        for idx, query in enumerate(queries):
            task = process_single_query(idx, query, all_embeddings[idx])
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

        # Create response with individual query times and additional info
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

        # Return answers with vector search results and detailed timing information
        return JSONResponse({
            "answers": answers,  # This now contains the full result structure for each query
            "timings": all_timings,
            "cleanup_status": "Vectors deleted from Pinecone index",
            "api_version": "2.1",  # Updated API version to reflect the new response format
            "model_info": {
                "embedding_model": "multilingual-e5-large",
                "temperature": 0.7
            }
        })
     
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

# To run: uvicorn backend:app --reload