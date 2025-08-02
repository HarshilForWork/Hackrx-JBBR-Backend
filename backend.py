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
        async def embed_queries():
            t0 = time.time()
            embeddings = model.encode(queries, show_progress_bar=False)
            timings["query_embedding"] = time.time() - t0
            return embeddings

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

        # Run both tasks concurrently
        query_embed_task = asyncio.create_task(embed_queries())
        pdf_task = asyncio.create_task(process_pdf())
        query_embeddings, (result, pinecone_key) = await asyncio.gather(query_embed_task, pdf_task)

        if not result.get("success"):
            return JSONResponse({"error": result.get("error", "Pipeline failed")}, status_code=500)

        # Process all queries concurrently
        gemini_key = os.getenv("GEMINI_API_KEY")
        answers = []
        query_times = []
        for idx, query in enumerate(queries):
            t0 = time.time()
            embedding = query_embeddings[idx]
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            verdict = query_documents_sync(
                query=query,
                pinecone_api_key=pinecone_key,
                gemini_api_key=gemini_key,
                index_name="policy-index",
                query_embedding=embedding  # Pass the precomputed embedding
            )
            answer = verdict.get("evaluation", {}).get("answer")
            answers.append(answer)
            query_times.append(time.time() - t0)
        timings["queries"] = query_times
        timings["total_execution_time"] = time.time() - total_start_time

        # Only return answers in the response along with timing information
        return JSONResponse({
            "answers": answers,
            "performance": {
                "total_time_seconds": timings["total_execution_time"],
                "avg_query_time_seconds": sum(timings["queries"]) / len(timings["queries"]) if timings["queries"] else 0
            }
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

# To run: uvicorn backend:app --reload