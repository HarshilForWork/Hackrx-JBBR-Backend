# backend.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
import tempfile
import os
import requests
from dotenv import load_dotenv
from src.pipeline import process_all_documents_pipeline, query_documents_sync
from pydantic import BaseModel
from typing import List
import time
from sentence_transformers import SentenceTransformer
import asyncio

load_dotenv()
app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
class QueryPDFRequest(BaseModel):
    pdf_url: str
    queries: List[str]

@app.post("/query_pdf")
async def query_pdf(input: QueryPDFRequest):
    timings = {}
    pdf_url = input.pdf_url
    queries = input.queries
    if not pdf_url or not queries or not isinstance(queries, list):
        return JSONResponse({"error": "pdf_url and queries (list) are required"}, status_code=400)

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

        # Query the indexed document for each query
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

        return JSONResponse({"answers": answers, "timings": timings, "query_embeddings": query_embeddings.tolist()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

# To run: uvicorn backend:app --reload