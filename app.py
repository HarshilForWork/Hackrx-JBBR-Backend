from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
import shutil
import os
from docling.document_converter import DocumentConverter

app = FastAPI()

@app.post("/convert", response_class=PlainTextResponse)
async def convert_pdf(file: UploadFile = File(...)):
    # Save uploaded PDF to disk
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Convert PDF to markdown
    converter = DocumentConverter()
    result = converter.convert(temp_pdf_path)
    markdown_content = result.document.export_to_markdown()
    
    # Remove temp file
    os.remove(temp_pdf_path)
    
    return markdown_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=3000, reload=True)
