"""
Module: parse_documents.py
Functionality: Simple hybrid PDF parsing using PDFplumber + PyMuPDF.
"""
import os
import re
from typing import List, Dict
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd

def parse_document_hybrid(pdf_path: str) -> dict:
    """
    Hybrid PDF parsing: PDFplumber for tables, PyMuPDF for text extraction.
    """
    try:
        import pdfplumber
        import fitz  # PyMuPDF
        import pandas as pd
        
        all_text = []
        all_tables = []
        
        # Use PDFplumber for table extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables using pdfplumber
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:  # Valid table
                        # Clean the table
                        df = pd.DataFrame(table)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if df.shape[0] >= 2 and df.shape[1] >= 2:
                            # Convert to markdown
                            table_str = df.to_markdown(index=False, tablefmt='pipe')
                            all_tables.append(table_str)
        
        # Use PyMuPDF for text extraction
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                all_text.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        
        # Clean and split paragraphs
        full_text = '\n\n'.join(all_text)
        paragraphs = clean_and_split_paragraphs(full_text)
        
        return {
            "paragraphs": paragraphs,
            "tables": all_tables,
            "method": "PDFplumber + PyMuPDF Hybrid"
        }
        
    except Exception as e:
        return {"error": f"Error parsing {pdf_path}: {str(e)}"}

def clean_and_split_paragraphs(text: str) -> List[str]:
    """
    Clean text and split into meaningful paragraphs.
    """
    if not text or not text.strip():
        return []
    
    # Basic text cleaning
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Split into paragraphs
    paragraphs = []
    for para in text.split('\n\n'):
        para = para.strip()
        if len(para) > 20:  # Filter out very short segments
            # Remove page markers but keep content
            if para.startswith('--- Page'):
                lines = para.split('\n')
                if len(lines) > 1:
                    content = '\n'.join(lines[1:]).strip()
                    if content:
                        paragraphs.append(content)
            else:
                paragraphs.append(para)
    
    return paragraphs

def load_and_parse_documents(document_paths: List[str]) -> List[Dict]:
    """
    Parse multiple PDF documents using the hybrid approach.
    """
    parsed_docs = []
    
    for path in document_paths:
        doc_name = os.path.basename(path)
        
        if not os.path.exists(path):
            parsed_docs.append({
                'document_name': doc_name, 
                'parsed_output': {"error": f"File not found: {path}"}
            })
            continue
        
        parsed_output = parse_document_hybrid(path)
        parsed_docs.append({
            'document_name': doc_name, 
            'parsed_output': parsed_output
        })
    
    return parsed_docs