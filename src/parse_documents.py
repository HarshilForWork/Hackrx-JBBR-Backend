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
    Returns separate lists with an order indicator to maintain content sequence.
    """
    try:
        import pdfplumber
        import fitz  # PyMuPDF
        import pandas as pd
        
        # Three lists approach:
        paragraphs = []  # Will store all paragraphs
        tables = []      # Will store all tables
        order = []       # Will store 'P' or 'T' to indicate the correct order
        elements = []    # For backward compatibility
        
        # Step 1: Use PDFplumber to extract tables with their positions
        table_positions = []  # Store (page_num, y_position, table_index)
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table in page_tables:
                    if table and len(table) > 1:  # Valid table
                        # Clean the table
                        df = pd.DataFrame(table)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        
                        if df.shape[0] >= 2 and df.shape[1] >= 2:
                            # Convert to markdown
                            table_str = df.to_markdown(index=False, tablefmt='pipe')
                            
                            # Get table position
                            table_y_pos = 0
                            if hasattr(page, 'bbox'):
                                table_y_pos = page.bbox[1]  # Y-coordinate
                            
                            # Store position for later ordering
                            table_positions.append((page_num, table_y_pos, len(tables)))
                            tables.append(table_str)
        
        # Step 2: Use PyMuPDF for text extraction with positions
        paragraph_positions = []  # Store (page_num, y_position, paragraph_index)
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with their positions
            blocks = page.get_text("blocks")
            for block in blocks:
                if isinstance(block, tuple) and len(block) >= 5:
                    # block format: (x0, y0, x1, y1, text, block_no, ...)
                    x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
                    
                    if text.strip() and len(text.strip()) > 20:  # Filter very short segments
                        paragraphs.append(text.strip())
                        paragraph_positions.append((page_num, y0, len(paragraphs)-1))
        
        # Step 3: Combine and sort all positions to determine the proper order
        all_positions = []
        
        # Add paragraph positions with 'P' indicator
        for page, y_pos, para_idx in paragraph_positions:
            all_positions.append((page, y_pos, 'P', para_idx))
            
        # Add table positions with 'T' indicator
        for page, y_pos, table_idx in table_positions:
            all_positions.append((page, y_pos, 'T', table_idx))
            
        # Sort by page and then by position on page
        all_positions.sort(key=lambda pos: (pos[0], pos[1]))
        
        # Create order list and elements list
        for _, _, elem_type, idx in all_positions:
            order.append(elem_type)
            
            # Also build elements list for backward compatibility
            if elem_type == 'P':
                elements.append({
                    "type": "paragraph",
                    "content": paragraphs[idx],
                })
            else:  # elem_type == 'T'
                elements.append({
                    "type": "table",
                    "content": tables[idx],
                })
        
        # If no blocks were found, fall back to traditional method
        if not paragraphs:
            all_text = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
            # Clean and split paragraphs
            if all_text:
                full_text = '\n\n'.join(all_text)
                paragraphs = clean_and_split_paragraphs(full_text)
                order = ['P'] * len(paragraphs)  # All paragraphs, no tables
                
                # Update elements for backward compatibility
                elements = [{"type": "paragraph", "content": p} for p in paragraphs]
        
        doc.close()
        
        return {
            "paragraphs": paragraphs,
            "tables": tables,
            "order": order,          # New field with order indicators
            "elements": elements,    # For backward compatibility
            "method": "PDFplumber + PyMuPDF Hybrid (Order Preserved)"
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