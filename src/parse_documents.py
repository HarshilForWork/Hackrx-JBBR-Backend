"""
Module: parse_documents.py
Functionality: Hybrid PDF parsing using PDFplumber + PyMuPDF.
"""
import os
import re
import time
from typing import List, Dict, Optional
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd

def parse_document_hybrid(pdf_path: str, save_parsed_text: bool = False) -> dict:
    """
    Hybrid PDF parsing: Maintains EXACT order of content as it appears in the document.
    Simple approach: Extract all content in reading order, then identify tables.
    """
    try:
        ordered_content = []  # Will maintain exact order: text, tables, text, etc.
        
        # Step 1: Extract all content with PyMuPDF in reading order
        doc = fitz.open(pdf_path)
        
        # Also get tables from PDFplumber for better formatting
        table_texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table)
                        df = df.dropna(how='all').dropna(axis=1, how='all')
                        if df.shape[0] >= 2 and df.shape[1] >= 2:
                            table_str = df.to_markdown(index=False, tablefmt='pipe')
                            table_texts.append(table_str)
        
        # Step 2: Process each page and extract content in order
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get all text blocks with positions
            blocks = page.get_text("dict")  # Get detailed text with positions
            
            # Extract blocks in reading order (top to bottom, left to right)
            text_blocks = []
            if "blocks" in blocks:
                for block in blocks["blocks"]:
                    if "lines" in block:
                        block_text = ""
                        block_bbox = block.get("bbox", [0, 0, 0, 0])
                        
                        for line in block["lines"]:
                            if "spans" in line:
                                line_text = ""
                                for span in line["spans"]:
                                    line_text += span.get("text", "")
                                if line_text.strip():
                                    block_text += line_text + "\n"
                        
                        if block_text.strip() and len(block_text.strip()) > 20:
                            text_blocks.append({
                                'content': block_text.strip(),
                                'bbox': block_bbox,
                                'y_pos': block_bbox[1],  # Top y-coordinate
                                'page': page_num + 1,
                                'type': 'text'
                            })
            
            # Sort blocks by position (top to bottom)
            text_blocks.sort(key=lambda x: x['y_pos'])
            
            # Add blocks to ordered content
            for block in text_blocks:
                ordered_content.append(block)
        
        doc.close()
        
        # Step 3: Insert tables in approximate positions
        # This is a simplified approach - we'll interleave tables
        if table_texts:
            # Distribute tables throughout the content
            text_items = [item for item in ordered_content if item['type'] == 'text']
            table_insertion_points = []
            
            if text_items:
                items_per_table = len(text_items) // len(table_texts) if table_texts else 1
                for i, table_text in enumerate(table_texts):
                    insert_pos = min((i + 1) * items_per_table, len(text_items))
                    table_insertion_points.append({
                        'position': insert_pos,
                        'content': f"[TABLE]\n{table_text}\n[/TABLE]",
                        'type': 'table',
                        'page': text_items[min(insert_pos, len(text_items)-1)]['page'] if text_items else 1
                    })
            
            # Insert tables at calculated positions
            for table_item in reversed(table_insertion_points):  # Reverse to maintain positions
                ordered_content.insert(table_item['position'], table_item)
        
        # Step 4: Extract content in final order
        ordered_text_content = []
        for item in ordered_content:
            ordered_text_content.append(item['content'])
        
        # Separate for backwards compatibility
        paragraphs = [item['content'] for item in ordered_content if item['type'] == 'text']
        tables = [item['content'] for item in ordered_content if item['type'] == 'table']
        
        # Optional: Save parsed text in CORRECT ORDER
        if save_parsed_text:
            parsed_content = save_parsed_text_file_ordered(pdf_path, ordered_content)
        
        print(f"✅ Parsed {len(ordered_content)} items in order: {len(paragraphs)} text blocks, {len(tables)} tables")
        
        return {
            "paragraphs": paragraphs,  # For backwards compatibility
            "tables": tables,  # For backwards compatibility
            "ordered_content": ordered_text_content,  # NEW: Content in exact order
            "method": "Hybrid PDFplumber + PyMuPDF (Order Preserved)",
            "parsed_file": parsed_content if save_parsed_text else None,
            "total_items": len(ordered_content)
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

def save_parsed_text_file(pdf_path: str, paragraphs: List[str], tables: List[str]) -> str:
    """
    Save parsed content to a text file for inspection.
    
    Args:
        pdf_path: Path to the original PDF
        paragraphs: List of extracted paragraphs
        tables: List of extracted tables
        
    Returns:
        Path to the saved text file
    """
    try:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(pdf_path), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{base_name}_parsed.txt")
        
        # Create content
        content_lines = [
            f"PARSED CONTENT FROM: {os.path.basename(pdf_path)}",
            "=" * 80,
            f"Extraction Method: Hybrid PDFplumber + PyMuPDF",
            f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Paragraphs: {len(paragraphs)}",
            f"Total Tables: {len(tables)}",
            "=" * 80,
            "",
            "PARAGRAPHS:",
            "-" * 40,
        ]
        
        # Add paragraphs
        for i, para in enumerate(paragraphs, 1):
            content_lines.extend([
                f"[PARAGRAPH {i}]",
                para,
                ""
            ])
        
        # Add tables
        if tables:
            content_lines.extend([
                "",
                "TABLES:",
                "-" * 40,
            ])
            
            for i, table in enumerate(tables, 1):
                content_lines.extend([
                    f"[TABLE {i}]",
                    table,
                    ""
                ])
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        print(f"📄 Parsed content saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Could not save parsed text file: {e}")
        return ""

def save_parsed_text_file_ordered(pdf_path: str, ordered_content: List[Dict]) -> str:
    """
    Save ordered parsed content to a text file for inspection.
    
    Args:
        pdf_path: Path to the original PDF
        ordered_content: List of ordered content items with type and page info
        
    Returns:
        Path to the saved text file
    """
    try:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(pdf_path), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{base_name}_parsed_ORDERED.txt")
        
        # Count content types
        text_count = sum(1 for item in ordered_content if item.get('type') == 'text')
        table_count = sum(1 for item in ordered_content if item.get('type') == 'table')
        
        # Create content
        content_lines = [
            f"PARSED CONTENT FROM: {os.path.basename(pdf_path)} (EXACT ORDER PRESERVED)",
            "=" * 80,
            f"Extraction Method: Hybrid PDFplumber + PyMuPDF (Order Preserved)",
            f"Extraction Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Items: {len(ordered_content)}",
            f"Text Items: {text_count}",
            f"Table Items: {table_count}",
            "=" * 80,
            "",
            "CONTENT IN EXACT ORDER OF APPEARANCE:",
            "-" * 50,
        ]
        
        # Add content in exact order
        for i, item in enumerate(ordered_content, 1):
            content_type = item.get('type', 'unknown').upper()
            page_num = item.get('page', '?')
            content = item.get('content', '')
            
            content_lines.extend([
                f"[ITEM {i}] {content_type} (Page {page_num})",
                content,
                ""
            ])
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        print(f"📄 Ordered parsed content saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Could not save ordered parsed text file: {e}")
        return ""

def load_and_parse_documents(document_paths: List[str], save_parsed_text: bool = False) -> List[Dict]:
    """
    Parse multiple PDF documents using the hybrid approach.
    
    Args:
        document_paths: List of paths to PDF documents
        save_parsed_text: Whether to save parsed content to text files for inspection
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
        
        parsed_output = parse_document_hybrid(path, save_parsed_text=save_parsed_text)
        parsed_docs.append({
            'document_name': doc_name, 
            'parsed_output': parsed_output
        })
    
    return parsed_docs

def load_and_parse_from_folder(docs_folder: str, file_filter: Optional[List[str]] = None, save_parsed_text: bool = False) -> List[Dict]:
    """
    Load and parse documents from a folder, optionally filtering by filenames.
    
    Args:
        docs_folder: Folder containing PDF documents
        file_filter: Optional list of filenames to process. If None, processes all PDFs.
        save_parsed_text: Whether to save parsed content to text files for inspection
    """
    if not os.path.exists(docs_folder):
        return []
    
    # Get all PDF files in folder
    all_files = [f for f in os.listdir(docs_folder) if f.lower().endswith('.pdf')]
    
    # Apply filter if provided
    if file_filter:
        files_to_process = [f for f in all_files if f in file_filter]
    else:
        files_to_process = all_files
    
    # Create full paths
    document_paths = [os.path.join(docs_folder, filename) for filename in files_to_process]
    
    return load_and_parse_documents(document_paths, save_parsed_text=save_parsed_text)