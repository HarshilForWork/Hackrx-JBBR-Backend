#!/usr/bin/env python3
"""
Simple test for the PDFplumber + PyMuPDF hybrid parsing approach.
"""
import os
from src.parse_documents import parse_document_hybrid, load_and_parse_documents

def test_hybrid_parsing():
    """Test the hybrid parsing approach on available PDFs."""
    print("Testing PDFplumber + PyMuPDF Hybrid Parsing")
    print("=" * 50)
    
    # Find PDF files
    docs_dir = 'docs'
    if not os.path.exists(docs_dir):
        print(f"Directory '{docs_dir}' not found!")
        return
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{docs_dir}' directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"{i}. {pdf}")
    
    # Test parsing
    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_dir, pdf_file)
        print(f"\n--- PARSING: {pdf_file} ---")
        
        try:
            result = parse_document_hybrid(pdf_path)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                paragraphs = result.get("paragraphs", [])
                tables = result.get("tables", [])
                method = result.get("method", "Unknown")
                
                print(f"✅ Success! Method: {method}")
                print(f"📄 Extracted {len(paragraphs)} paragraphs and {len(tables)} tables")
                
                # Show first paragraph
                if paragraphs:
                    print("\nFirst paragraph:")
                    print("-" * 30)
                    print(paragraphs[0][:200] + "..." if len(paragraphs[0]) > 200 else paragraphs[0])
                
                # Show first table
                if tables:
                    print("\nFirst table:")
                    print("-" * 30)
                    table_preview = tables[0][:300] + "..." if len(tables[0]) > 300 else tables[0]
                    print(table_preview)
                
                # Save output
                output_dir = 'output'
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{pdf_file}_hybrid_parsed.txt")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Method: {method}\n")
                    f.write(f"Paragraphs: {len(paragraphs)}\n")
                    f.write(f"Tables: {len(tables)}\n\n")
                    f.write("=== PARAGRAPHS ===\n")
                    for i, para in enumerate(paragraphs, 1):
                        f.write(f"\n[{i}] {para}\n")
                    f.write("\n=== TABLES ===\n")
                    for i, table in enumerate(tables, 1):
                        f.write(f"\n[{i}] {table}\n")
                
                print(f"💾 Saved to: {output_file}")
        
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    test_hybrid_parsing()
