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
    docs_dir = 'Test_pdfs'
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
                    
                    # Use the new order-based approach
                    order = result.get("order", [])
                    
                    if order and len(order) > 0:
                        f.write("=== CONTENT IN DOCUMENT ORDER ===\n\n")
                        
                        # Keep copies of paragraphs and tables that we can pop from
                        paras_copy = paragraphs.copy()
                        tables_copy = tables.copy()
                        
                        for i, item_type in enumerate(order, 1):
                            if item_type == 'P' and paras_copy:
                                f.write(f"\n[P{i}] {paras_copy.pop(0)}\n")
                                f.write("-" * 40 + "\n")
                            elif item_type == 'T' and tables_copy:
                                f.write(f"\n[T{i}] {tables_copy.pop(0)}\n")
                                f.write("-" * 40 + "\n")
                    
                    # Fallback to legacy format if no order info
                    elif not order:
                        # Check if we have elements for backward compatibility
                        elements = result.get("elements", [])
                        if elements:
                            f.write("=== CONTENT IN DOCUMENT ORDER ===\n")
                            for i, element in enumerate(elements, 1):
                                if element["type"] == "paragraph":
                                    f.write(f"\n[P{i}] {element['content']}\n")
                                elif element["type"] == "table":
                                    f.write(f"\n[T{i}] {element['content']}\n")
                                f.write("-" * 40 + "\n")
                        else:
                            # Legacy output format as last resort
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
