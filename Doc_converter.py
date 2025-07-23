import fitz  # PyMuPDF
import os
import re

def convert_pdf_to_markdown(pdf_path):
    """Convert a PDF file to markdown text using PyMuPDF"""
    markdown_content = []
    
    # Open the PDF
    doc = fitz.open(pdf_path)
    
    # Process each page
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        
        # Add page header
        markdown_content.append(f"## Page {page_num + 1}\n")
        
        # Process text
        # Remove excessive newlines and add proper markdown line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Process headings (lines in all caps or with numbers might be headings)
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Check if line looks like a heading (all caps, numbered, etc.)
            if line.isupper() or re.match(r'^\d+\.', line):
                lines[i] = f"### {line}"
        
        processed_text = '\n'.join(lines)
        markdown_content.append(processed_text + "\n\n")
        
        # Extract images (optional)
        # for img_index, img in enumerate(page.get_images(full=True)):
        #     xref = img[0]
        #     image = doc.extract_image(xref)
        #     pix = fitz.Pixmap(doc, xref)
        #     img_filename = f"image_page{page_num + 1}_{img_index}.png"
        #     pix.save(img_filename)
        #     markdown_content.append(f"![Image {img_index}]({img_filename})\n\n")
        
        # Try to extract tables using page structure
        # This is simplified and might not work for complex tables
        # For better table extraction, consider using specialized libraries
    
    # Close the document
    doc.close()
    
    return "".join(markdown_content)

# Path to the PDF
pdf_path = "Test_pdfs\\ICIHLIP22012V012223.pdf"

# Convert to markdown
markdown_content = convert_pdf_to_markdown(pdf_path)

# Save to a markdown file
with open("converted_document.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("Markdown file has been created successfully.")