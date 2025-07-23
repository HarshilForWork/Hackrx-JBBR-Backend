from docling.document_converter import DocumentConverter

source = "Test_pdfs\\BAJHLIP23020V012223.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
markdown_content = result.document.export_to_markdown()

# Save to a markdown file
with open("converted_document.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("Markdown file has been created successfully.")