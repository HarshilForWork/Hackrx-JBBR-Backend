# PDF Parsing Pipeline

A simple and efficient PDF parsing pipeline using PDFplumber and PyMuPDF hybrid approach for extracting text and tables from PDF documents.

## Features

- **Hybrid Parsing**: Combines PDFplumber for table extraction and PyMuPDF for text extraction
- **Clean Output**: Extracts structured paragraphs and tables in markdown format
- **Simple Architecture**: Minimal dependencies and straightforward implementation
- **Batch Processing**: Process multiple PDF documents at once

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd BajajHackerX
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.parse_documents import parse_document_hybrid

# Parse a single PDF
result = parse_document_hybrid("path/to/your/document.pdf")

print(f"Extracted {len(result['paragraphs'])} paragraphs")
print(f"Extracted {len(result['tables'])} tables")
```

### Batch Processing

```python
from src.parse_documents import load_and_parse_documents

# Parse multiple PDFs
pdf_paths = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = load_and_parse_documents(pdf_paths)

for doc in results:
    print(f"Document: {doc['document_name']}")
    parsed = doc['parsed_output']
    if 'error' not in parsed:
        print(f"  Paragraphs: {len(parsed['paragraphs'])}")
        print(f"  Tables: {len(parsed['tables'])}")
```

### Testing

Run the test script to verify everything works:

```bash
python test_hybrid_simple.py
```

## Project Structure

```
📁 BajajHackerX/
├── 📁 docs/                    # PDF documents to parse
├── 📁 output/                  # Parsed output files
├── 📁 src/                     # Source code
│   ├── parse_documents.py      # Main hybrid parser
│   ├── chunk_documents.py      # Document chunking
│   ├── embed_and_index.py      # Embedding and indexing
│   ├── parse_query.py          # Query parsing
│   └── __init__.py            # Package initialization
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── test_hybrid_simple.py       # Test script
```

## Dependencies

- **pdfplumber**: Table extraction from PDFs
- **PyMuPDF**: Text extraction from PDFs
- **pandas**: Data processing and table formatting

## Output Format

The parser returns a dictionary with:

- `paragraphs`: List of extracted text paragraphs
- `tables`: List of tables in markdown format
- `method`: Parsing method used

Example output:
```python
{
    "paragraphs": ["First paragraph...", "Second paragraph..."],
    "tables": ["| Col1 | Col2 |\n|------|------|\n| data | data |"],
    "method": "PDFplumber + PyMuPDF Hybrid"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## License

This project is licensed under the MIT License.
