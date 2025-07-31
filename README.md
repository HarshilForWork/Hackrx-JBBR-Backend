# Document Processing Pipeline

An optimized document processing system that parses PDF documents, creates semantic chunks, generates embeddings, and indexes them for efficient retrieval. The system uses CPU-only processing for fast and reliable document handling.

## Features

- Complete single-click pipeline: Parse → Chunk → Embed → Index
- CPU-only processing with no GPU requirements
- Optimized text chunking with paragraph and sentence awareness
- SentenceTransformer embeddings for semantic understanding
- Pinecone vector database for scalable indexing
- Document registry to avoid reprocessing unchanged files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/HarshilForWork/Hackrx-JBBR-Backend.git
cd Hackrx-JBBR-Backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.streamlit/secrets.toml` file:
```toml
PINECONE_API_KEY = "your-pinecone-api-key"
```

## Usage

1. Place PDF documents in the `docs/` folder
2. Run the application:
```bash
streamlit run app.py
```
3. Open browser at `http://localhost:8501`
4. Click "Process All Documents" to run the complete pipeline

## Architecture

```
Project Structure:
├── app.py                          # Streamlit interface
├── src/
│   ├── pipeline.py                 # Main processing pipeline
│   ├── parse_documents.py          # PDF parsing with PyMuPDF
│   ├── chunk_documents_optimized.py # Fast text chunking
│   ├── embed_and_index.py          # Embedding and Pinecone indexing
│   └── document_registry.py        # Document state management
├── docs/                           # PDF documents
└── results/                        # Processing outputs
```

## Technical Details

### Processing Pipeline
1. **Document Parsing**: Extracts text from PDFs using PyMuPDF
2. **Text Chunking**: Creates overlapping chunks with paragraph/sentence awareness
3. **Embedding Generation**: Uses SentenceTransformer (all-MiniLM-L6-v2) for semantic vectors
4. **Vector Indexing**: Stores embeddings in Pinecone for efficient retrieval

### Chunking Strategy
- Paragraph-based chunking when possible
- Sentence-based chunking as fallback
- Character-based chunking for edge cases
- Configurable chunk size (default: 800 characters)
- Overlap between chunks (default: 150 characters)

### Performance Features
- CPU-only processing for reliability
- Document registry prevents reprocessing
- Batch processing for multiple documents
- Optimized chunk creation with minimal memory usage

## Configuration

Default configuration in `PipelineConfig`:
- Chunk size: 800 characters
- Chunk overlap: 150 characters
- Index name: "policy-index"
- Embedding model: all-MiniLM-L6-v2

## Requirements

- Python 3.10+
- Pinecone API key
- PDF documents in `docs/` folder

## Dependencies

Core libraries:
- streamlit: Web interface
- sentence-transformers: Embedding generation
- pinecone: Vector database
- PyMuPDF: PDF processing
- scikit-learn: Text processing utilities

## License

This project is part of the HackRx hackathon submission.
