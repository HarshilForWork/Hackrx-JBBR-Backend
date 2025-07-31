# Insurance Policy RAG System

A complete **Retrieval-Augmented Generation (RAG)** system for insurance policy analysis. The system processes PDF documents, creates semantic chunks, generates embeddings, indexes them for efficient retrieval, and provides an AI-powered query interface to answer questions about insurance policies.

## Features

### 📄 Document Processing Pipeline
- **Single-click pipeline**: Parse → Chunk → Embed → Index
- **CPU-only processing** with no GPU requirements
- **Optimized text chunking** with paragraph and sentence awareness
- **NVIDIA NV-Embed-QA embeddings** for semantic understanding
- **Pinecone vector database** for scalable indexing
- **Document registry** to avoid reprocessing unchanged files

### 🔍 AI-Powered Query System
- **Natural language queries** about insurance policies
- **Semantic search** with similarity scoring
- **LLM-powered analysis** using Google Gemini
- **Coverage decisions**: Covered/Not Covered/Partial/Unclear
- **Source citations** with confidence scores
- **Fallback mode** when LLM quota is exceeded

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

4. Set up API keys in `.streamlit/secrets.toml`:
```toml
# Required for vector search
PINECONE_API_KEY = "your-pinecone-api-key"

# Optional for LLM analysis (uses fallback if not provided)
GEMINI_API_KEY = "your-gemini-api-key"

# Already configured (NVIDIA embeddings)
NVIDIA_API_KEY = "your-nvidia-api-key"
```

### Getting API Keys
- **Pinecone**: Sign up at [pinecone.io](https://www.pinecone.io/) (free tier available)
- **Gemini**: Get free API key at [Google AI Studio](https://aistudio.google.com/)
- **NVIDIA**: Get API key at [NVIDIA Developer](https://build.nvidia.com/)

## Usage

### Quick Start
1. Place PDF documents in the `docs/` folder
2. Run the application:
```bash
streamlit run app.py
```
3. Open browser at `http://localhost:8501`

### Complete Workflow
1. **Process Documents** (Tab 2):
   - Click "Process All Documents" to run the complete pipeline
   - Wait for parsing, chunking, embedding, and indexing to complete

2. **Query Documents** (Tab 1):
   - Ask natural language questions about your policies
   - Get AI-powered answers with source citations
   - Examples: "What is covered under accidental death benefit?"

### Testing
Run the test script to verify your setup:
```bash
python test_rag_system.py
```

## Architecture

```
Project Structure:
├── app.py                          # Streamlit RAG interface
├── src/
│   ├── pipeline.py                 # Main processing + query pipeline
│   ├── parse_documents.py          # PDF parsing with PyMuPDF  
│   ├── chunk_documents_optimized.py # Fast text chunking
│   ├── embed_and_index.py          # NVIDIA embeddings + Pinecone indexing
│   ├── query_processor.py          # Complete RAG query system
│   └── document_registry.py        # Document state management
├── docs/                           # PDF documents to process
├── results/                        # Processing outputs and parsed data
└── .streamlit/secrets.toml         # API keys configuration
```

## Technical Details

### Document Processing Pipeline
1. **Document Parsing**: Extracts text from PDFs using PyMuPDF with page structure preservation
2. **Text Chunking**: Creates semantic chunks with paragraph/sentence awareness (800 chars, 150 overlap)
3. **Embedding Generation**: Uses NVIDIA NV-Embed-QA for high-quality 4096-dimensional vectors
4. **Vector Indexing**: Stores embeddings in Pinecone for millisecond-scale retrieval

### RAG Query System
1. **Query Encoding**: Converts user questions to embeddings using same NVIDIA model
2. **Semantic Search**: Retrieves top-K most relevant document chunks from Pinecone
3. **Entity Extraction**: Identifies key terms, amounts, and concepts using LLM or rules
4. **LLM Analysis**: Google Gemini analyzes chunks to determine coverage decisions
5. **Response Generation**: Provides structured answers with confidence scores and citations

### Chunking Strategy
- **Paragraph-based**: Maintains semantic coherence for policy clauses
- **Sentence-based**: Fallback for complex layouts  
- **Character-based**: Handles edge cases and tables
- Configurable chunk size (default: 800 characters)
- Overlap between chunks (default: 150 characters)

### Advanced Features
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Confidence Scoring**: Provides reliability metrics for each answer
- **Fallback Mode**: Works without LLM when API quota exceeded
- **Source Citation**: Links answers back to specific document sections
- **Entity Recognition**: Extracts amounts, dates, and policy terms

### Performance Features
- CPU-only processing for reliability
- Document registry prevents reprocessing
- Batch processing for multiple documents
- Optimized chunk creation with minimal memory usage
- Sub-second query response times

## Sample Queries

The system can answer various types of insurance policy questions:

### Coverage Questions
- "What is covered under accidental death benefit?"
- "Are pre-existing conditions covered?"
- "What is the waiting period for maternity benefits?"
- "Is mental health treatment covered?"

### Claims Questions  
- "How do I file a claim for medical expenses?"
- "What documents are needed for death claim?"
- "What is the claim settlement timeline?"
- "Can I submit claims online?"

### Policy Questions
- "What is the premium payment term?"
- "Can I surrender my policy early?"
- "What are the tax benefits available?"
- "How do I change my nominee?"

## Configuration

Default configuration in `PipelineConfig`:
- Chunk size: 800 characters (optimal for insurance policies)
- Chunk overlap: 150 characters (preserves context)
- Index name: "policy-index"
- Embedding model: NVIDIA NV-Embed-QA (4096 dimensions)

## Requirements

- Python 3.10+
- Pinecone API key (free tier available)
- Google Gemini API key (optional, for enhanced analysis)  
- NVIDIA API key (for embeddings)
- PDF documents in `docs/` folder

## Dependencies

Core libraries:
- streamlit: Web interface and user interaction
- pinecone-client: Vector database operations
- google-generativeai: LLM analysis with Gemini
- PyMuPDF: PDF parsing and text extraction
- numpy: Numerical operations for embeddings
- requests: API calls to NVIDIA embeddings

## Troubleshooting

### Common Issues

**"No PDF files found"**
- Ensure PDFs are in the `docs/` folder
- Check file extensions are `.pdf` (case-insensitive)

**"Pinecone API error"**  
- Verify API key in `.streamlit/secrets.toml`
- Check Pinecone project and environment settings
- Ensure index name matches (default: "policy-index")

**"Gemini quota exceeded"**
- System automatically falls back to rule-based analysis
- Wait for quota reset (usually 24 hours)
- Consider upgrading API plan for higher limits

**"No search results"**
- Ensure documents are processed first (Tab 2)
- Check if embeddings were created successfully
- Try rephrasing your question

### Performance Tips

- Process documents once, query many times
- Use specific questions for better results
- Include relevant keywords in queries
- Check confidence scores for answer reliability

## License

This project is part of the HackRx hackathon submission by Team JBBR.

## Contributors

- **Backend & RAG System**: Akshat, Harshil
- **AI Integration**: Bhavy, Jay
- **Data Processing**: Team JBBR
