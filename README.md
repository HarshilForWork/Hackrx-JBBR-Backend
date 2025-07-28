# 🏥 Advanced Insurance Document Q&A System

An intelligent document processing and query system that uses cutting-edge embedding models and re-ranking technology to answer questions about insurance policy documents.

## 🚀 Features

### **Advanced RAG Pipeline**
- **Two-Stage Retrieval**: Initial vector search (top 50) → Advanced re-ranking → Final results (top 3)
- **Semantic Chunking**: LangChain-based chunking with Llama Text Embed v2 for better context preservation
- **Context Expansion**: Automatically expands context around relevant chunks for richer answers
- **Smart Document Management**: Tracks indexed documents to avoid re-processing

### **Intelligent Query Processing**
- **Entity Extraction**: Automatically identifies age, gender, procedures, locations, policy details
- **Reasoning Engine**: Provides detailed justification and relevant policy clauses
- **Confidence Scoring**: Shows system confidence in the answers
- **Concise Answers**: Direct yes/no responses with brief explanations

### **Cutting-Edge Tech Stack**
- **Vector Database**: Pinecone for scalable similarity search
- **LLM Integration**: Google Gemini for intelligent analysis
- **Embeddings**: **Llama Text Embed v2** (NVIDIA Hosted) - 2,048 token context, dense vectors
- **Re-ranking**: **BGE Reranker v2 M3** (BAAI Hosted) - 1,024 token context, advanced semantic scoring
- **Web Interface**: Streamlit for user-friendly interaction
- **Fallbacks**: Sentence-Transformers and CrossEncoder for offline operation

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/HarshilForWork/Hackrx-JBBR-Backend.git
cd Hackrx-JBBR-Backend
```

2. **Create a virtual environment (Python 3.10.11):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.streamlit/secrets.toml` file:
```toml
PINECONE_API_KEY = "your-pinecone-api-key"
GEMINI_API_KEY = "your-gemini-api-key"
```

## 🎯 Usage

### **Quick Start**
1. **Place PDF documents** in the `docs/` folder
2. **Run the application:**
   ```bash
   streamlit run app.py
   ```
3. **Open browser** at `http://localhost:8501`

### **Step-by-Step Process**
1. **📄 Parse Documents**: Click to extract text and tables from PDFs
2. **🚀 Index Documents**: Create vector embeddings using Llama Text Embed v2
3. **🔍 Query**: Ask questions and get re-ranked results using BGE Reranker v2 M3

### **Sample Queries**
```
46M, knee surgery, Pune, 3-month policy
30F, heart surgery, Mumbai, 6-month policy
25M, eye surgery, Delhi, 1-year policy
```

### **Expected Responses**
```
✅ Yes, knee surgery is covered under the policy.
ℹ️ Coverage depends on policy terms and waiting period.
```

## 🏗️ Architecture

```
📁 Project Structure
├── app.py                    # Main Streamlit application
├── src/
│   ├── parse_documents.py    # PDF parsing with PDFplumber + PyMuPDF
│   ├── chunk_documents.py    # LangChain semantic chunking with Llama embeddings
│   ├── embed_and_index.py    # Llama Text Embed v2 + Pinecone indexing
│   ├── query_processor.py    # Advanced RAG with BGE Reranker v2 M3
│   └── document_registry.py  # Smart document management
├── docs/                     # PDF documents to process
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Technical Details

### **Llama Text Embed v2 (NVIDIA Hosted)**
- **Context Length**: 2,048 tokens (4x larger than MiniLM)
- **Vector Type**: Dense embeddings
- **Starter Limits**: 5M tokens for development
- **Performance**: Superior semantic understanding

### **BGE Reranker v2 M3 (BAAI Hosted)**
- **Context Length**: 1,024 tokens
- **Purpose**: Advanced semantic re-ranking beyond vector similarity
- **Starter Limits**: 500 requests for development
- **Advantage**: Multi-language and cross-lingual capabilities

### **Two-Stage Retrieval Process**
1. **Initial Retrieval**: Llama Text Embed v2 vector similarity (top 50 candidates)
2. **Advanced Re-ranking**: BGE Reranker v2 M3 semantic scoring
3. **Context Expansion**: Retrieve adjacent chunks for richer context

### **Semantic Chunking**
- Uses LangChain's `SemanticChunker` with Llama-compatible embeddings
- Breakpoint threshold: 95th percentile for optimal chunk boundaries
- Fallback to `RecursiveCharacterTextSplitter` for reliability

## 🚀 Advanced Features

### **Entity Extraction**
Automatically identifies:
- Age and Gender
- Medical procedures
- Locations
- Policy duration
- Claim amounts

### **Intelligent Analysis**
- **Reasoning**: Step-by-step justification powered by Gemini
- **Policy Clauses**: Relevant sections referenced with high accuracy
- **Confidence Scoring**: System certainty percentage
- **Contextual Answers**: Expanded information using advanced embeddings

## 📊 Performance

- **Embedding Quality**: Llama Text Embed v2 provides superior semantic understanding
- **Re-ranking Accuracy**: BGE Reranker v2 M3 significantly improves relevance
- **Context Length**: 2,048 tokens for embeddings, 1,024 for re-ranking
- **Search Speed**: Sub-second query response with advanced models
- **Scalability**: Pinecone handles millions of vectors

## 🔄 Fallback Strategy

The system includes robust fallbacks:
- **Embeddings**: Falls back to `sentence-transformers/all-MiniLM-L6-v2` if API unavailable
- **Re-ranking**: Falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2` if BGE unavailable
- **LLM Processing**: Falls back to rule-based analysis if Gemini unavailable

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is part of the HackRx hackathon submission.

## 🙏 Acknowledgments

- **NVIDIA**: Llama Text Embed v2 hosting and API access
- **BAAI**: BGE Reranker v2 M3 development and hosting
- **Harshit's Suggestion**: Advanced RAG implementation with cutting-edge models
- **HackRx Team**: JBBR Backend development
- **LangChain**: Semantic chunking capabilities
- **Pinecone**: Vector database infrastructure
