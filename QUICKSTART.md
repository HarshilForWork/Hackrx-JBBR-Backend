# 🚀 Quick Start Guide

## ⚡ Run the System

```bash
streamlit run app.py
```

## 🔑 Setup API Keys

Create `.streamlit/secrets.toml`:
```toml
PINECONE_API_KEY = "your-pinecone-key"
GEMINI_API_KEY = "your-gemini-key"
```

## 📋 3-Step Process

1. **Parse Documents** 📄 - Click "Parse Documents" in sidebar
2. **Index Documents** 📚 - Enter API keys, click "Index Documents"  
3. **Query System** 🔍 - Enter query, click "Process Query"

## 🎯 Sample Queries

```
46-year-old male, knee surgery in Pune, 3-month policy
30F, heart surgery, Mumbai, 1 year policy
Is dental treatment covered for 35-year-old in Bangalore?
```

## �️ Fallback Mode

System works without API keys using rule-based analysis:
- ✅ Document parsing
- ✅ Entity extraction
- ✅ Basic decisions
- ❌ Semantic search (needs Pinecone)
- ❌ Advanced LLM evaluation (needs Gemini)

## 🎉 You're Ready!

The system provides intelligent fallbacks and works reliably even with API limits.
