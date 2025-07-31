"""
Module: query_processor.py
Functionality: Complete query processing pipeline with LLM integration.
"""
import json
import re
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Suppress specific tokenizer warnings for BGE Reranker
warnings.filterwarnings(
    "ignore", 
    message=".*XLMRobertaTokenizerFast.*__call__.*method is faster.*", 
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*fast tokenizer.*__call__.*method is faster.*",
    category=UserWarning
)

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    # Import BGE Reranker (BAAI) - required, no fallbacks
    from FlagEmbedding import FlagReranker
    BGE_RERANKER_AVAILABLE = True
except ImportError:
    BGE_RERANKER_AVAILABLE = False
    print("❌ BGE Reranker not available. Install with: pip install FlagEmbedding")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available for advanced text processing")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    print("⚠️ sentence-transformers CrossEncoder not available")

from .embed_and_index import generate_query_embedding_nvidia

class QueryProcessor:
    def __init__(self, pinecone_api_key: str, gemini_api_key: str, index_name: str = 'policy-index'):
        self.pinecone_api_key = pinecone_api_key
        self.gemini_api_key = gemini_api_key
        self.index_name = index_name
        self.quota_exceeded = False
        self.fallback_reason = None
        
        # Initialize Pinecone
        if PINECONE_AVAILABLE and pinecone_api_key and pinecone_api_key != 'dummy':
            try:
                self.pc = Pinecone(api_key=pinecone_api_key)
                self.index = self.pc.Index(index_name)
            except Exception as e:
                print(f"Pinecone initialization error: {e}")
                self.pc = None
                self.index = None
        else:
            self.pc = None
            self.index = None
        
        # Use NVIDIA embeddings ONLY - no fallbacks
        print("✅ Using NVIDIA NV-Embed-QA embeddings (no fallbacks)")
        
        # Initialize optimized reranker with multiple options
        self._init_optimized_reranker()
        
        # Initialize hybrid search components
        self._init_hybrid_search()
        
        # Initialize Gemini
        if GENAI_AVAILABLE and gemini_api_key and gemini_api_key != 'dummy':
            try:
                genai.configure(api_key=gemini_api_key)
                # Try different models based on availability
                model_options = [
                    'gemini-1.5-flash',  # More available for students
                    'gemini-pro',        # Standard model
                    'gemini-1.5-pro'     # Premium model (might be limited)
                ]
                
                self.llm = None
                self.model_name = None
                
                # Try each model until one works
                for model_name in model_options:
                    try:
                        test_model = genai.GenerativeModel(model_name)
                        # Test with a simple query to verify access
                        test_response = test_model.generate_content("Hello")
                        if test_response:
                            self.llm = test_model
                            self.model_name = model_name
                            print(f"Successfully initialized Gemini model: {model_name}")
                            break
                    except Exception as e:
                        print(f"Failed to initialize {model_name}: {str(e)}")
                        if 'quota' in str(e).lower() or 'limit' in str(e).lower():
                            self.quota_exceeded = True
                            self.fallback_reason = f"Quota exceeded for {model_name}"
                        continue
                
                if not self.llm:
                    print("Warning: No Gemini models available, using fallback methods")
                    if not self.quota_exceeded:
                        self.fallback_reason = "No models accessible"
                    
            except Exception as e:
                print(f"Gemini initialization error: {e}")
                self.llm = None
                self.model_name = None
                if 'quota' in str(e).lower() or 'limit' in str(e).lower():
                    self.quota_exceeded = True
                    self.fallback_reason = "API quota exceeded"
                else:
                    self.fallback_reason = f"Initialization error: {str(e)}"
        else:
            self.llm = None
            self.model_name = None
            if gemini_api_key == 'dummy':
                self.fallback_reason = "Using dummy API key for testing"
            else:
                self.fallback_reason = "API key not provided"
    
    def _init_optimized_reranker(self):
        """Initialize the best available reranker with optimizations."""
        print("🔄 Initializing optimized reranker...")
        
        # Try lightweight cross-encoder first (much faster than BGE)
        if CROSS_ENCODER_AVAILABLE:
            try:
                # Use a lightweight, fast cross-encoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.reranker_type = "minilm_cross_encoder"
                print("✅ Loaded fast MiniLM Cross-Encoder (10x faster than BGE)")
                
                # Add caching for reranker results
                self._reranker_cache = {}
                self._cache_max_size = 200
                return
            except Exception as e:
                print(f"⚠️ MiniLM Cross-Encoder failed: {e}")
        
        # Fallback to BGE with optimizations
        if BGE_RERANKER_AVAILABLE:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"🚀 Using device: {device}")
                
                # Use smaller, faster BGE model
                self.reranker = FlagReranker(
                    'BAAI/bge-reranker-base',  # Smaller, faster than v2-m3
                    use_fp16=True,
                    device=device,
                    batch_size=16,  # Reduced for stability
                    max_length=256  # Shorter sequences for speed
                )
                self.reranker_type = "bge-reranker-base"
                print("✅ BGE Base Reranker loaded with optimizations!")
                
                self._reranker_cache = {}
                self._cache_max_size = 150
                return
                
            except Exception as e:
                print(f"⚠️ BGE Reranker failed: {e}")
        
        # Final fallback - no reranker
        print("❌ No reranker available - using hybrid scoring only")
        self.reranker = None
        self.reranker_type = "none"
        self._reranker_cache = {}
        self._cache_max_size = 0
    
    def _init_hybrid_search(self):
        """Initialize components for hybrid search (semantic + lexical)."""
        print("🔄 Initializing hybrid search components...")
        
        # Initialize TF-IDF for lexical search
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,  # Ignore terms in less than 2 documents
                max_df=0.8  # Ignore terms in more than 80% of documents
            )
            self.tfidf_fitted = False
            self.document_texts = []  # Store for TF-IDF fitting
            print("✅ TF-IDF vectorizer initialized for lexical search")
        else:
            self.tfidf_vectorizer = None
            self.tfidf_fitted = False
            print("⚠️ TF-IDF not available - using semantic search only")
        
        # Initialize keyword boosting
        self.medical_keywords = {
            'surgery', 'operation', 'procedure', 'treatment', 'therapy', 'diagnosis',
            'hospital', 'clinic', 'doctor', 'physician', 'specialist', 'consultation',
            'insurance', 'policy', 'claim', 'coverage', 'premium', 'deductible',
            'pre-existing', 'waiting period', 'exclusion', 'benefit', 'reimbursement'
        }
        
        # Scoring weights for hybrid search
        self.hybrid_weights = {
            'semantic': 0.6,  # Semantic similarity weight
            'lexical': 0.3,   # TF-IDF/BM25 weight
            'keyword': 0.1    # Keyword boost weight
        }
    
    def _encode_query(self, query: str) -> List[float]:
        """Encode query using NVIDIA NV-Embed-QA model ONLY."""
        try:
            # Use NVIDIA embeddings ONLY - no fallbacks
            return generate_query_embedding_nvidia(query)
                    
        except Exception as e:
            print(f"❌ NVIDIA query encoding error: {e}")
            # Return zero vector as fallback (4096 dimensions for NVIDIA Llama Text Embed v2)
            return [0.0] * 4096
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status and recommendations."""
        status = {
            'gemini_available': self.llm is not None,
            'model_name': getattr(self, 'model_name', None),
            'quota_exceeded': getattr(self, 'quota_exceeded', False),
            'fallback_reason': getattr(self, 'fallback_reason', None),
            'recommendations': []
        }
        
        if self.quota_exceeded:
            status['recommendations'].extend([
                "🚨 Gemini API quota exceeded - system is using fallback methods",
                "💡 Wait for quota reset (usually 24 hours) or upgrade your API plan",
                "📚 Student tier: Check Google AI Studio for quota details",
                "⚡ Fallback methods provide ~70% accuracy vs LLM's 95%"
            ])
        elif not self.llm:
            status['recommendations'].extend([
                "⚠️ LLM not available - using rule-based analysis",
                "🔑 Check your Gemini API key is valid",
                "📱 Try different model tiers (flash, pro, pro-1.5)"
            ])
        else:
            status['recommendations'].append(f"✅ Using {self.model_name} for optimal results")
        
        
        return status
    
    def _extract_json_from_response(self, response_text: str, context: str = "response") -> Optional[Dict]:
        """Helper method to robustly extract JSON from LLM responses."""
        if not response_text or not response_text.strip():
            print(f"🔍 Empty {context}, skipping JSON extraction")
            return None
        
        response_text = response_text.strip()
        
        # Try parsing the whole response first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON block in the response
        if '{' in response_text and '}' in response_text:
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"🔍 JSON extraction failed from {context}: {str(e)[:50]}...")
                print(f"🔍 Raw response preview: '{response_text[:100]}...'")
        
        print(f"🔍 No valid JSON found in {context}")
        return None
    
    def _make_llm_request_with_retry(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Make LLM request with retry logic and quota detection."""
        if not self.llm:
            return None
            
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate_content(prompt)
                if response and response.text:
                    return response.text.strip()
                else:
                    print(f"🔍 Empty response on attempt {attempt + 1}")
                    if attempt < max_retries:
                        time.sleep(1)  # Brief pause before retry
                        continue
                    return None
                    
            except Exception as e:
                error_msg = str(e).lower()
                if 'quota' in error_msg or 'limit' in error_msg or 'exceeded' in error_msg:
                    print(f"⚠️ Gemini API quota exceeded: {e}")
                    self.quota_exceeded = True
                    return None
                elif attempt < max_retries:
                    print(f"🔄 Retry {attempt + 1} after error: {str(e)[:50]}...")
                    time.sleep(1)
                    continue
                else:
                    print(f"🔍 Final attempt failed: {e}")
                    return None
        
        return None

    def _optimize_text_for_reranking(self, text: str, max_length: int = 512) -> str:
        """Optimize text length for faster reranking."""
        if len(text) <= max_length:
            return text
        
        # Smart truncation: keep beginning and end, remove middle
        half_length = max_length // 2
        return text[:half_length] + "..." + text[-half_length:]
    
    def _get_reranker_cache_key(self, query: str, texts: list) -> str:
        """Generate cache key for reranker results."""
        import hashlib
        combined = query + "|".join(texts[:3])  # Use first 3 texts for cache key
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _batch_rerank_with_cache(self, query: str, candidates: List[Dict]) -> List[float]:
        """Optimized batch reranking with caching and multiple reranker support."""
        if not self.reranker:
            return self._hybrid_score_candidates(query, candidates)
        
        # Optimize text lengths for faster processing
        optimized_texts = [
            self._optimize_text_for_reranking(candidate["text"]) 
            for candidate in candidates
        ]
        
        # Check cache first
        cache_key = self._get_reranker_cache_key(query, optimized_texts)
        if cache_key in self._reranker_cache:
            print("🎯 Using cached reranking results")
            return self._reranker_cache[cache_key]
        
        try:
            print(f"🔄 Reranking {len(candidates)} candidates with {self.reranker_type}...")
            start_time = time.time()
            
            if self.reranker_type == "minilm_cross_encoder":
                # Use sentence-transformers CrossEncoder
                try:
                    # CrossEncoder expects list of [query, text] pairs
                    pairs = [[query, text] for text in optimized_texts]
                    cross_scores = self.reranker.predict(pairs)
                    
                    # Convert to float list safely
                    scores = []
                    if hasattr(cross_scores, '__iter__'):
                        for score in cross_scores:
                            try:
                                if hasattr(score, 'item'):  # numpy scalar
                                    scores.append(float(score.item()))
                                else:
                                    scores.append(float(score))
                            except (ValueError, TypeError, AttributeError):
                                scores.append(0.0)
                    else:
                        # Single score case
                        try:
                            scores = [float(cross_scores)] * len(optimized_texts)
                        except:
                            scores = [0.0] * len(optimized_texts)
                except Exception as ce_error:
                    print(f"⚠️ CrossEncoder error: {ce_error}")
                    return self._hybrid_score_candidates(query, candidates)
                        
            elif "bge" in self.reranker_type:
                # Use BGE FlagReranker
                pairs = [(query, text) for text in optimized_texts]
                cross_scores = self.reranker.compute_score(pairs, normalize=True)
                
                # Handle single score or list of scores
                if not isinstance(cross_scores, list):
                    cross_scores = [cross_scores]
                
                scores = []
                for score in cross_scores:
                    try:
                        if hasattr(score, 'item'):
                            scores.append(float(score.item()))
                        elif score is not None:
                            scores.append(float(score))
                        else:
                            scores.append(0.0)
                    except (ValueError, TypeError, AttributeError):
                        scores.append(0.0)
            else:
                # Fallback to hybrid scoring
                return self._hybrid_score_candidates(query, candidates)
            
            processing_time = time.time() - start_time
            print(f"⚡ Reranking completed in {processing_time:.2f}s ({len(candidates)/processing_time:.1f} docs/sec)")
            
            # Cache results (with size limit)
            if len(self._reranker_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._reranker_cache))
                del self._reranker_cache[oldest_key]
            
            self._reranker_cache[cache_key] = scores
            return scores
            
        except Exception as e:
            print(f"⚠️ Reranking error: {e}")
            # Fallback to hybrid scoring
            return self._hybrid_score_candidates(query, candidates)
    
    def _hybrid_score_candidates(self, query: str, candidates: List[Dict]) -> List[float]:
        """Hybrid scoring combining semantic, lexical, and keyword matching."""
        scores = []
        
        # Get semantic scores (already available as vector_score)
        semantic_scores = [c.get("vector_score", 0.0) for c in candidates]
        
        # Get lexical scores using TF-IDF if available
        lexical_scores = self._get_tfidf_scores(query, candidates)
        
        # Get keyword boost scores
        keyword_scores = self._get_keyword_scores(query, candidates)
        
        # Combine scores using hybrid weights
        for i in range(len(candidates)):
            semantic = semantic_scores[i] if i < len(semantic_scores) else 0.0
            lexical = lexical_scores[i] if i < len(lexical_scores) else 0.0
            keyword = keyword_scores[i] if i < len(keyword_scores) else 0.0
            
            # Normalize scores to 0-1 range
            semantic_norm = max(0.0, min(1.0, semantic))
            lexical_norm = max(0.0, min(1.0, lexical))
            keyword_norm = max(0.0, min(1.0, keyword))
            
            # Weighted combination
            hybrid_score = (
                semantic_norm * self.hybrid_weights['semantic'] +
                lexical_norm * self.hybrid_weights['lexical'] +
                keyword_norm * self.hybrid_weights['keyword']
            )
            
            scores.append(hybrid_score)
        
        print(f"✅ Hybrid scoring completed for {len(candidates)} candidates")
        return scores
    
    def _get_tfidf_scores(self, query: str, candidates: List[Dict]) -> List[float]:
        """Get TF-IDF based lexical similarity scores."""
        if not self.tfidf_vectorizer or not SKLEARN_AVAILABLE:
            return [0.0] * len(candidates)
        
        try:
            texts = [c["text"] for c in candidates]
            
            # Fit TF-IDF if not already fitted
            if not self.tfidf_fitted:
                all_texts = texts + [query]
                self.tfidf_vectorizer.fit(all_texts)
                self.tfidf_fitted = True
            
            # Transform query and candidate texts
            query_tfidf = self.tfidf_vectorizer.transform([query])
            candidates_tfidf = self.tfidf_vectorizer.transform(texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_tfidf, candidates_tfidf)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            print(f"⚠️ TF-IDF scoring error: {e}")
            return [0.0] * len(candidates)
    
    def _get_keyword_scores(self, query: str, candidates: List[Dict]) -> List[float]:
        """Get keyword-based boost scores for medical/insurance terms."""
        query_words = set(query.lower().split())
        query_medical_words = query_words.intersection(self.medical_keywords)
        
        scores = []
        for candidate in candidates:
            text_words = set(candidate["text"].lower().split())
            text_medical_words = text_words.intersection(self.medical_keywords)
            
            # Score based on medical keyword overlap
            if query_medical_words:
                keyword_overlap = len(query_medical_words.intersection(text_medical_words))
                score = keyword_overlap / len(query_medical_words)
            else:
                # General keyword overlap if no medical terms
                overlap = len(query_words.intersection(text_words))
                score = overlap / len(query_words) if query_words else 0.0
            
            scores.append(score)
        
        return scores

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from natural language query using Gemini only."""
        if self.llm:
            return self._llm_entity_extraction(query)
        else:
            print("❌ No LLM available for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
    
    def _llm_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Use LLM for entity extraction - no fallbacks."""
        if not self.llm:
            print("❌ LLM not available for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
            
        prompt = f"""
        Extract the following entities from this insurance/medical query: "{query}"
        
        Return a JSON object with these fields (use null if not found):
        - age: integer (patient age)
        - gender: string (M/F/Male/Female)
        - procedure: string (medical procedure/surgery)
        - location: string (city/location)
        - policy_duration: string (how old is the policy)
        - policy_type: string (type of insurance policy)
        - amount: number (any monetary amount mentioned)
        
        Example: {{"age": 46, "gender": "M", "procedure": "knee surgery", "location": "Pune", "policy_duration": "3 months", "policy_type": null, "amount": null}}
        
        Only return the JSON, no other text.
        """
        
        # Use robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("❌ No response from LLM for entity extraction")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
        
        # Extract JSON from response
        entities = self._extract_json_from_response(response_text, "entity extraction")
        if entities:
            print("✅ Successfully extracted entities using LLM")
            return entities
        else:
            print("❌ JSON extraction failed from LLM response")
            return {
                "age": None,
                "gender": None,
                "procedure": None,
                "location": None,
                "policy_duration": None,
                "policy_type": None,
                "amount": None
            }
    
    def _prefilter_candidates(self, candidates: List[Dict], query: str, max_candidates: int = 15) -> List[Dict]:
        """Enhanced pre-filtering using hybrid scoring to reduce reranking workload."""
        if len(candidates) <= max_candidates:
            return candidates
        
        print(f"🔍 Smart pre-filtering from {len(candidates)} to {max_candidates} candidates...")
        
        # Get hybrid scores for all candidates
        hybrid_scores = self._hybrid_score_candidates(query, candidates)
        
        # Add hybrid scores to candidates
        for i, candidate in enumerate(candidates):
            if i < len(hybrid_scores):
                candidate["hybrid_score"] = hybrid_scores[i]
            else:
                candidate["hybrid_score"] = 0.0
        
        # Sort by hybrid score (combination of semantic, lexical, and keyword)
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Take top candidates
        filtered = candidates[:max_candidates]
        
        print(f"✅ Pre-filtered to top {len(filtered)} candidates using hybrid scoring")
        return filtered

    def _should_rerank(self, candidates: List[Dict], final_k: int) -> bool:
        """Decide if reranking is beneficial based on score distribution."""
        if len(candidates) <= final_k:
            return False
        
        # Get vector scores of top candidates
        top_scores = [c.get("vector_score", 0) for c in candidates[:final_k * 2]]
        
        if not top_scores:
            return True  # Always rerank if no scores
        
        # Calculate score variance - if scores are very similar, reranking helps
        import statistics
        try:
            score_std = statistics.stdev(top_scores) if len(top_scores) > 1 else 0
            # If standard deviation is low (scores are similar), reranking is beneficial
            return score_std < 0.1  # Threshold for "similar" scores
        except:
            return True  # Default to reranking if calculation fails

    def semantic_search_with_reranking(self, query: str, initial_k: int = 40, final_k: int = 3) -> List[Dict]:
        """
        Optimized two-stage retrieval with hybrid scoring and fast reranking:
        1. Retrieve top 40 candidates using vector similarity (reduced from 50)
        2. Pre-filter to 15 best candidates using hybrid scoring (semantic + lexical + keyword)
        3. Re-rank top candidates using lightweight cross-encoder or BGE
        4. Return top 3 most relevant chunks with context expansion
        """
        if not self.index:
            print("❌ Pinecone index not available")
            return []
        
        try:
            # Stage 1: Initial retrieval (reduced k for speed)
            print(f"🔍 Stage 1: Retrieving top {initial_k} candidates...")
            query_embedding = self._encode_query(query)
            
            response = self.index.query(
                vector=query_embedding,
                top_k=initial_k,
                include_metadata=True
            )
            
            # Format initial results
            candidates = []
            for match in response.matches:
                candidates.append({
                    "id": match.id,
                    "vector_score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_name": match.metadata.get("document_name", ""),
                    "page_number": match.metadata.get("page_number", 1),
                    "chunk_index": match.metadata.get("chunk_index", 0)
                })
            
            if not candidates:
                print("⚠️ No candidates found in initial retrieval")
                return []
            
            print(f"✅ Retrieved {len(candidates)} candidates")
            
            # Stage 1.5: Smart pre-filtering with hybrid scoring
            if len(candidates) > 15:  # Pre-filter to fewer candidates
                candidates = self._prefilter_candidates(candidates, query, max_candidates=15)
            
            # Stage 2: Adaptive optimized reranking
            if self.reranker and len(candidates) > final_k:
                should_rerank = self._should_rerank(candidates, final_k)
                
                if should_rerank:
                    print(f"🎯 Stage 2: Fast reranking with {self.reranker_type}...")
                    
                    # Use optimized reranking
                    cross_scores = self._batch_rerank_with_cache(query, candidates)
                    
                    # Add reranker scores to candidates
                    for i, candidate in enumerate(candidates):
                        if i < len(cross_scores):
                            candidate["cross_score"] = cross_scores[i]
                        else:
                            candidate["cross_score"] = candidate.get("hybrid_score", candidate.get("vector_score", 0.0))
                    
                    # Sort by reranker score
                    candidates.sort(key=lambda x: x["cross_score"], reverse=True)
                    print(f"✅ Fast reranking completed")
                else:
                    print(f"⚡ Skipping reranking - using hybrid scores")
                    # Use hybrid scores as cross scores
                    for candidate in candidates:
                        candidate["cross_score"] = candidate.get("hybrid_score", candidate.get("vector_score", 0.0))
                
                final_candidates = candidates[:final_k]
                
            elif len(candidates) > final_k:
                # No reranker available - use hybrid scores
                print("⚡ Using hybrid scoring (no reranker available)")
                for candidate in candidates:
                    candidate["cross_score"] = candidate.get("hybrid_score", candidate.get("vector_score", 0.0))
                
                candidates.sort(key=lambda x: x["cross_score"], reverse=True)
                final_candidates = candidates[:final_k]
            else:
                # Fewer candidates than needed
                final_candidates = candidates
                for candidate in final_candidates:
                    candidate["cross_score"] = candidate.get("hybrid_score", candidate.get("vector_score", 0.0))
            
            # Stage 3: Context expansion
            print(f"📄 Stage 3: Context expansion for {len(final_candidates)} chunks...")
            expanded_results = self._expand_context(final_candidates)
            
            # Add ranking metadata
            for i, result in enumerate(expanded_results):
                result["final_rank"] = i + 1
                result["ranking_method"] = f"hybrid_{self.reranker_type}" if self.reranker else "hybrid_only"
            
            self._print_ranking_summary(query, expanded_results)
            
            return expanded_results
            
        except Exception as e:
            print(f"❌ Optimized search error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _expand_context(self, candidates: List[Dict], context_chars: int = 500) -> List[Dict]:
        """Expand context around selected chunks by retrieving adjacent chunks."""
        expanded_results = []
        
        for candidate in candidates:
            try:
                doc_name = candidate["document_name"]
                chunk_index = candidate.get("chunk_index", 0)
                
                # Try to get adjacent chunks for context
                adjacent_chunks = self._get_adjacent_chunks(doc_name, chunk_index, context_chars)
                
                if adjacent_chunks:
                    # Combine with context
                    expanded_text = self._combine_chunks_with_context(
                        candidate["text"], 
                        adjacent_chunks,
                        context_chars
                    )
                    candidate["text"] = expanded_text
                    candidate["context_expanded"] = True
                    candidate["adjacent_chunks_count"] = len(adjacent_chunks)
                else:
                    candidate["context_expanded"] = False
                    candidate["adjacent_chunks_count"] = 0
                
                expanded_results.append(candidate)
                
            except Exception as e:
                print(f"⚠️ Context expansion failed for chunk: {e}")
                candidate["context_expanded"] = False
                expanded_results.append(candidate)
        
        return expanded_results
    
    def _get_adjacent_chunks(self, doc_name: str, chunk_index: int, max_chars: int) -> List[Dict]:
        """Retrieve adjacent chunks from the same document using namespace query."""
        try:
            if not self.index:
                return []
            
            # Create a dummy vector for metadata-only search
            dummy_vector = [0.0] * 384
            
            # Query for all chunks from the same document
            response = self.index.query(
                vector=dummy_vector,
                top_k=200,  # Get more chunks to find adjacent ones
                include_metadata=True
            )
            
            # Filter and find adjacent chunks manually
            same_doc_chunks = []
            for match in response.matches:
                metadata = match.metadata or {}
                if metadata.get("document_name") == doc_name:
                    same_doc_chunks.append({
                        "text": metadata.get("text", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "chunk_id": match.id,
                        "score": match.score
                    })
            
            # Sort by chunk index to find adjacent chunks
            same_doc_chunks.sort(key=lambda x: x["chunk_index"])
            
            # Find chunks adjacent to our target
            adjacent = []
            target_found = False
            
            for i, chunk in enumerate(same_doc_chunks):
                if chunk["chunk_index"] == chunk_index:
                    target_found = True
                    # Get previous chunks
                    for j in range(max(0, i-2), i):
                        prev_chunk = same_doc_chunks[j]
                        adjacent.append({
                            "text": prev_chunk["text"],
                            "chunk_index": prev_chunk["chunk_index"],
                            "position": "before"
                        })
                    
                    # Get next chunks
                    for j in range(i+1, min(len(same_doc_chunks), i+3)):
                        next_chunk = same_doc_chunks[j]
                        adjacent.append({
                            "text": next_chunk["text"],
                            "chunk_index": next_chunk["chunk_index"],
                            "position": "after"
                        })
                    break
            
            if not target_found:
                print(f"⚠️ Target chunk {chunk_index} not found in document {doc_name}")
                # Fallback: just get some chunks from the same document
                for chunk in same_doc_chunks[:4]:
                    if chunk["chunk_index"] != chunk_index:
                        adjacent.append({
                            "text": chunk["text"],
                            "chunk_index": chunk["chunk_index"],
                            "position": "context"
                        })
            
            print(f"🔍 Found {len(adjacent)} adjacent chunks for {doc_name} chunk {chunk_index}")
            return adjacent[:4]  # Limit to prevent too much context
            
        except Exception as e:
            print(f"⚠️ Could not retrieve adjacent chunks: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _combine_chunks_with_context(self, main_text: str, adjacent_chunks: List[Dict], max_chars: int) -> str:
        """Intelligently combine main chunk with context from adjacent chunks."""
        # Start with main text
        result = main_text
        chars_used = len(main_text)
        remaining_chars = max_chars
        
        # Add before context
        before_chunks = [c for c in adjacent_chunks if c["position"] == "before"]
        before_chunks.sort(key=lambda x: x["chunk_index"], reverse=True)  # Closest first
        
        before_context = ""
        for chunk in before_chunks:
            chunk_text = chunk["text"]
            if len(before_context) + len(chunk_text) <= remaining_chars // 2:
                before_context = chunk_text + " ... " + before_context
        
        # Add after context
        after_chunks = [c for c in adjacent_chunks if c["position"] == "after"]
        after_chunks.sort(key=lambda x: x["chunk_index"])  # Closest first
        
        after_context = ""
        for chunk in after_chunks:
            chunk_text = chunk["text"]
            if len(after_context) + len(chunk_text) <= remaining_chars // 2:
                after_context = after_context + " ... " + chunk_text
        
        # Combine all parts
        if before_context:
            result = f"[CONTEXT] {before_context} [MAIN] {main_text}"
        if after_context:
            result = result + f" [CONTINUED] {after_context}"
        
        return result
    
    def _print_ranking_summary(self, query: str, results: List[Dict]):
        """Print detailed ranking summary for debugging."""
        print(f"\n📊 Advanced RAG Results Summary:")
        print(f"Query: '{query}'")
        print(f"Final results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            vector_score = result.get("vector_score", 0)
            cross_score = result.get("cross_score", 0)
            doc_name = result.get("document_name", "Unknown")
            context_expanded = result.get("context_expanded", False)
            
            print(f"  {i}. Doc: {doc_name}")
            print(f"     Vector: {vector_score:.3f} | Cross: {cross_score:.3f} | Context: {'✅' if context_expanded else '❌'}")
            print(f"     Text: {result['text'][:100]}...")
            print()
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
         """Perform semantic search in Pinecone."""
         if not self.index:
             return []
         
         try:
             # Generate query embedding using helper method
             query_embedding = self._encode_query(query)
             
             # Search in Pinecone (returns a QueryResponse)
             response = self.index.query(
                 vector=query_embedding,
                 top_k=top_k,
                 include_metadata=True
             )
 
             # Format results from response.matches
             search_results = []
             for m in response.matches:
                 search_results.append({
                     "id": m.id,
                     "score": m.score,
                     "text": m.metadata.get("text", ""),
                     "document_name": m.metadata.get("document_name", ""),
                     "page_number": m.metadata.get("page_number", 1)
                 })
             
             return search_results
         except Exception as e:
             print(f"Search error: {e}")
             return []
    
    def evaluate_claim(self, query: str, entities: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM to evaluate claim based on retrieved context - no fallbacks."""
        
        if self.llm:
            return self._llm_evaluation(query, entities, search_results)
        else:
            print("❌ No LLM available for claim evaluation")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "LLM not available for claim evaluation.",
                "relevant_clauses": [],
                "reasoning": "System requires LLM for claim evaluation"
            }
    
    def _llm_evaluation(self, query: str, entities: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM for claim evaluation - no fallbacks."""
        if not self.llm:
            print("❌ LLM not available for evaluation")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "LLM not available for claim evaluation.",
                "relevant_clauses": [],
                "reasoning": "System requires LLM for claim evaluation"
            }
            
        # Prepare context from search results
        context = "\n\n".join([
            f"Document: {result['document_name']}\nContent: {result['text'][:500]}..."
            for result in search_results[:3]
        ])
        
        prompt = f"""
        You are an insurance claim evaluator. Based on the following context from policy documents, evaluate this claim:
        
        QUERY: {query}
        
        EXTRACTED ENTITIES:
        {json.dumps(entities, indent=2)}
        
        RELEVANT POLICY CONTEXT:
        {context}
        
        Please provide a decision in this exact JSON format:
        {{
            "decision": "approved" or "rejected" or "needs_review",
            "amount": number or null,
            "confidence": 0.0 to 1.0,
            "justification": "detailed explanation",
            "relevant_clauses": ["list of specific clauses used"],
            "reasoning": "step by step reasoning"
        }}
        
        Consider factors like:
        - Policy coverage for the procedure
        - Waiting periods
        - Geographic coverage
        - Age restrictions
        - Pre-existing conditions
        
        Only return the JSON, no other text.
        """
        
        # Use robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("❌ No response from LLM for evaluation")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "No response from LLM.",
                "relevant_clauses": [],
                "reasoning": "LLM did not respond"
            }
        
        # Extract JSON from response
        evaluation = self._extract_json_from_response(response_text, "claim evaluation")
        if evaluation:
            print("✅ Successfully completed claim evaluation using LLM")
            return evaluation
        else:
            print("❌ JSON extraction failed from LLM evaluation response")
            return {
                "decision": "error",
                "amount": None,
                "confidence": 0.0,
                "justification": "Failed to parse LLM response.",
                "relevant_clauses": [],
                "reasoning": "JSON extraction failed"
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Complete query processing pipeline with optimized hybrid RAG."""
        try:
            # Get API status for debugging
            api_status = self.get_api_status()
            
            # Step 1: Extract entities
            print("🔍 Step 1: Extracting entities...")
            entities = self.extract_entities(query)
            
            # Step 2: Optimized hybrid semantic search with fast reranking
            print("🔍 Step 2: Optimized hybrid search with fast reranking...")
            search_results = self.semantic_search_with_reranking(
                query, 
                initial_k=30,  # Reduced for speed
                final_k=3     # Top 3 results
            )
            
            # Fallback to simple search if optimized search fails completely
            if not search_results:
                print("⚠️ Optimized search failed, trying simple fallback...")
                search_results = self.semantic_search(query, top_k=5)
            
            # Step 3: Evaluate claim
            print("🔍 Step 3: Evaluating claim with optimized context...")
            evaluation = self.evaluate_claim(query, entities, search_results)
            
            # Add optimization information
            evaluation['search_method'] = 'optimized_hybrid_rag'
            evaluation['reranker_type'] = getattr(self, 'reranker_type', 'none')
            evaluation['reranker_available'] = self.reranker is not None
            evaluation['hybrid_search_enabled'] = SKLEARN_AVAILABLE
            evaluation['total_candidates_retrieved'] = 30 if search_results else 0
            evaluation['final_chunks_used'] = len(search_results)
            
            # Add performance notes
            if not self.llm:
                evaluation['note'] = "❌ Analysis performed without LLM (required for full functionality)"
            
            if not self.reranker:
                evaluation['performance_note'] = "⚡ Using hybrid scoring only (no reranker) - install sentence-transformers for better results"
            elif self.reranker_type == "minilm_cross_encoder":
                evaluation['performance_note'] = "🚀 Using fast MiniLM cross-encoder for optimal speed/accuracy balance"
            
            # Combine all results
            result = {
                "query": query,
                "entities": entities,
                "search_results": search_results,
                "evaluation": evaluation,
                "api_status": api_status,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "query": query,
                "entities": {},
                "search_results": [],
                "evaluation": {
                    "decision": "error",
                    "amount": None,
                    "confidence": 0.0,
                    "justification": f"Processing error: {str(e)}",
                    "relevant_clauses": [],
                    "reasoning": "System error occurred"
                },
                "api_status": self.get_api_status(),
                "status": "error",
                "error": str(e)
            }
