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

# Pinecone embeddings integration
from .embed_and_index import generate_query_embedding_pinecone

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
                from .embed_and_index import check_or_create_pinecone_index
                
                # Ensure index has correct dimensions for Pinecone inference
                print("🔍 Checking/creating Pinecone index with correct dimensions...")
                if check_or_create_pinecone_index(pinecone_api_key, index_name, 1024):
                    self.pc = Pinecone(api_key=pinecone_api_key)
                    self.index = self.pc.Index(index_name)
                    print("✅ Initialized Pinecone client and index")
                else:
                    print("❌ Failed to create/verify Pinecone index")
                    self.pc = None
                    self.index = None
            except Exception as e:
                print(f"Pinecone initialization error: {e}")
                self.pc = None
                self.index = None
        else:
            self.pc = None
            self.index = None
        
        # Use Pinecone embeddings - no fallbacks
        print("✅ Using Pinecone multilingual-e5-large embeddings")
        
        # Reranking disabled - using similarity scores only
        print("✅ Using similarity scores only (no reranking)")
        self.reranker = None
        self.reranker_type = "none"
        
        # Initialize Gemini
        if GENAI_AVAILABLE and gemini_api_key and gemini_api_key != 'dummy':
            try:
                genai.configure(api_key=gemini_api_key)
                # Try different models based on availability
                model_options = [
                    'gemini-2.5-flash',  # More available for students
                    'gemini-2.5-pro',        # Standard model
                    'gemini-2.5-pro'     # Premium model (might be limited)
                ]
                
                self.llm = None
                self.model_name = None
                
                # Try each model until one works
                for model_name in model_options:
                    try:
                        # Set generation config with temperature 0.7
                        # Gemini temperature ranges from 0-2, where 0 is deterministic and 2 is highly random
                        generation_config = {
                            #"temperature": 0.3,  # Medium-high creative temperature (default is 0.9)
                            "top_p": 0.95,
                            "top_k": 40
                        }
                        
                        test_model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config
                        )
                        
                        # Test with a simple query to verify access
                        test_response = test_model.generate_content("Hello")
                        if test_response:
                            self.llm = test_model
                            self.model_name = model_name
                            print(f"Successfully initialized Gemini model: {model_name} with temperature=0.7")
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
    
    def _encode_query(self, query: str) -> List[float]:
        """Encode query using Pinecone's embedding service."""
        try:
            # Use Pinecone embeddings with the same API key
            return generate_query_embedding_pinecone(query, self.pinecone_api_key)
                    
        except Exception as e:
            print(f"❌ Pinecone query encoding error: {e}")
            # Return zero vector as fallback (1024 dimensions for multilingual-e5-large)
            return [0.0] * 1024
    
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
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query for hybrid search."""
        # Remove stop words and special characters
        import re
        import string
        
        # Common stop words
        stop_words = {
            "a", "an", "the", "in", "on", "at", "by", "for", "with", "about", 
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "is", "am", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "having", "do",
            "does", "did", "doing", "would", "should", "could", "ought", "i'm",
            "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've",
            "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
            "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't",
            "wasn't", "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't",
            "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot",
            "couldn't", "mustn't", "let's", "that's", "who's", "what's", "here's",
            "there's", "when's", "where's", "why's", "how's", "of", "this", "that",
            "these", "those", "is", "are", "will", "be"
        }
        
        # Clean the query
        query = query.lower()
        # Remove punctuation
        query = re.sub(f'[{string.punctuation}]', ' ', query)
        # Split into words
        words = query.split()
        # Filter out stop words and single-character words
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add any numbers as they are likely important
        numbers = re.findall(r'\d+', query)
        keywords.extend(numbers)
        
        # Deduplicate
        keywords = list(set(keywords))
        
        # Take the most important keywords (limit to avoid too restrictive filtering)
        if len(keywords) > 5:
            keywords = keywords[:5]
            
        return keywords
        
    def _calculate_hybrid_score(self, candidate: Dict, keywords: List[str]) -> float:
        """Calculate a hybrid score based on vector similarity and keyword presence."""
        # Start with vector similarity score (typically 0-1)
        score = candidate.get("vector_score", 0.0)
        
        if not keywords:
            return score
            
        # Get text from candidate
        text = candidate.get("text", "").lower()
        
        # Count keywords present in the text
        keyword_count = sum(1 for kw in keywords if kw.lower() in text)
        
        # Boost score based on keyword matches (0.05 boost per keyword match)
        keyword_boost = min(0.3, keyword_count * 0.05)  # Cap at 0.3 to avoid dominating vector score
        
        # Combine scores
        hybrid_score = score #+ #keyword_boost
        
        # Add debug info
        candidate["keyword_matches"] = keyword_count
        candidate["keyword_boost"] = keyword_boost
        
        return hybrid_score

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

    def semantic_search_with_similarity(self, query: str, top_k: int = 3, query_embedding: Optional[list] = None) -> List[Dict]:
        """
        Hybrid search using vector similarity and keyword matching:
        1. Retrieve top candidates using vector similarity 
        2. Boost results that contain keywords from the query (post-retrieval)
        3. Return top results sorted by combined scores with context expansion
        """
        if not self.index:
            print("❌ Pinecone index not available")
            return []
        try:
            # Use provided query_embedding if available, else encode
            if query_embedding is not None:
                print("🔍 Using precomputed query embedding for search...")
                embedding = query_embedding
            else:
                embedding = self._encode_query(query)
                
            print(f"🔍 Retrieving top {top_k} candidates using vector search...")
            
            # Extract keywords for hybrid search (post-retrieval)
            keywords = self._extract_keywords(query)
            if keywords:
                print(f"🔍 Will apply keyword boosting after retrieval: {keywords}")
            
            # Get more candidates for post-filtering
            retrieve_k = min(top_k , 20)  # Get more results but cap at 20
            
            # First retrieve with vector search only
            response = self.index.query(
                vector=embedding,
                top_k=retrieve_k,
                include_metadata=True
            )
            
            # Format results (already sorted by similarity score)
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
                print("⚠️ No candidates found")
                return []
                
            print(f"✅ Retrieved {len(candidates)} candidates with vector search")
            
            # Apply hybrid scoring that combines vector similarity with keyword matching
            if keywords:
                print("🔍 Applying hybrid scoring with keyword boosting...")
                for candidate in candidates:
                    hybrid_score = self._calculate_hybrid_score(candidate, keywords)
                    candidate["hybrid_score"] = hybrid_score
                    candidate["final_score"] = hybrid_score
                    
                # Re-sort based on hybrid scores
                candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
                print(f"✅ Re-ranked results using hybrid scoring (vector + keyword boost)")
            else:
                # Use similarity scores as final scores (no hybrid scoring)
                for candidate in candidates:
                    candidate["final_score"] = candidate["vector_score"]
                print("ℹ️ Using pure vector similarity scores (no keywords found)")
            
            # Take top-k after hybrid scoring
            candidates = candidates[:top_k]
            
            # Context expansion
            print(f"📄 Context expansion for {len(candidates)} chunks...")
            expanded_results = self._expand_context(candidates)
            
            # Add ranking metadata
            for i, result in enumerate(expanded_results):
                result["final_rank"] = i + 1
                result["ranking_method"] = "hybrid_post_retrieval" if keywords else "vector_similarity"
                
            self._print_ranking_summary(query, expanded_results)
            return expanded_results
        except Exception as e:
            print(f"❌ Search error: {e}")
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
        print(f"\n📊 Semantic Search Results Summary:")
        print(f"Query: '{query}'")
        print(f"Final results: {len(results)}")
        
        for i, result in enumerate(results, 1):
            similarity_score = result.get("vector_score", 0)
            final_score = result.get("final_score", similarity_score)
            doc_name = result.get("document_name", "Unknown")
            context_expanded = result.get("context_expanded", False)
            
            print(f"  {i}. Doc: {doc_name}")
            print(f"     Similarity: {similarity_score:.3f} | Final: {final_score:.3f} | Context: {'✅' if context_expanded else '❌'}")
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
            
        # Prepare context from search results - include all top vectors
        # Include more context by using the full text and all search results
        context = "\n\n".join([
            f"Document: {result['document_name']}\nRelevance Score: {result.get('score', 0.0):.3f}\nContent: {result['text']}"
            for result in search_results
        ])
        
        prompt = f"""
        You are an insurance policy expert. Based on the following context from policy documents, provide a clear and concise answer in 2-3 sentences.

        QUERY: {query}

        EXTRACTED ENTITIES:
        {json.dumps(entities, indent=2)}

        RELEVANT POLICY CONTEXT:
        {context}

        Return your answer as a JSON object with these fields:
        - answer: string (your concise answer in 2-3 sentences)
        - source_document: string (the document name you primarily referenced)

        For yes/no questions, format your answer as "Yes, [brief reason]" or "No, [brief reason]"
        Consider all the vectors in the context and provide a comprehensive answer.
        Even if the vector had low similarity, it may still contain relevant information.
        Compare the query with the context and provide a decision based on the policy documents.
        Don't look for exact words, but rather the intent and coverage of the query.
        For query questions, focus on the coverage and intent of the query.
        Even if something is not explicitly mentioned, infer from the context.

        Example response for a coverage question:
        {{
          "answer": "Yes, dental implants are covered up to ₹50,000 under your policy's dental care benefit. This is subject to a 6-month waiting period.",
        }}

        Please provide accurate information based solely on the policy documents provided. Only return the JSON object, nothing else.
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
    
    def process_query(self, query: str, query_embedding: Optional[list] = None) -> Dict[str, Any]:
        """Complete query processing pipeline with optimized hybrid RAG."""
        try:
            # Get API status for debugging
            api_status = self.get_api_status()
            # Step 1: (Removed) Extract entities
            # Step 2: Hybrid search using post-retrieval keyword boosting
            print("🔍 Step 1: Performing vector search with post-retrieval keyword boosting...")
            try:
                # Modified semantic search without filter to avoid $contains error
                # Use provided query_embedding if available, else encode
                if query_embedding is not None:
                    print("🔍 Using precomputed query embedding for search...")
                    embedding = query_embedding
                else:
                    embedding = self._encode_query(query)
                
                # Get top candidates using vector search
                response = self.index.query(
                    vector=embedding,
                    top_k=5,  # Get more for post-filtering
                    include_metadata=True
                )
                
                # Format results (already sorted by similarity score)
                search_results = []
                for match in response.matches:
                    search_results.append({
                        "id": match.id,
                        "score": match.score,
                        "text": match.metadata.get("text", ""),
                        "document_name": match.metadata.get("document_name", ""),
                        "page_number": match.metadata.get("page_number", 1)
                    })
                
                # Extract keywords for post-retrieval boosting
                keywords = self._extract_keywords(query)
                if keywords:
                    print(f"🔍 Applying keyword boosting: {keywords}")
                    # Boost scores for results containing keywords
                    for result in search_results:
                        text = result.get("text", "").lower()
                        keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
                        keyword_boost = min(0.3, keyword_matches * 0.05)
                        result["hybrid_score"] = result["score"] + keyword_boost
                    
                    # Re-sort based on hybrid scores
                    search_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
                
                # Limit to top 10
                search_results = search_results[:10]
            except Exception as e:
                print(f"❌ Hybrid search error: {e}")
                # Fall back to simple search
                search_results = self.semantic_search(query, top_k=5)
            # Step 3: Evaluate claim (now just LLM answer)
            print("🔍 Step 2: Evaluating with semantic context...")
            evaluation = self.evaluate_claim(query, {}, search_results)
            # Add search method information
            evaluation['search_method'] = 'hybrid_post_retrieval'
            evaluation['reranker_type'] = 'none'
            evaluation['reranker_available'] = False
            evaluation['hybrid_search_enabled'] = True  # Hybrid search enabled
            evaluation['reranking_enabled'] = False  # Reranking removed
            evaluation['total_candidates_retrieved'] = len(search_results) if search_results else 0
            evaluation['final_chunks_used'] = len(search_results)
            # Add performance notes
            if not self.llm:
                evaluation['note'] = "❌ Analysis performed without LLM (required for full functionality)"
            evaluation['performance_note'] = "⚡ Using post-retrieval keyword boosting for improved accuracy"
            # Combine all results
            result = {
                "query": query,
                # 'entities' removed
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
                # 'entities' removed
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
    
    async def process_queries_batch(self, queries: List[str], query_embeddings: Optional[List[List[float]]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in parallel using asyncio.
        
        Args:
            queries: List of queries to process
            query_embeddings: Optional list of precomputed embeddings for each query
            
        Returns:
            List of results in the same order as input queries
        """
        import asyncio
        
        # Validate inputs
        if not queries:
            return []
        
        if query_embeddings and len(query_embeddings) != len(queries):
            raise ValueError("Number of embeddings must match number of queries")
        
        # Create semaphore to limit concurrent processing (prevent overwhelming the APIs)
        max_concurrent = min(10, len(queries))  # Limit to 10 concurrent queries
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query_async(idx: int, query: str) -> Dict[str, Any]:
            """Process a single query with semaphore protection."""
            async with semaphore:
                try:
                    # Use precomputed embedding if available
                    embedding = query_embeddings[idx] if query_embeddings else None
                    
                    # Run the synchronous process_query in a thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, 
                        self.process_query, 
                        query, 
                        embedding
                    )
                    
                    # Add index to track original order
                    result["original_index"] = idx
                    return result
                    
                except Exception as e:
                    # Return error result in same format
                    return {
                        "query": query,
                        "search_results": [],
                        "evaluation": {
                            "decision": "error",
                            "amount": None,
                            "confidence": 0.0,
                            "justification": f"Batch processing error: {str(e)}",
                            "relevant_clauses": [],
                            "reasoning": "Batch processing system error"
                        },
                        "api_status": self.get_api_status(),
                        "status": "error",
                        "error": str(e),
                        "original_index": idx
                    }
        
        # Create tasks for all queries
        print(f"🚀 Starting batch processing of {len(queries)} queries with max {max_concurrent} concurrent...")
        tasks = []
        for idx, query in enumerate(queries):
            task = process_single_query_async(idx, query)
            tasks.append(task)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Handle any exceptions that occurred
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed queries
                error_result = {
                    "query": queries[idx],
                    "search_results": [],
                    "evaluation": {
                        "decision": "error",
                        "amount": None,
                        "confidence": 0.0,
                        "justification": f"Exception during processing: {str(result)}",
                        "relevant_clauses": [],
                        "reasoning": "Exception occurred"
                    },
                    "api_status": self.get_api_status(),
                    "status": "error",
                    "error": str(result),
                    "original_index": idx
                }
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Sort results by original index to maintain order
        processed_results.sort(key=lambda x: x.get("original_index", 0))
        
        # Remove the original_index field from final results
        final_results = []
        for result in processed_results:
            if "original_index" in result:
                del result["original_index"]
            final_results.append(result)
        
        print(f"✅ Batch processing completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Processed {len(final_results)} queries successfully")
        
        return final_results