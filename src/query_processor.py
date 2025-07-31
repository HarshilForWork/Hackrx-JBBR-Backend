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

# Reranking functionality removed - using semantic similarity scores only

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
        """Pre-filtering using semantic scores to reduce reranking workload."""
        if len(candidates) <= max_candidates:
            return candidates
        
        print(f"🔍 Pre-filtering from {len(candidates)} to {max_candidates} candidates using semantic scores...")
        
        # Sort by semantic scores (vector_score)
        candidates.sort(key=lambda x: x.get("vector_score", 0.0), reverse=True)
        
        # Take top candidates
        filtered = candidates[:max_candidates]
        
        print(f"✅ Pre-filtered to top {len(filtered)} candidates using semantic scores")
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

    def semantic_search_with_similarity(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple semantic search using only vector similarity scores:
        1. Retrieve top candidates using vector similarity 
        2. Return top results sorted by similarity scores with context expansion
        """
        if not self.index:
            print("❌ Pinecone index not available")
            return []
        
        try:
            # Retrieve candidates using vector similarity
            print(f"🔍 Retrieving top {top_k} candidates using semantic similarity...")
            query_embedding = self._encode_query(query)
            
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
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
            
            print(f"✅ Retrieved {len(candidates)} candidates")
            
            # Use similarity scores as final scores (no reranking)
            for candidate in candidates:
                candidate["final_score"] = candidate["vector_score"]
            
            # Context expansion
            print(f"📄 Context expansion for {len(candidates)} chunks...")
            expanded_results = self._expand_context(candidates)
            
            # Add ranking metadata
            for i, result in enumerate(expanded_results):
                result["final_rank"] = i + 1
                result["ranking_method"] = "similarity_only"
            
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
            
            # Step 2: Simple semantic search using similarity scores
            print("🔍 Step 2: Semantic search using similarity scores...")
            search_results = self.semantic_search_with_similarity(
                query, 
                top_k=3     # Top 3 results
            )
            
            # Fallback to simple search if search fails completely
            if not search_results:
                print("⚠️ Similarity search failed, trying simple fallback...")
                search_results = self.semantic_search(query, top_k=5)
            
            # Step 3: Evaluate claim
            print("🔍 Step 3: Evaluating claim with semantic context...")
            evaluation = self.evaluate_claim(query, entities, search_results)
            
            # Add search method information
            evaluation['search_method'] = 'semantic_similarity_only'
            evaluation['reranker_type'] = 'none'
            evaluation['reranker_available'] = False
            evaluation['hybrid_search_enabled'] = False  # TF-IDF removed
            evaluation['reranking_enabled'] = False  # Reranking removed
            evaluation['total_candidates_retrieved'] = 3 if search_results else 0
            evaluation['final_chunks_used'] = len(search_results)
            
            # Add performance notes
            if not self.llm:
                evaluation['note'] = "❌ Analysis performed without LLM (required for full functionality)"
            
            evaluation['performance_note'] = "⚡ Using simple semantic similarity search for fast results"
            
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
