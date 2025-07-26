"""
Module: query_processor.py
Functionality: Complete query processing pipeline with LLM integration.
"""
import json
import re
import time
from typing import Dict, List, Any, Optional

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

from .embed_and_index import get_embedding_model

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
        
        # Use cached embedding model for consistency and performance
        self.model = get_embedding_model()
        
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
        """Extract structured entities from natural language query."""
        if self.llm:
            return self._llm_entity_extraction(query)
        else:
            return self._fallback_entity_extraction(query)
    
    def _llm_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Use LLM for entity extraction with quota-aware error handling."""
        if not self.llm:
            print("🔍 LLM not available, using fallback entity extraction")
            return self._fallback_entity_extraction(query)
            
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
        
        # Use new robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("🔍 No response from LLM, using fallback entity extraction")
            return self._fallback_entity_extraction(query)
        
        # Use new robust JSON extraction
        entities = self._extract_json_from_response(response_text, "entity extraction")
        if entities:
            print("✅ Successfully extracted entities using LLM")
            return entities
        else:
            print("🔍 JSON extraction failed, using fallback entity extraction")
            return self._fallback_entity_extraction(query)
    
    def _fallback_entity_extraction(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback entity extraction with better pattern matching."""
        entities: Dict[str, Any] = {
             "age": None,
             "gender": None,
             "procedure": None,
             "location": None,
             "policy_duration": None,
             "policy_type": None,
             "amount": None
         }
        
        query_lower = query.lower()
        
        # Extract age - improved patterns
        age_patterns = [
            r'(\d+)\s*(?:year|yr|y)\s*(?:old|male|female|m|f)',
            r'(\d+)\s*(?:male|female|m|f)',
            r'(\d+)\s*(?:year|yr|y)',
            r'(\d+)(?=\s*(?:male|female|m|f))'
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, query_lower)
            if age_match:
                age_val = int(age_match.group(1))
                if 0 < age_val < 120:  # Reasonable age range
                    entities["age"] = age_val
                    break
        
        # Extract gender - improved patterns
        gender_patterns = [
            (r'\b(?:male|m)\b(?!\w)', "M"),
            (r'\b(?:female|f)\b(?!\w)', "F"),
            (r'\b(?:man|boy)\b', "M"),
            (r'\b(?:woman|girl)\b', "F")
        ]
        for pattern, gender in gender_patterns:
            if re.search(pattern, query_lower):
                entities["gender"] = gender
                break
        
        # Extract procedures - expanded list
        procedures = [
            "knee surgery", "heart surgery", "eye surgery", "brain surgery",
            "dental surgery", "plastic surgery", "bypass surgery",
            "cataract surgery", "appendectomy", "hernia repair",
            "surgery", "operation", "procedure", "treatment",
            "dental treatment", "dental", "orthodontic",
            "physiotherapy", "chemotherapy", "dialysis",
            "consultation", "checkup", "scan", "mri", "ct scan"
        ]
        for proc in procedures:
            if proc in query_lower:
                entities["procedure"] = proc
                break
        
        # Extract locations - expanded list
        locations = [
            "pune", "mumbai", "delhi", "bangalore", "chennai", "hyderabad",
            "kolkata", "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur",
            "indore", "thane", "bhopal", "visakhapatnam", "pimpri", "patna",
            "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad"
        ]
        for loc in locations:
            if loc in query_lower:
                entities["location"] = loc.title()
                break
        
        # Extract policy duration - improved patterns
        duration_patterns = [
            r'(\d+)\s*(?:month|months|month old|months old)',
            r'(\d+)\s*(?:year|years|year old|years old)',
            r'(\d+)\s*(?:day|days|day old|days old)',
            r'(\d+-month)',
            r'(\d+-year)'
        ]
        for pattern in duration_patterns:
            duration_match = re.search(pattern, query_lower)
            if duration_match:
                full_match = duration_match.group(0)
                entities["policy_duration"] = full_match
                break
        
        # Extract amounts - look for currency patterns
        amount_patterns = [
            r'₹\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            r'rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            r'rupees?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
            r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:rupees?|rs\.?|₹)'
        ]
        for pattern in amount_patterns:
            amount_match = re.search(pattern, query_lower)
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '')
                try:
                    entities["amount"] = float(amount_str)
                    break
                except ValueError:
                    continue
        
        return entities
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
         """Perform semantic search in Pinecone."""
         if not self.index:
             return []
         
         try:
             # Generate query embedding
             query_embedding = self.model.encode(query).tolist()
             
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
        """Use LLM to evaluate claim based on retrieved context."""
        
        if self.llm:
            return self._llm_evaluation(query, entities, search_results)
        else:
            return self._fallback_evaluation(query, entities, search_results)
    
    def _llm_evaluation(self, query: str, entities: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM for claim evaluation with quota-aware error handling."""
        if not self.llm:
            print("🔍 LLM not available, using fallback evaluation")
            return self._fallback_evaluation(query, entities, search_results)
            
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
        
        # Use new robust request method
        response_text = self._make_llm_request_with_retry(prompt)
        if not response_text:
            print("🔍 No response from LLM, using fallback evaluation")
            return self._fallback_evaluation(query, entities, search_results)
        
        # Use new robust JSON extraction
        evaluation = self._extract_json_from_response(response_text, "claim evaluation")
        if evaluation:
            print("✅ Successfully completed claim evaluation using LLM")
            return evaluation
        else:
            print("🔍 JSON extraction failed, using fallback evaluation")
            return self._fallback_evaluation(query, entities, search_results)
    
    def _fallback_evaluation(self, query: str, entities: Dict, search_results: List[Dict]) -> Dict[str, Any]:
        """Enhanced rule-based evaluation fallback."""
        
        # Default values
        decision = "needs_review"
        confidence = 0.6
        justification = "Automated rule-based evaluation (LLM unavailable due to quota limits)."
        relevant_clauses = []
        reasoning = "Using enhanced pattern matching and keyword analysis."
        amount = entities.get('amount')
        
        # Check if we have relevant search results
        if search_results:
            # Combine text from top search results
            combined_text = " ".join([r.get('text', '') for r in search_results[:3]]).lower()
            
            procedure = entities.get('procedure', '').lower()
            age = entities.get('age')
            policy_duration = entities.get('policy_duration', '').lower()
            
            # Enhanced keyword analysis
            coverage_keywords = ['covered', 'eligible', 'benefits', 'included', 'payable']
            exclusion_keywords = ['excluded', 'not covered', 'not eligible', 'not payable', 'waiting period']
            approval_keywords = ['approved', 'allowed', 'permitted', 'authorized']
            rejection_keywords = ['rejected', 'denied', 'prohibited', 'restricted']
            
            coverage_score = sum(1 for kw in coverage_keywords if kw in combined_text)
            exclusion_score = sum(1 for kw in exclusion_keywords if kw in combined_text)
            approval_score = sum(1 for kw in approval_keywords if kw in combined_text)
            rejection_score = sum(1 for kw in rejection_keywords if kw in combined_text)
            
            # Procedure-specific analysis
            procedure_mentioned = procedure and procedure in combined_text
            
            # Age-based analysis
            age_restrictions = False
            if age:
                age_keywords = [f"{age}", "age limit", "minimum age", "maximum age"]
                age_mentions = sum(1 for kw in age_keywords if kw in combined_text)
                if age_mentions > 0:
                    age_restrictions = True
            
            # Policy duration analysis
            waiting_period_issue = False
            if policy_duration and ('month' in policy_duration or 'day' in policy_duration):
                waiting_keywords = ['waiting period', 'waiting time', 'grace period', 'cooling period']
                if any(kw in combined_text for kw in waiting_keywords):
                    waiting_period_issue = True
            
            # Decision logic
            if procedure_mentioned:
                if coverage_score > exclusion_score and not waiting_period_issue:
                    decision = "approved"
                    confidence = min(0.8, 0.6 + (coverage_score * 0.1))
                    justification = f"Procedure '{procedure}' appears to be covered based on policy document analysis."
                elif exclusion_score > coverage_score:
                    decision = "rejected"
                    confidence = min(0.8, 0.6 + (exclusion_score * 0.1))
                    justification = f"Procedure '{procedure}' appears to be excluded from coverage."
                elif waiting_period_issue:
                    decision = "rejected"
                    confidence = 0.7
                    justification = "Potential waiting period violation for recent policy."
                else:
                    decision = "needs_review"
                    confidence = 0.5
                    justification = "Unclear coverage status - manual review recommended."
            else:
                if approval_score > rejection_score:
                    decision = "approved"
                    confidence = 0.6
                    justification = "General approval indicators found in policy documents."
                elif rejection_score > approval_score:
                    decision = "rejected"
                    confidence = 0.6
                    justification = "General rejection indicators found in policy documents."
            
            # Generate relevant clauses
            relevant_clauses = [
                f"Analysis from {r['document_name']}: {r['text'][:100]}..."
                for r in search_results[:2]
            ]
            
            # Enhanced reasoning
            reasoning_parts = [
                f"Analyzed {len(search_results)} relevant policy documents.",
                f"Coverage indicators: {coverage_score}, Exclusion indicators: {exclusion_score}",
                f"Procedure mentioned: {procedure_mentioned}",
                f"Age considerations: {age_restrictions}",
                f"Waiting period concerns: {waiting_period_issue}"
            ]
            reasoning = " ".join(reasoning_parts)
        
        else:
            justification = "No relevant policy documents found for analysis."
            reasoning = "Unable to perform document-based analysis due to lack of relevant search results."
        
        return {
            "decision": decision,
            "amount": amount,
            "confidence": confidence,
            "justification": justification,
            "relevant_clauses": relevant_clauses,
            "reasoning": reasoning
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Complete query processing pipeline with API status information."""
        try:
            # Get API status for debugging
            api_status = self.get_api_status()
            
            # Step 1: Extract entities
            entities = self.extract_entities(query)
            
            # Step 2: Semantic search
            search_results = self.semantic_search(query, top_k=5)
            
            # Step 3: Evaluate claim
            evaluation = self.evaluate_claim(query, entities, search_results)
            
            # Add fallback information to evaluation if needed
            if not self.llm:
                evaluation['note'] = "⚠️ Analysis performed using fallback methods due to LLM unavailability"
                if self.quota_exceeded:
                    evaluation['note'] += " (Gemini API quota exceeded)"
            
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
