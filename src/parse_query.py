"""
Module: parse_query.py
Functionality: Query parsing and entity extraction using Gemini 2.5 Pro (Google Generative AI API).
"""
import os
from typing import Dict
# Placeholder import for Gemini API
# import google.generativeai as genai

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')  # Replace with your actual key or set as env var


def extract_entities_from_query(query: str) -> Dict:
    """
    Given a natural language query, use Gemini 2.5 Pro to extract key entities.
    Returns a dictionary like {'age': 46, 'procedure': 'knee surgery', 'location': 'Pune', 'policy_duration': '3 months'}
    """
    # --- Placeholder for Gemini API call ---
    # In production, use the Gemini API to extract entities.
    # For prototype, simulate extraction with a mock response.
    # Example Gemini prompt: "Extract age, procedure, location, and policy_duration from: {query}"
    # response = genai.generate_entities(query, api_key=GEMINI_API_KEY)
    # return response.entities

    # MOCKED for prototype/demo:
    mock_entities = {
        'age': 46,
        'procedure': 'knee surgery',
        'location': 'Pune',
        'policy_duration': '3 months'
    }
    return mock_entities 