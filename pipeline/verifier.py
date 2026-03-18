import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from .utils import count_words

class BookVerifier:
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = """You are a strict verification system for extracted book mentions.
        Your task is to validate whether each extracted field is correct based ONLY on the provided context_quote.

        RULES:
        1. Do NOT use external knowledge.
        2. Only rely on the context_quote.
        3. If the quote does not support the field, mark it as incorrect.

        For each mention, evaluate:

        - is_book: true if this is clearly a book (not movie, podcast, etc.)
        - author_correct: true if the author is explicitly supported by the quote
        - mention_type_correct: true if the label matches the quote
        - intensity_correct: true if the sentiment matches the quote
        - author_present_correct: true if the quote clearly indicates the author is present

        If a field is incorrect:
        - Provide a corrected value in a separate field

        OUTPUT FORMAT:

        {
        "is_book": true/false,
        "author_correct": true/false,
        "correct_author": null or string,
        "mention_type_correct": true/false,
        "correct_mention_type": null or string,
        "intensity_correct": true/false,
        "correct_intensity": null or string,
        "author_present_correct": true/false,
        "correct_author_present": null or boolean,
        "verification_notes": "short explanation"
        }

        IMPORTANT:
        - Always return ALL fields
        - Never mix booleans and strings in the same field
        - Keep verification_notes concise"""

    def verify_mention(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies a single book mention.
        """
        prompt = f"Book Mention to Verify:\n\n{json.dumps(mention, indent=2)}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                )
            )
            
            # Parse JSON response
            verification = json.loads(response.text)
            if isinstance(verification, dict):
                # Update the mention with verification results
                mention.update(verification)
                # Recalculate word count of the context quote in Python
                mention['word_count'] = count_words(mention.get('context_quote', ''))
                return mention
            
            return mention
        except Exception as e:
            print(f"Error verifying mention: {e}")
            return mention
