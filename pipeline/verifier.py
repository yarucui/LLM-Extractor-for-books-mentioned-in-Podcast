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
        self.system_instruction = """You are a senior research auditor specializing in book metadata.
        Your task is to verify and normalize book mentions.

        For each mention, use your internal knowledge and the provided search tool to:
        1. is_book: Confirm if this is definitely a book (true/false).
        2. normalized_book_name: Provide the OFFICIAL FULL TITLE of the book.
        3. normalized_author_name: Provide the OFFICIAL FULL NAME of the author.
        4. isbn_verified: Provide the verified 13-digit ISBN.
        5. verification_notes: Briefly explain any corrections made (e.g., "Normalized title from 'Lore' to 'The World of Lore'").

        OUTPUT FORMAT:
        {
        "is_book": boolean,
        "normalized_book_name": string,
        "normalized_author_name": string,
        "isbn_verified": string or null,
        "verification_notes": string
        }"""

    def verify_mention(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies and normalizes a single book mention.
        """
        prompt = f"Verify and normalize this book mention. Use Google Search to find official metadata if needed:\n\n{json.dumps(mention, indent=2)}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    tools=[{ "google_search": {} }]
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
