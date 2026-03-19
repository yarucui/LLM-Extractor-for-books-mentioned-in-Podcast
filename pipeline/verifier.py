import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from .utils import count_words

class BookVerifier:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = """You are a senior research auditor specializing in book metadata.
        Your task is to verify and normalize book mentions.

        For each mention, use your internal knowledge and the provided search tool to:
        1. is_book: Confirm if this is definitely a book (true/false).
        2. is_normalized_book_name: Confirm if the OFFICIAL FULL TITLE of the book correct (true/false).
        3. normalized_author_name: Confirm if the OFFICIAL FULL NAME of the author correct (true/false).
        4. isbn_verified: Confirm if the verified 13-digit ISBN correct (true/false).
        5. verification_notes: Briefly explain any corrections made (e.g., "Normalized title from 'Lore' to 'The World of Lore'").

        OUTPUT FORMAT:
        {
        "is_book": boolean,
        "is_normalized_book_name": boolean,
        "normalized_author_name": boolean,
        "isbn_verified": boolean,
        "verification_notes": string
        }
        IMPORTANT: Your response must be a valid JSON object and nothing else. Do not include markdown formatting like ```json.
        """

    def verify_mention(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies and normalizes a single book mention.
        """
        prompt = f"Verify and normalize this book mention. Use Google Search to find official metadata if needed. Return the result as a JSON object:\n\n{json.dumps(mention, indent=2)}"
        
        try:
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=[grounding_tool]
                )
            )
            
            # Clean response text for JSON parsing
            text = response.text.strip()
            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            text = text.strip()

            # Parse JSON response
            verification = json.loads(text)
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
