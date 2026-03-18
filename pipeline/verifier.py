import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from .utils import count_words

class BookVerifier:
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.system_instruction = """You are a senior research auditor.
Your task is to verify the accuracy of book mentions extracted from podcast transcripts.

For each mention, you will receive:
1. book_name: The title of the book.
2. author_name: The name of the author.
3. context_quote: A substantial quote from the transcript.
4. mention_type: The nature of the mention.
5. recommend_intensity: A scale from 'Critical' to 'Strong Recommendation'.
6. author_present: Boolean (True if the author is a guest).

Verify:
1. is_book: Is this definitely a book? (Exclude movies, TV shows, etc.)
2. author_correct: Is the author name correct based on the context? (Hallucination check)
3. mention_type_correct: Is the mention type accurate based on the context?
4. intensity_appropriate: Is the recommendation intensity appropriate based on the context?
5. author_present_correct: Is the author actually present as a guest?

Return a JSON object with these boolean fields and a 'verification_notes' string.
If the mention is not a book, set is_book to False."""

    def verify_mention(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies a single book mention.
        """
        prompt = f"Book Mention to Verify:\n\n{json.dumps(mention, indent=2)}"
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                ),
                system_instruction=self.system_instruction
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
