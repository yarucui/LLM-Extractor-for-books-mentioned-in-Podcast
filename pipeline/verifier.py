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
        Your task is to verify and normalize book mentions based on your internal knowledge.

        For each mention, confirm:
        1. is_book: Confirm if this is definitely a book (true/false).
        2. is_normalized_book_name: Confirm if the OFFICIAL FULL TITLE of the book correct (true/false).
        3. normalized_author_name: Confirm if the OFFICIAL FULL NAME of the author correct (true/false).
        4. isbn_verified: Confirm if the verified 13-digit ISBN correct (true/false).
        5. verification_notes: Briefly explain any corrections made.

        OUTPUT FORMAT:
        Return your findings as a JSON object with the following structure:
        {
        "is_book": boolean,
        "is_normalized_book_name": boolean,
        "normalized_author_name": boolean,
        "isbn_verified": boolean,
        "verification_notes": string
        }
        """

    def verify_mention(self, mention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifies and normalizes a single book mention.
        """
        prompt = f"Verify and normalize this book mention. Return the result as a JSON object:\n\n{json.dumps(mention, indent=2)}"
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    max_output_tokens=2048
                )
            )
            
            # Check if response has text safely
            try:
                raw_text = response.text if response.text else ""
                
                if not raw_text:
                    # Check for parts if text is missing
                    parts = response.candidates[0].content.parts if response.candidates else []
                    print(f"Warning: No text in response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                    return mention
            except Exception as e:
                print(f"Warning: Could not access response text: {e}")
                return mention

            # Clean response text for JSON parsing
            text = raw_text.strip()
            
            # If the model still outputs markdown blocks despite strict mode (rare but possible)
            if text.startswith("```"):
                if "```json" in text:
                    text = text.split("```json")[-1].split("```")[0]
                else:
                    text = text.split("```")[-1].split("```")[0]
            
            text = text.strip()

            # Parse JSON response
            try:
                verification = json.loads(text, strict=False)
            except json.JSONDecodeError as e:
                # Fallback: try regex if direct parsing fails
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        verification = json.loads(json_match.group(0), strict=False)
                    except:
                        print(f"JSON parsing error during verification (even with regex): {e}")
                        print(f"Problematic text snippet: {text[:100]}...{text[-100:]}")
                        return mention
                else:
                    print(f"JSON parsing error during verification: {e}")
                    print(f"Problematic text snippet: {text[:100]}...{text[-100:]}")
                    return mention
            
            if isinstance(verification, dict):
                # Update the mention with verification results
                mention.update(verification)
                # Recalculate word count of the context quote
                mention['word_count'] = count_words(mention.get('context_quote', ''))
                return mention
            
            return mention
        except Exception as e:
            print(f"Error verifying mention: {e}")
            return mention
