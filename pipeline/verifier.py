import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from .utils import count_words

class BookVerifier:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
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
                # Access parts directly to see if there's anything useful
                parts = response.candidates[0].content.parts if response.candidates else []
                raw_text = response.text if response.text else ""
                
                if not raw_text:
                    # Check if there are tool calls or other parts
                    has_tool_calls = any(part.tool_call for part in parts) if parts else False
                    print(f"Warning: No text in response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}. Has tool calls: {has_tool_calls}")
                    # If it's just tool calls without a final response, the model might be stuck in a loop or failing to summarize.
                    return mention
            except Exception as e:
                print(f"Warning: Could not access response text: {e}")
                return mention

            # Clean response text for JSON parsing
            text = raw_text.strip()
            
            # Robust JSON object extraction using regex
            import re
            # Look for the last JSON object in the text (in case there are citations before it)
            json_matches = list(re.finditer(r'\{.*\}', text, re.DOTALL))
            if json_matches:
                # Take the last match as it's most likely the final JSON object after citations
                text = json_matches[-1].group(0)
            else:
                # Fallback to markdown block cleaning if regex fails
                if "```json" in text:
                    text = text.split("```json")[-1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[-1].split("```")[0]
            
            text = text.strip()

            # Parse JSON response
            # Use strict=False to allow control characters like newlines inside strings
            try:
                verification = json.loads(text, strict=False)
            except json.JSONDecodeError as e:
                # Log a snippet of the problematic text for debugging
                if text:
                    print(f"JSON parsing error during verification: {e}")
                    print(f"Problematic text snippet: {text[:100]}...{text[-100:]}")
                return mention
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
