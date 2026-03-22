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
        Your task is to verify and normalize book mentions using your internal knowledge and the provided search tool.

        For each mention, confirm:
        1. is_book: Confirm if this is definitely a book (true/false).
        2. is_normalized_book_name: Confirm if the OFFICIAL FULL TITLE of the book correct (true/false).
        3. normalized_author_name: Confirm if the OFFICIAL FULL NAME of the author correct (true/false).
        4. isbn_verified: Confirm if the verified 13-digit ISBN correct (true/false).
        5. verification_notes: Briefly explain any corrections made.

        OUTPUT FORMAT:
        You MUST return your findings as a JSON object at the end of your response.
        The JSON object MUST have this structure:
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
        prompt = f"Verify and normalize this book mention using Google Search. Return the result as a JSON object at the end of your response:\n\n{json.dumps(mention, indent=2)}"
        
        try:
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=[grounding_tool],
                    max_output_tokens=2048
                )
            )
            
            # Check if response has text safely
            try:
                raw_text = response.text if response.text else ""
                
                if not raw_text:
                    # Check for parts if text is missing (might be tool calls only)
                    parts = response.candidates[0].content.parts if response.candidates else []
                    has_tool_calls = any(part.tool_call for part in parts) if parts else False
                    print(f"Warning: No text in response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}. Has tool calls: {has_tool_calls}")
                    return mention
            except Exception as e:
                print(f"Warning: Could not access response text: {e}")
                return mention

            # Clean response text for JSON parsing
            text = raw_text.strip()
            
            # Robust JSON object extraction using regex
            import re
            # Look for the last JSON object in the text (in case there are citations/text before it)
            json_matches = list(re.finditer(r'\{.*\}', text, re.DOTALL))
            if json_matches:
                # Take the last match as it's most likely the final JSON object
                json_text = json_matches[-1].group(0)
            else:
                # Fallback to markdown block cleaning
                if "```json" in text:
                    json_text = text.split("```json")[-1].split("```")[0]
                elif "```" in text:
                    json_text = text.split("```")[-1].split("```")[0]
                else:
                    json_text = text
            
            json_text = json_text.strip()

            # Parse JSON response
            try:
                verification = json.loads(json_text, strict=False)
            except json.JSONDecodeError as e:
                # One last attempt: try to find anything that looks like a JSON object
                match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if match:
                    try:
                        verification = json.loads(match.group(0), strict=False)
                    except:
                        print(f"JSON parsing error during verification: {e}")
                        print(f"Problematic text snippet: {json_text[:100]}...{json_text[-100:]}")
                        return mention
                else:
                    print(f"JSON parsing error during verification: {e}")
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
