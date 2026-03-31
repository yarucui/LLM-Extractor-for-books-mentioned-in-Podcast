import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import count_words

class VerificationResult(BaseModel):
    is_book: bool = Field(description="Confirm if this is definitely a book (true/false).")
    book_name: str = Field(description="The OFFICIAL FULL TITLE of the book. If the input was incomplete, provide the corrected full title here.")
    author_name: Optional[str] = Field(description="The OFFICIAL FULL NAME of the author. If the input was incorrect, provide the corrected name here.")
    goodreads_url: Optional[str] = Field(description="The OFFICIAL Goodreads URL. If the input was missing or incorrect, provide the corrected URL here.")
    verification_notes: str = Field(description="Set to 'all good' if the input was already perfect. Otherwise, briefly explain what you corrected.")

class BookVerifier:
    def __init__(self, api_key: str, model_name: str = "google/gemini-3.1-pro-preview"):
        # Clean API key
        api_key = api_key.strip().strip('"').strip("'") if api_key else ""
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Podcast Book Verifier",
            }
        )
        # Append :online to enable web search if not already present
        if not model_name.endswith(":online"):
            self.model_name = f"{model_name}:online"
        else:
            self.model_name = model_name
            
        self.system_instruction = """You are a senior research auditor specializing in book metadata.
        Your task is to verify and normalize book mentions.
        
        1. VERIFY: Use web search to verify the book's OFFICIAL FULL TITLE, author, and Goodreads URL.
        2. CORRECT: If any information (title, author, URL) is incomplete or incorrect, you MUST provide the corrected version in the respective fields.
        3. NOTES: 
           - If the input was already 100% correct, set verification_notes to 'all good'.
           - If you made changes, explain them briefly in verification_notes.
        
        Search Strategy: For Goodreads URLs, use 'short book name + author name' for better matching.
        """

    def verify_mention(self, mention: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """
        Verifies and normalizes a single book mention with retry logic for rate limits.
        """
        prompt = f"Verify and normalize this book mention. Return the result as a JSON object:\n\n{json.dumps(mention, indent=2)}"
        
        retries = 0
        while retries < max_retries:
            try:
                # OpenRouter Structured Output format
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "verification_result",
                            "strict": True,
                            "schema": VerificationResult.model_json_schema()
                        }
                    }
                )
                
                raw_text = response.choices[0].message.content
                if not raw_text:
                    print(f"Warning: No content in response.")
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
                    # Recalculate word count of the book mention quote
                    mention['word_count'] = count_words(mention.get('book_mention_quote', ''))
                    return mention
                
                return mention
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    retries += 1
                    wait_time = 60 # Default wait time
                    
                    # Try to extract retry delay from error message
                    delay_match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                    if delay_match:
                        wait_time = float(delay_match.group(1)) + 2
                    
                    print(f"Rate limit hit (429) in verifier. Retry {retries}/{max_retries}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error verifying mention: {e}")
                    return mention
        
        print(f"Max retries reached for mention verification.")
        return mention
