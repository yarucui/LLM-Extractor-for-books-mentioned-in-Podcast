import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import safe_json_loads

class SearchResult(BaseModel):
    goodreads_url: Optional[str] = Field(description="The official Goodreads URL for the book. Must be a /book/show/ link.")
    page_title: Optional[str] = Field(description="The title of the Goodreads page found. Used for verification.")
    confidence: float = Field(description="Confidence score from 0 to 1.")
    search_query_used: str = Field(description="The exact query used for the search.")

class BookSearcher:
    def __init__(self, api_key: str, model_name: str = "google/gemini-3.1-pro-preview"):
        # Clean API key
        api_key = api_key.strip().strip('"').strip("'") if api_key else ""
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Podcast Book Search Agent",
            }
        )
        # Always use :online for the searcher
        if not model_name.endswith(":online"):
            self.model_name = f"{model_name}:online"
        else:
            self.model_name = model_name
            
        self.system_instruction = """You are a dedicated Search Agent specialized in finding official Goodreads URLs for books.
        
        STRATEGY:
        1. CONSTRUCT QUERY: Use the pattern: "site:goodreads.com [Book Title] [Author]"
        2. SEARCH: Perform a web search.
        3. EXTRACT: You MUST find the actual URL from the search results. 
           - Valid format: https://www.goodreads.com/book/show/[ID]-[Title]
           - DO NOT guess the ID. If you don't see a URL with '/book/show/', it's likely not a direct book page.
        4. VERIFY: Capture the title of the page you found to ensure it matches the requested book.
        
        CRITICAL RULES:
        - DO NOT CONSTRUCT OR GUESS THE URL. If the search results don't provide a clear '/book/show/' link, return null.
        - AVOID search result pages (e.g., /search?q=...).
        - If the URL you provide leads to a 404 or "Page not found", it's because you guessed the ID. DO NOT GUESS.
        - Accuracy is more important than providing a link. If unsure, return null.
        """

    def search_goodreads(self, book_name: str, author_name: Optional[str], exclude_urls: List[str] = [], max_retries: int = 5) -> Dict[str, Any]:
        """
        Searches for a Goodreads URL for a specific book and author.
        Returns a dict with 'result' and 'usage'.
        """
        query_hint = f"'{book_name}' by {author_name}" if author_name else f"'{book_name}'"
        prompt = f"Find the official Goodreads URL for the book: {query_hint}. Use the query pattern: '[Title] [Author] book goodreads'."
        
        if exclude_urls:
            prompt += f"\n\nCRITICAL: DO NOT return any of these URLs as they have been verified as incorrect: {', '.join(exclude_urls)}"
        
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "search_result",
                            "strict": True,
                            "schema": SearchResult.model_json_schema()
                        }
                    }
                )
                
                # Capture usage
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }

                raw_text = response.choices[0].message.content
                if not raw_text:
                    return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": usage}

                data = safe_json_loads(raw_text)
                if not data or not isinstance(data, dict):
                    return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": usage}
                return {"result": data, "usage": usage}
                    
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    retries += 1
                    wait_time = 60
                    delay_match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                    if delay_match:
                        wait_time = float(delay_match.group(1)) + 2
                    
                    print(f"Rate limit hit (429) in searcher. Retry {retries}/{max_retries}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error searching for book: {e}")
                    return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        
        return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
