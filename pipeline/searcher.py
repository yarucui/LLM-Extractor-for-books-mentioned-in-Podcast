import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

class SearchResult(BaseModel):
    goodreads_url: Optional[str] = Field(description="The official Goodreads URL for the book.")
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
        1. CONSTRUCT QUERY: Use the pattern: "[Book Title] [Author] book goodreads"
        2. SEARCH: Perform a web search using this specific query.
        3. EXTRACT: Find the URL that matches the pattern 'https://www.goodreads.com/book/show/...'.
        4. VERIFY: Ensure the URL points to the correct book and author mentioned in the request.
        
        RULES:
        - ONLY return a Goodreads URL.
        - If multiple editions exist, prefer the main one.
        - If no high-confidence match is found, return null for the URL.
        - DO NOT guess or hallucinate URLs.
        """

    def search_goodreads(self, book_name: str, author_name: Optional[str], max_retries: int = 5) -> Dict[str, Any]:
        """
        Searches for a Goodreads URL for a specific book and author.
        """
        query_hint = f"'{book_name}' by {author_name}" if author_name else f"'{book_name}'"
        prompt = f"Find the official Goodreads URL for the book: {query_hint}. Use the query pattern: '[Title] [Author] book goodreads'."
        
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
                
                raw_text = response.choices[0].message.content
                if not raw_text:
                    return {"goodreads_url": None, "confidence": 0, "search_query_used": ""}

                data = json.loads(raw_text, strict=False)
                return data
                    
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
                    return {"goodreads_url": None, "confidence": 0, "search_query_used": ""}
        
        return {"goodreads_url": None, "confidence": 0, "search_query_used": ""}
