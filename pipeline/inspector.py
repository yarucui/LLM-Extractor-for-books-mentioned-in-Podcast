import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import safe_json_loads

class InspectionResult(BaseModel):
    actual_title: str = Field(description="The book title actually displayed on the webpage.")
    actual_author: str = Field(description="The author name actually displayed on the webpage.")
    is_match: bool = Field(description="True if the page content matches the expected book and author.")
    reason: str = Field(description="Brief explanation of why it matches or doesn't match.")

class URLInspector:
    def __init__(self, api_key: str, model_name: str = "google/gemini-3-flash-preview"):
        # Clean API key
        api_key = api_key.strip().strip('"').strip("'") if api_key else ""
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Podcast URL Metadata Inspector",
            }
        )
        # Use standard model (no :online needed now)
        self.model_name = model_name.replace(":online", "")
            
        self.system_instruction = """You are a precise metadata auditor.
        Your task is to compare the book metadata extracted from a podcast with the metadata scraped from a Goodreads URL.
        
        DECIDE: 'is_match' is true if the scraped title and author correspond to the expected book.
        
        RULES:
        - Be strict but allow for fuzzy matching (e.g., 'The Communist Manifesto' vs 'Manifesto of the Communist Party').
        - If the scraped title or author is null/empty, set is_match to false.
        - If the scraped page is clearly a different book, set is_match to false.
        """

    def inspect_metadata(self, scraped_title: Optional[str], scraped_author: Optional[str], expected_title: str, expected_author: Optional[str]) -> Dict[str, Any]:
        """
        Compares scraped metadata with expected metadata using LLM.
        """
        if not scraped_title:
            return {"result": {"is_match": False, "reason": "Scraped title is empty", "actual_title": "", "actual_author": ""}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        prompt = f"""
        EXPECTED (from Podcast):
        Title: {expected_title}
        Author: {expected_author if expected_author else 'Unknown'}
        
        SCRAPED (from Goodreads):
        Title: {scraped_title}
        Author: {scraped_author if scraped_author else 'Unknown'}
        
        Do these represent the same book?"""
        
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
                        "name": "inspection_result",
                        "strict": True,
                        "schema": InspectionResult.model_json_schema()
                    }
                }
            )
            
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }

            raw_text = response.choices[0].message.content
            data = safe_json_loads(raw_text)
            
            if data:
                return {"result": data, "usage": usage}
            
            return {"result": {"is_match": False, "reason": "Failed to parse inspector response", "actual_title": scraped_title, "actual_author": scraped_author}, "usage": usage}
                
        except Exception as e:
            print(f"Inspector Error: {e}")
            return {"result": {"is_match": False, "reason": f"Error: {e}", "actual_title": scraped_title, "actual_author": scraped_author}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
