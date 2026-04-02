import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import count_words

class BookAnalysis(BaseModel):
    book_name: str = Field(description="The OFFICIAL FULL TITLE of the book, including subtitles.")
    author_name: Optional[str] = Field(description="The full name of the author.")
    mention_type: str = Field(description="Nature of mention: ['critique', 'reference', 'recommendation', 'author_interview', 'self_promotion', 'advertisement'].")
    recommend_intensity: str = Field(description="Intensity: ['critical', 'negative', 'neutral', 'positive', 'strong_recommendation'].")
    author_present: bool = Field(description="True if the author is a guest.")
    search_query_used: str = Field(default="", description="Leave empty. This will be populated by the search agent.")
    goodreads_url: Optional[str] = Field(default=None, description="Leave null. This will be populated by the search agent.")

class BookContextBlock(BaseModel):
    context_quote: str = Field(description="A long, continuous segment from the transcript where a book or its content is being discussed. Include enough surrounding dialogue to capture the full essence of the discussion, even if the book title isn't explicitly repeated in every sentence.")
    books: List[BookAnalysis] = Field(description="List of books identified and analyzed within this specific context quote.")
    episode_id: str = Field(description="The ID of the episode.")

class BookMentionsResponse(BaseModel):
    blocks: List[BookContextBlock]

class BookExtractor:
    def __init__(self, api_key: str, model_name: str = "google/gemini-3.1-pro-preview"):
        # Clean API key (remove quotes and whitespace)
        api_key = api_key.strip().strip('"').strip("'") if api_key else ""
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Podcast Book Context Extractor",
            }
        )
        self.model_name = model_name
            
        self.system_instruction = """You are a senior research analyst specializing in literary discussions in podcasts.
  
  STRATEGY: DEEP-CONTEXT EXTRACTION
  1. IDENTIFY: Find all segments in the transcript where a book, its themes, or its content are being discussed.
  2. EXTRACT CONTEXT: Capture the ENTIRE discussion block (Context Quote). 
     - This must be a long, continuous segment.
     - Include host/guest interactions, philosophical or practical tangents derived from the book, and any in-depth analysis.
     - DO NOT truncate. If the discussion spans multiple minutes, capture the whole span to reflect the 'intellectual arc' of the conversation.
  3. ANALYZE: For each Context Quote, identify the specific book(s) and author(s).
  
  RULES:
  - Focus on books only. Exclude other media (podcasts, movies, etc.).
  - Context Quote is your primary unit. It must be substantial and capture the 'why' and 'so what' of the discussion.
  - You have access to web search via the ':online' model suffix. Use it to verify book titles and authors.
  """

    def extract_mentions_batch(self, episodes: List[Dict[str, Any]], max_retries: int = 5) -> List[Dict[str, Any]]:
        """
        Extracts book context blocks and metadata from a batch of episodes.
        """
        combined_prompt = "Analyze the following podcast episodes for book discussions. Extract long context quotes and detailed book metadata including Goodreads URLs:\n\n"
        for ep in episodes:
            combined_prompt += f"--- EPISODE START ---\n"
            combined_prompt += f"Episode ID: {ep['episode_id']}\n"
            combined_prompt += f"Episode Title: {ep['episode_title']}\n"
            combined_prompt += f"Transcript:\n{ep['episode_transcript']}\n"
            combined_prompt += f"--- EPISODE END ---\n\n"

        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": combined_prompt}
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "book_context_extraction",
                            "strict": True,
                            "schema": BookMentionsResponse.model_json_schema()
                        }
                    }
                )
                
                raw_text = response.choices[0].message.content
                if not raw_text:
                    return []

                data = json.loads(raw_text, strict=False)
                blocks = data.get("blocks", [])
                
                all_flattened_results = []
                id_to_title = {ep['episode_id']: ep['episode_title'] for ep in episodes}
                
                for block in blocks:
                    context = block.get('context_quote', '')
                    eid = str(block.get('episode_id', 'unknown'))
                    ep_name = id_to_title.get(eid, 'Unknown Episode')
                    
                    for book in block.get('books', []):
                        # Flatten the structure for the existing database/CSV index
                        result = {
                            'context_quote': context,
                            'episode_id': eid,
                            'episode_name': ep_name,
                            'word_count': count_words(context),
                            **book
                        }
                        all_flattened_results.append(result)
                
                return all_flattened_results
                    
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    retries += 1
                    wait_time = 60 # Default wait time
                    
                    # Try to extract retry delay from error message
                    delay_match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                    if delay_match:
                        wait_time = float(delay_match.group(1)) + 2 # Add a small buffer
                    
                    print(f"Rate limit hit (429). Retry {retries}/{max_retries}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error extracting from batch: {e}")
                    return []
        
        print(f"Max retries reached for batch extraction.")
        return []
