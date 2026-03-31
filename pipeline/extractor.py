import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import count_words

class BookAnalysis(BaseModel):
    book_name: str = Field(description="The OFFICIAL FULL TITLE of the book, including subtitles. Use web search to verify.")
    author_name: Optional[str] = Field(description="The full name of the author.")
    mention_type: str = Field(description="Nature of mention: ['critique', 'reference', 'recommendation', 'author_interview', 'self_promotion', 'advertisement'].")
    recommend_intensity: str = Field(description="Intensity: ['critical', 'negative', 'neutral', 'positive', 'strong_recommendation'].")
    author_present: bool = Field(description="True if the author is a guest.")
    search_query_used: str = Field(description="The specific search query used to find the Goodreads URL.")
    goodreads_url: Optional[str] = Field(description="The official Goodreads URL for this book.")

class BookMention(BaseModel):
    book_mention_quote: str = Field(description="A specific segment from the episode_quote where this particular book is discussed. Capture the immediate context of the mention.")
    analysis: BookAnalysis

class EpisodeBookSummary(BaseModel):
    episode_id: str = Field(description="The ID of the episode.")
    episode_quote: str = Field(description="The continuous segment of the transcript starting from the very first book mention in the episode until the very last book mention and its related discussion concludes. If no books are mentioned, leave empty.")
    mentions: List[BookMention] = Field(description="List of specific book mentions identified within the episode_quote.")

class BookMentionsResponse(BaseModel):
    episodes: List[EpisodeBookSummary]

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
        # Append :online to enable web search if not already present
        if not model_name.endswith(":online"):
            self.model_name = f"{model_name}:online"
        else:
            self.model_name = model_name
            
        self.system_instruction = """You are a senior research analyst specializing in literary discussions in podcasts.
  
  STRATEGY: HIERARCHICAL EXTRACTION
  1. IDENTIFY EPISODE RANGE: For each episode, find the entire range of the transcript that contains book discussions. 
     - The 'episode_quote' MUST start from the very first mention of any book and continue until the last book discussion and its related reflections have concluded.
  2. EXTRACT MENTIONS: Within that 'episode_quote', identify specific segments ('book_mention_quote') for each individual book discussed.
  3. ANALYZE: For each mention, provide detailed book metadata.
  4. SEARCH & GROUND: For every book identified, you MUST perform a web search to find its official Goodreads URL.
  
  RULES:
  - Focus on books only.
  - 'episode_quote' is the macro-context (one per episode).
  - 'book_mention_quote' is the micro-context (multiple per episode, split from the episode_quote).
  - You have access to web search via the ':online' model suffix.
  """

    def extract_mentions_batch(self, episodes: List[Dict[str, Any]], max_retries: int = 5) -> List[Dict[str, Any]]:
        """
        Extracts hierarchical book mentions from a batch of episodes.
        """
        combined_prompt = "Analyze the following podcast episodes for book discussions. Extract the full episode-level discussion range and specific book mention quotes with metadata:\n\n"
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
                ep_summaries = data.get("episodes", [])
                
                all_flattened_results = []
                id_to_title = {ep['episode_id']: ep['episode_title'] for ep in episodes}
                
                for ep_summary in ep_summaries:
                    ep_quote = ep_summary.get('episode_quote', '')
                    eid = str(ep_summary.get('episode_id', 'unknown'))
                    ep_name = id_to_title.get(eid, 'Unknown Episode')
                    
                    for mention in ep_summary.get('mentions', []):
                        mention_quote = mention.get('book_mention_quote', '')
                        analysis = mention.get('analysis', {})
                        
                        # Flatten the structure
                        result = {
                            'episode_id': eid,
                            'episode_name': ep_name,
                            'episode_quote': ep_quote,
                            'book_mention_quote': mention_quote,
                            'word_count': count_words(mention_quote),
                            **analysis
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
