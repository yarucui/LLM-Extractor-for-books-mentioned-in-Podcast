import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import count_words, safe_json_loads

class BookAnalysis(BaseModel):
    book_name: str = Field(description="The OFFICIAL FULL TITLE of the book, including subtitles.")
    author_name: Optional[str] = Field(description="The full name of the author.")
    mention_type: str = Field(description="Nature of mention: ['critique', 'reference', 'recommendation', 'author_interview', 'self_promotion', 'advertisement'].")
    recommend_intensity: str = Field(description="Intensity: ['critical', 'negative', 'neutral', 'positive', 'strong_recommendation'].")
    author_present: bool = Field(description="True if the author is a guest.")
    search_query_used: str = Field(default="", description="Leave empty. This will be populated by the search agent.")
    goodreads_url: Optional[str] = Field(default=None, description="Leave null. This will be populated by the search agent.")

class BookContextBlock(BaseModel):
    start_snippet: str = Field(description="The EXACT first 15-20 words of the discussion block as they appear in the transcript. This must be a verbatim copy.")
    end_snippet: str = Field(description="The EXACT last 15-20 words of the discussion block as they appear in the transcript. This must be a verbatim copy.")
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
  
  STRATEGY: ANCHOR-BASED EXTRACTION
  1. IDENTIFY: Find all segments in the transcript where a book, its themes, or its content are being discussed.
  2. MARK BOUNDARIES: For each discussion, identify the EXACT starting sentence and the EXACT ending sentence.
     - Start Snippet: The first 15-20 words of the segment.
     - End Snippet: The last 15-20 words of the segment.
     - IMPORTANT: These snippets MUST be copied VERBATIM from the transcript. Do not change a single character.
  3. ANALYZE: For each segment, identify the specific book(s) and author(s).
  
  RULES:
  - Focus on books only.
  - Ensure the start and end snippets are unique enough to be found in the text.
  - Do not summarize or use ellipses. Just provide the markers.
  """

    def _normalize_text(self, text: str) -> str:
        """
        Removes all non-alphanumeric characters and converts to lowercase for robust matching.
        """
        return re.sub(r'[^a-z0-9]', '', text.lower())

    def _extract_verbatim_text(self, transcript: str, start_snippet: str, end_snippet: str) -> str:
        """
        Advanced helper to extract text between two markers using normalized fuzzy matching.
        """
        try:
            # 1. Try exact match first (fastest)
            start_snippet = start_snippet.strip().replace('...', '')
            end_snippet = end_snippet.strip().replace('...', '')
            
            start_idx = transcript.find(start_snippet)
            if start_idx != -1:
                end_idx = transcript.find(end_snippet, start_idx + len(start_snippet))
                if end_idx != -1:
                    return transcript[start_idx : end_idx + len(end_snippet)].strip()

            # 2. Robust Normalized Search
            # We create a mapping of normalized characters back to original indices
            norm_transcript = ""
            index_map = []
            for i, char in enumerate(transcript):
                norm_char = char.lower()
                if norm_char.isalnum():
                    norm_transcript += norm_char
                    index_map.append(i)
            
            norm_start = self._normalize_text(start_snippet)
            norm_end = self._normalize_text(end_snippet)
            
            # Find start in normalized text
            norm_start_idx = norm_transcript.find(norm_start)
            if norm_start_idx == -1:
                # Try even shorter prefix if still not found
                norm_start_idx = norm_transcript.find(norm_start[:15])
            
            if norm_start_idx != -1:
                # Find end in normalized text starting from norm_start_idx
                norm_end_idx = norm_transcript.find(norm_end, norm_start_idx + len(norm_start))
                if norm_end_idx == -1:
                    # Try even shorter suffix
                    norm_end_idx = norm_transcript.find(norm_end[-15:], norm_start_idx + len(norm_start))
                
                if norm_end_idx != -1:
                    # Map back to original indices
                    real_start = index_map[norm_start_idx]
                    real_end = index_map[norm_end_idx + len(norm_end) - 1] + 1
                    return transcript[real_start:real_end].strip()
            
            return f"[Extraction failed for markers: {start_snippet[:30]}...{end_snippet[-30:]}]"
        except Exception as e:
            return f"[Extraction Error: {str(e)}]"

    def extract_mentions_batch(self, episodes: List[Dict[str, Any]], max_retries: int = 5) -> Dict[str, Any]:
        """
        Extracts book context blocks and metadata from a batch of episodes.
        Returns a dict with 'mentions' and 'usage'.
        """
        combined_prompt = "Analyze the following podcast episodes for book discussions. Identify the start and end snippets for each discussion block and extract detailed book metadata:\n\n"
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
                
                # Capture usage
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }

                raw_text = response.choices[0].message.content
                if not raw_text:
                    return {"mentions": [], "usage": usage}

                data = safe_json_loads(raw_text)
                if not data or not isinstance(data, dict):
                    return {"mentions": [], "usage": usage}
                
                blocks = data.get("blocks", [])
                
                all_flattened_results = []
                id_to_transcript = {str(ep['episode_id']): ep['episode_transcript'] for ep in episodes}
                id_to_title = {str(ep['episode_id']): ep['episode_title'] for ep in episodes}
                
                for block in blocks:
                    eid = str(block.get('episode_id', 'unknown'))
                    transcript = id_to_transcript.get(eid, "")
                    ep_name = id_to_title.get(eid, 'Unknown Episode')
                    
                    # Use anchor-based extraction
                    context = self._extract_verbatim_text(
                        transcript, 
                        block.get('start_snippet', ''), 
                        block.get('end_snippet', '')
                    )
                    
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
                
                return {"mentions": all_flattened_results, "usage": usage}
                    
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
                    return {"mentions": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        
        print(f"Max retries reached for batch extraction.")
        return {"mentions": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
