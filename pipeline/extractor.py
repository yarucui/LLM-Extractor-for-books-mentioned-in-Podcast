import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from .utils import count_words

class BookExtractor:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = """You are a precise research assistant specializing in podcast analysis.
  Your task is to extract and normalize book mentions from podcast transcripts.

  NORMALIZATION RULES:
  1. Always resolve nicknames, partial titles, or acronyms to the OFFICIAL FULL LIBRARY TITLE (e.g., 'Lore' -> 'The World of Lore').
  2. Use your internal knowledge and the provided search tool to identify the correct book entity.
  3. If multiple books have similar names, use the context to determine the most likely one.

  For each book mention, extract:
  1. book_name: The OFFICIAL FULL TITLE of the book.
  2. author_name: The full name of the author.
  3. isbn: The 13-digit ISBN of the book (if identifiable). This helps ensure unique entity identification.
  4. context_quote: A substantial quote from the transcript providing context.
  5. mention_type: The nature of the mention, must be one of:
  ["critique", "reference", "recommendation", "author_interview", "self_promotion", "advertisement"].
  6. recommend_intensity: must be one of: ["critical", "negative", "neutral", "positive", "strong_recommendation"]
  7. author_present: Boolean (True if the author is a guest on the episode, False otherwise).
  8. episode_id: The ID of the episode where the mention occurred.

  Return a JSON list of objects. If no books are mentioned, return an empty list [].
  Do not include podcasts, movies, or TV shows. Only books.
  """

    def extract_mentions_batch(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts book mentions from a batch of episodes.
        """
        combined_prompt = "Extract and normalize book mentions from the following podcast episodes. Use Google Search to verify titles and authors. Return the results as a JSON list:\n\n"
        for ep in episodes:
            combined_prompt += f"--- EPISODE START ---\n"
            combined_prompt += f"Episode ID: {ep['episode_id']}\n"
            combined_prompt += f"Episode Title: {ep['episode_title']}\n"
            combined_prompt += f"Transcript:\n{ep['episode_transcript']}\n"
            combined_prompt += f"--- EPISODE END ---\n\n"

        try:
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=combined_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    tools=[grounding_tool],
                    max_output_tokens=8192
                )
            )
            
            # Check if response has text safely
            try:
                raw_text = response.text
            except Exception as e:
                print(f"Warning: Could not access response text: {e}")
                return []
                
            if not raw_text:
                print(f"Warning: No text in response. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                return []

            # Clean response text for JSON parsing
            text = raw_text.strip()
            
            # Robust JSON array extraction using regex
            import re
            json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            else:
                # Fallback to markdown block cleaning if regex fails
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
            
            text = text.strip()

            # Parse JSON response
            # Use strict=False to allow control characters like newlines inside strings
            try:
                mentions = json.loads(text, strict=False)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                # Log a snippet of the problematic text for debugging
                print(f"Problematic text snippet: {text[:200]}...{text[-200:]}")
                return []
            all_mentions = []
            
            if isinstance(mentions, list):
                # Create a map of episode_id to episode_title for metadata
                id_to_title = {ep['episode_id']: ep['episode_title'] for ep in episodes}
                
                for m in mentions:
                    # Add metadata
                    eid = str(m.get('episode_id', 'unknown'))
                    m['episode_id'] = eid
                    m['episode_name'] = id_to_title.get(eid, 'Unknown Episode')
                    # Calculate word count of the context quote
                    m['word_count'] = count_words(m.get('context_quote', ''))
                    all_mentions.append(m)
            
            return all_mentions
                
        except Exception as e:
            print(f"Error extracting from batch: {e}")
            return []
