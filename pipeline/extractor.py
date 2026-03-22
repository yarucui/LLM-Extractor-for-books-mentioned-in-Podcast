import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from .utils import count_words

class BookExtractor:
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.system_instruction = """You are a precise research assistant specializing in podcast analysis.
  Your task is to extract book mentions from podcast transcripts.

  For each book mention, extract:
  1. book_name: The title as mentioned in the podcast.
  2. author_name: The author as mentioned (or null if not mentioned).
  3. context_quote: A substantial quote from the transcript providing context.
  4. mention_type: The nature of the mention, must be one of:
  ["critique", "reference", "recommendation", "author_interview", "self_promotion", "advertisement"].
  5. recommend_intensity: must be one of: ["critical", "negative", "neutral", "positive", "strong_recommendation"]
  6. author_present: Boolean (True if the author is a guest on the episode, False otherwise).
  7. episode_id: The ID of the episode where the mention occurred.

  Return a JSON list of objects. If no books are mentioned, return an empty list [].
  Do not include podcasts, movies, or TV shows. Only books.
  """

    def extract_mentions_batch(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts book mentions from a batch of episodes.
        """
        combined_prompt = "Extract all book mentions from the following podcast episodes. Return the results as a JSON list:\n\n"
        for ep in episodes:
            combined_prompt += f"--- EPISODE START ---\n"
            combined_prompt += f"Episode ID: {ep['episode_id']}\n"
            combined_prompt += f"Episode Title: {ep['episode_title']}\n"
            combined_prompt += f"Transcript:\n{ep['episode_transcript']}\n"
            combined_prompt += f"--- EPISODE END ---\n\n"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=combined_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
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
            
            # If the model still outputs markdown blocks despite strict mode (rare but possible)
            if text.startswith("```"):
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                else:
                    text = text.split("```")[1].split("```")[0]
            
            text = text.strip()

            # Parse JSON response
            # Use strict=False to allow control characters like newlines inside strings
            try:
                mentions = json.loads(text, strict=False)
            except json.JSONDecodeError as e:
                # Fallback: try regex if direct parsing fails
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
                if json_match:
                    try:
                        mentions = json.loads(json_match.group(0), strict=False)
                    except:
                        print(f"JSON parsing error (even with regex): {e}")
                        print(f"Problematic text snippet: {text[:200]}...{text[-200:]}")
                        return []
                else:
                    print(f"JSON parsing error: {e}")
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
