import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from .utils import count_words

class BookExtractor:
    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite-preview"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.system_instruction = """You are a precise research assistant specializing in podcast analysis.
Your task is to extract book mentions from podcast transcripts.

For each book mention, extract:
1. book_name: The title of the book.
2. author_name: The name of the author (if mentioned).
3. context_quote: A substantial quote from the transcript providing context (at least 2-3 sentences around the mention).
4. mention_type: The nature of the mention (e.g., 'Recommendation', 'Critique', 'Casual Mention', 'Author Interview', 'Reference').
5. recommend_intensity: A scale from 'Critical' to 'Strong Recommendation' (e.g., 'Critical', 'Negative', 'Neutral', 'Positive', 'Strong Recommendation').
6. author_present: Boolean (True if the author is a guest on the episode, False otherwise).

Return a JSON list of objects. If no books are mentioned, return an empty list [].
Do not include podcasts, movies, or TV shows. Only books."""

    def extract_mentions(self, transcript: str, episode_name: str, episode_id: str) -> List[Dict[Dict[str, Any], Any]]:
        """
        Extracts book mentions from a transcript.
        Handles chunking if the transcript is too long.
        """
        # Simple chunking logic (e.g., 50k characters per chunk for research context)
        chunk_size = 50000
        chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
        
        all_mentions = []
        
        for i, chunk in enumerate(chunks):
            prompt = f"Transcript Chunk {i+1}/{len(chunks)}:\n\n{chunk}"
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                    ),
                    system_instruction=self.system_instruction
                )
                
                # Parse JSON response
                mentions = json.loads(response.text)
                if isinstance(mentions, list):
                    for m in mentions:
                        # Add metadata
                        m['episode_name'] = episode_name
                        m['episode_id'] = episode_id
                        # Calculate word count of the context quote
                        m['word_count'] = count_words(m.get('context_quote', ''))
                        all_mentions.append(m)
                
            except Exception as e:
                print(f"Error extracting from chunk {i+1}: {e}")
                
        return all_mentions
