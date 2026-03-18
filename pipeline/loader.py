import json
import os
from typing import List, Dict, Any

class PodcastLoader:
    def __init__(self, raw_text_dir: str):
        self.raw_text_dir = raw_text_dir

    def get_all_json_files(self) -> List[str]:
        """
        Returns a list of all JSON files in the raw_text directory.
        """
        if not os.path.exists(self.raw_text_dir):
            print(f"Warning: Directory '{self.raw_text_dir}' not found.")
            return []
        
        return [f for f in os.listdir(self.raw_text_dir) if f.endswith('.json')]

    def load_episodes(self, filename: str) -> List[Dict[str, Any]]:
        """
        Loads episodes from a single JSON file.
        Handles both a list of episodes or a podcast object with an 'episodes' key.
        """
        filepath = os.path.join(self.raw_text_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If data is a list, assume it's a list of episodes
            if isinstance(data, list):
                return data
            
            # If data is a dict, check for 'episodes' key (common in Podscan)
            if isinstance(data, dict):
                if 'episodes' in data:
                    return data['episodes']
                # If no 'episodes' key, maybe it's a single episode object
                return [data]
            
            return []
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []
