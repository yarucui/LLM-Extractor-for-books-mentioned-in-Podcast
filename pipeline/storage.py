import pandas as pd
import sqlite3
import os
from typing import List, Dict, Any

class BookStorage:
    def __init__(self, output_file: str = "book_mentions_research.csv", db_file: str = "book_mentions_research.db"):
        self.output_file = output_file
        self.db_file = db_file

    def save_to_csv(self, mentions: List[Dict[str, Any]]):
        """
        Saves a list of book mentions to a CSV file.
        Appends to the file if it already exists.
        """
        df = pd.DataFrame(mentions)
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file, index=False)
        else:
            df.to_csv(self.output_file, mode='a', header=False, index=False)
        print(f"Saved {len(mentions)} mentions to {self.output_file}")

    def save_to_db(self, mentions: List[Dict[str, Any]]):
        """
        Saves a list of book mentions to a SQLite database.
        """
        conn = sqlite3.connect(self.db_file)
        df = pd.DataFrame(mentions)
        df.to_sql('book_mentions', conn, if_exists='append', index=False)
        conn.close()
        print(f"Saved {len(mentions)} mentions to {self.db_file}")

    def get_processed_episodes(self) -> List[str]:
        """
        Returns a list of already processed episode IDs from the CSV file.
        Used to avoid re-processing.
        """
        if not os.path.exists(self.output_file):
            return []
        
        try:
            df = pd.read_csv(self.output_file)
            if 'episode_id' in df.columns:
                return df['episode_id'].unique().tolist()
            return []
        except Exception:
            return []
