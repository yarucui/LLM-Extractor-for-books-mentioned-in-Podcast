import pandas as pd
import sqlite3
import os
import json
from typing import List, Dict, Any

class BookStorage:
    def __init__(self, output_file: str = "book_mentions_research.csv", db_file: str = "book_mentions_research.db"):
        self.output_file = output_file
        self.db_file = db_file

    def save_to_json(self, mentions: List[Dict[str, Any]]):
        """
        Saves a list of book mentions to a JSON file.
        Appends to the file if it already exists.
        Includes deduplication based on context_quote.
        """
        json_file = self.output_file.replace('.csv', '.json')
        
        existing_data = []
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = []
        
        # Combine
        combined_data = existing_data + mentions
        
        # Deduplicate
        seen = set()
        unique_data = []
        for item in combined_data:
            key = (item.get('context_quote', ''), item.get('book_name', ''))
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(mentions)} mentions to {json_file} (Deduplicated)")

    def save_to_csv(self, mentions: List[Dict[str, Any]]):
        """
        Saves a list of book mentions to a CSV file.
        Appends to the file if it already exists, handling new columns.
        Uses utf-8-sig encoding to prevent Mojibake in Excel.
        Includes deduplication based on context_quote.
        """
        df_new = pd.DataFrame(mentions)
        if not os.path.exists(self.output_file):
            df_combined = df_new
        else:
            df_old = pd.read_csv(self.output_file, encoding='utf-8-sig')
            # Combine old and new, ensuring all columns are present
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        
        # Deduplicate based on context_quote and book_name
        if not df_combined.empty:
            df_combined = df_combined.drop_duplicates(subset=['context_quote', 'book_name'], keep='first')
            
        df_combined.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        print(f"Saved {len(mentions)} mentions to {self.output_file} (Deduplicated)")

    def save_to_db(self, mentions: List[Dict[str, Any]]):
        """
        Saves a list of book mentions to a SQLite database.
        Handles schema evolution by merging with existing data if the table exists.
        """
        conn = sqlite3.connect(self.db_file)
        df_new = pd.DataFrame(mentions)
        
        try:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='book_mentions'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                df_new.to_sql('book_mentions', conn, index=False)
            else:
                # Read existing data
                df_old = pd.read_sql('SELECT * FROM book_mentions', conn)
                # Combine and handle missing columns
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                # Overwrite the table with the combined data
                df_combined.to_sql('book_mentions', conn, if_exists='replace', index=False)
        except Exception as e:
            print(f"Error saving to database: {e}")
            # Fallback: try to just append if something went wrong with the merge
            try:
                df_new.to_sql('book_mentions', conn, if_exists='append', index=False)
            except Exception:
                pass
        finally:
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
            df = pd.read_csv(self.output_file, encoding='utf-8-sig')
            if 'episode_id' in df.columns:
                return df['episode_id'].unique().tolist()
            return []
        except Exception:
            return []
