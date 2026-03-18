import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm
from pipeline.loader import PodcastLoader
from pipeline.extractor import BookExtractor
from pipeline.verifier import BookVerifier
from pipeline.storage import BookStorage

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Podcast Book Mention Extraction Pipeline")
    parser.add_argument("--raw_text_dir", default=os.getenv("RAW_TEXT_DIR", "raw_text"), help="Directory containing raw JSON files")
    parser.add_argument("--output_file", default=os.getenv("OUTPUT_FILE", "book_mentions_research.csv"), help="Output CSV file")
    parser.add_argument("--db_file", default=os.getenv("DB_FILE", "book_mentions_research.db"), help="Output SQLite database file")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"), help="Gemini model to use")
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: GEMINI_API_KEY not found. Please set it in your .env file or pass it as an argument.")
        return

    # Initialize components
    loader = PodcastLoader(args.raw_text_dir)
    extractor = BookExtractor(args.api_key, args.model)
    verifier = BookVerifier(args.api_key, args.model)
    storage = BookStorage(args.output_file, args.db_file)
    
    # Get all JSON files in the raw_text directory
    json_files = loader.get_all_json_files()
    if not json_files:
        print(f"No JSON files found in '{args.raw_text_dir}'. Please add some Podscan JSON files.")
        return
    
    # Get already processed episode IDs to avoid re-processing
    processed_episode_ids = storage.get_processed_episodes()
    print(f"Found {len(processed_episode_ids)} already processed episodes.")
    
    # Process each JSON file
    for filename in tqdm(json_files, desc="Processing Podcast Files"):
        episodes = loader.load_episodes(filename)
        
        for ep in tqdm(episodes, desc=f"Extracting from {filename}", leave=False):
            episode_id = str(ep.get('episode_id', 'unknown'))
            episode_name = ep.get('episode_title', 'Untitled')
            transcript = ep.get('episode_transcript', '')
            
            # Skip if already processed
            if episode_id in processed_episode_ids:
                continue
            
            if not transcript:
                continue
            
            # 1. Extraction
            mentions = extractor.extract_mentions(transcript, episode_name, episode_id)
            
            if not mentions:
                # Still record that we processed this episode to avoid re-scanning empty ones
                # We can add a dummy entry or just handle it in storage
                # For now, let's just skip to next
                continue
            
            # 2. Verification
            verified_mentions = []
            for m in tqdm(mentions, desc="Verifying Mentions", leave=False):
                verified_m = verifier.verify_mention(m)
                verified_mentions.append(verified_m)
            
            # 3. Storage
            if verified_mentions:
                storage.save_to_csv(verified_mentions)
                storage.save_to_db(verified_mentions)
                
    print("\nExtraction and Verification Pipeline Complete!")
    print(f"Results saved to {args.output_file} and {args.db_file}")

if __name__ == "__main__":
    main()
