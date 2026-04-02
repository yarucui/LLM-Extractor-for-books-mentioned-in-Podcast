import os
import argparse
import time
from dotenv import load_dotenv
from tqdm import tqdm
from pipeline.loader import PodcastLoader
from pipeline.extractor import BookExtractor
from pipeline.searcher import BookSearcher
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
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "google/gemini-3.1-pro-preview"), help="OpenRouter model to use")
    parser.add_argument("--api_key", default=os.getenv("GEMINI_API_KEY"), help="OpenRouter API key")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of episodes per batch")
    parser.add_argument("--rate_limit_delay", type=int, default=15, help="Seconds to wait between batches to avoid rate limits")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: GEMINI_API_KEY not found. Please set it in your .env file or pass it as an argument.")
        return

    # Diagnostic print (masked)
    masked_key = f"{args.api_key[:6]}...{args.api_key[-4:]}" if len(args.api_key) > 10 else "***"
    print(f"Using model: {args.model}")
    print(f"API Key detected: {masked_key}")

    # Initialize components
    loader = PodcastLoader(args.raw_text_dir)
    extractor = BookExtractor(args.api_key, args.model)
    searcher = BookSearcher(args.api_key, args.model)
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
        
        # Filter out already processed episodes
        pending_episodes = [
            ep for ep in episodes 
            if str(ep.get('episode_id', 'unknown')) not in processed_episode_ids and ep.get('episode_transcript')
        ]
        
        if not pending_episodes:
            continue

        # Process in batches
        for i in range(0, len(pending_episodes), args.batch_size):
            batch = pending_episodes[i:i + args.batch_size]
            
            print(f"\nProcessing batch {i//args.batch_size + 1} ({len(batch)} episodes)...")
            
            # 1. Extraction (Batch)
            mentions = extractor.extract_mentions_batch(batch)
            
            if mentions:
                print(f"Found {len(mentions)} potential book mentions.")
                
                # 2. Search & Verification (Individual)
                final_mentions = []
                try:
                    for m in tqdm(mentions, desc="Processing Mentions", leave=False):
                        try:
                            # 2a. Dedicated Search Agent for Goodreads URL
                            search_res = searcher.search_goodreads(m.get('book_name', ''), m.get('author_name'))
                            m['goodreads_url'] = search_res.get('goodreads_url')
                            m['search_query_used'] = search_res.get('search_query_used', '')
                            
                            # 2b. Audit/Verification
                            final_m = verifier.verify_mention(m)
                            final_mentions.append(final_m)
                        except Exception as ve:
                            print(f"Error processing mention: {ve}")
                            final_mentions.append(m)
                        # Small delay between calls to respect RPM
                        time.sleep(2) 
                except KeyboardInterrupt:
                    print("\nProcessing interrupted by user. Saving progress so far...")
                
                # 3. Storage
                if final_mentions:
                    storage.save_to_csv(final_mentions)
                    storage.save_to_db(final_mentions)
            elif mentions == []:
                # This could be either no mentions found OR extraction failed after retries
                # The extractor prints its own error messages, so we just continue
                pass
            
            # Rate limiting delay between batches
            if i + args.batch_size < len(pending_episodes):
                print(f"Waiting {args.rate_limit_delay}s for rate limits...")
                time.sleep(args.rate_limit_delay)
                
    print("\nExtraction and Verification Pipeline Complete!")
    print(f"Results saved to {args.output_file} and {args.db_file}")

if __name__ == "__main__":
    main()
