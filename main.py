import os
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

genai.configure(api_key=api_key)

SYSTEM_INSTRUCTION = """You are a precise information extraction system.
Extract books explicitly mentioned in a podcast episode transcript.
Rules:
1. Only extract books.
2. Exclude podcasts, movies, TV shows, articles, songs, newsletters, essays, papers, brands, and generic concepts.
3. Only include a book if it is explicitly named or very strongly supported by the transcript.
4. Set author to null unless explicitly stated in the transcript.
5. Use exact wording from the transcript when possible.
6. Do not fabricate quotations.
7. Return an empty list if no books are mentioned."""

def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    episodes = data if isinstance(data, list) else [data]
    print(f"Found {len(episodes)} episodes. Starting extraction...")

    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        system_instruction=SYSTEM_INSTRUCTION
    )

    results = []

    for ep in episodes:
        title = ep.get('title', 'Untitled')
        transcript = ep.get('transcript', '')
        
        if not transcript:
            print(f"Skipping {title}: No transcript.")
            continue

        print(f"Processing: {title}...")
        
        # Simple chunking
        chunk_size = 100000
        chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
        
        episode_mentions = []
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
            
            response = model.generate_content(
                chunk,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                )
            )
            
            try:
                chunk_data = json.loads(response.text)
                # Ensure it's a list
                if isinstance(chunk_data, list):
                    episode_mentions.extend(chunk_data)
                print(f"Done ({len(chunk_data) if isinstance(chunk_data, list) else 0} found)")
            except Exception as e:
                print(f"Error parsing JSON: {e}")

        results.append({
            "episode": title,
            "mentions": episode_mentions
        })

    output_file = f"extraction_results_{int(time.time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSuccess! Results saved to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <path-to-podcast-json>")
    else:
        process_file(sys.argv[1])
