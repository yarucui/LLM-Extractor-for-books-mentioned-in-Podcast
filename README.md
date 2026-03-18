# Podcast Book Mention Research Pipeline

A research-oriented Python pipeline to extract and verify book mentions from podcast transcripts using Gemini 2.5 Flash Lite.

## Features
- **Extraction**: Extracts book name, author, context quote, mention type, recommendation intensity, and author presence.
- **Verification**: A second LLM pass to verify if it's a book, hallucination check, and accuracy of extraction.
- **Token Counting**: Calculates the token length of each context quote.
- **Storage**: Saves results to both CSV and SQLite database for research analysis.
- **Resilience**: Skips already processed episodes to avoid redundant API calls.

## Prerequisites
- Python 3.8+
- Gemini API Key

## Setup

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure environment variables**:
    Create a `.env` file in the root directory and add your Gemini API key:
    ```env
    GEMINI_API_KEY=your_api_key_here
    GEMINI_MODEL=gemini-3.1-flash-lite-preview
    RAW_TEXT_DIR=raw_text
    OUTPUT_FILE=book_mentions_research.csv
    DB_FILE=book_mentions_research.db
    ```
    Note: The pipeline uses the new **Google GenAI SDK** (`google-genai`).

3.  **Prepare your data**:
    Place your Podscan JSON files in the `raw_text` directory. The pipeline handles both a list of episodes or a podcast object with an `episodes` key.

## Usage

Run the pipeline from your terminal:
```bash
python main.py
```

### Command Line Arguments
You can also pass arguments directly:
```bash
python main.py --raw_text_dir my_data --output_file results.csv --model gemini-3.1-flash-lite-preview
```

## Output
- `book_mentions_research.csv`: A CSV file containing all verified book mentions.
- `book_mentions_research.db`: A SQLite database with a `book_mentions` table.

## Research Data Fields
- `book_name`: Title of the book.
- `author_name`: Name of the author.
- `episode_name`: Title of the podcast episode.
- `episode_id`: Unique ID of the episode.
- `context_quote`: A substantial quote providing context.
- `mention_type`: Nature of the mention (Recommendation, Critique, etc.).
- `recommend_intensity`: Scale from 'Critical' to 'Strong Recommendation'.
- `author_present`: Boolean (True if author is a guest).
- `word_count`: Number of words in the context quote.
- `is_book`: Verification result (Is it a book?).
- `author_correct`: Verification result (Is the author correct?).
- `mention_type_correct`: Verification result (Is the mention type accurate?).
- `intensity_appropriate`: Verification result (Is the intensity appropriate?).
- `author_present_correct`: Verification result (Is the author presence correct?).
- `verification_notes`: Additional notes from the verifier LLM.
