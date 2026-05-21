"""
Goodreads URL Rematcher & Verifier

A standalone script that re-processes book mentions from the main pipeline's CSV output,
rematching failed or missing Goodreads URLs using web-search-enabled LLMs, then verifying
each URL by scraping the Goodreads page and comparing metadata.

Supports two search backends (both via OpenRouter):
  - gemini:     Gemini model with :online suffix (web search)
  - perplexity: Perplexity sonar model (native web search)

Usage:
  python rematch_goodreads.py --input book_mentions_research.csv --backend gemini
  python rematch_goodreads.py --input book_mentions_research.csv --backend perplexity --mode all
"""

import os
import sys
import argparse
import time
import json
import re
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from pipeline.cache import BookCache
from pipeline.scraper import GoodreadsScraper
from pipeline.utils import safe_json_loads, TokenTracker


# ── Pydantic schemas for structured LLM output ──────────────────────────

class GoodreadsSearchResult(BaseModel):
    goodreads_url: Optional[str] = Field(description="The official Goodreads URL for the book. Must contain /book/show/.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 that this is the correct book.")
    search_query_used: str = Field(description="The search query used to find this URL.")

class MetadataVerification(BaseModel):
    is_match: bool = Field(description="True if the scraped page matches the expected book and author.")
    reason: str = Field(description="Brief explanation of the match/mismatch.")


# ── Search backend implementations ──────────────────────────────────────

class GeminiSearchBackend:
    """Uses OpenRouter with :online suffix for web search."""

    def __init__(self, api_key: str, model: str):
        api_key = api_key.strip().strip('"').strip("'")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Goodreads Rematcher",
            }
        )
        self.model_name = model if model.endswith(":online") else f"{model}:online"
        self.name = "gemini"

    def search(self, book_name: str, author_name: Optional[str], exclude_urls: List[str], max_retries: int = 5) -> Dict[str, Any]:
        query_hint = f"'{book_name}' by {author_name}" if author_name else f"'{book_name}'"
        prompt = (
            f"Find the official Goodreads page URL for the book: {query_hint}.\n\n"
            f"IMPORTANT: The title above was extracted from a podcast transcript and may be inaccurate, "
            f"paraphrased, or a subtitle rather than the real title. If you cannot find this exact title, "
            f"search for books by the same author on Goodreads and find the one that best matches the topic "
            f"or keywords in the given title.\n\n"
            f"Try multiple search queries if needed:\n"
            f"  1. '{book_name} {author_name or ''} goodreads'\n"
            f"  2. '{author_name or ''} book goodreads' (search by author)\n"
            f"  3. Key words from the title + author + goodreads\n\n"
            f"Return the URL containing '/book/show/'. If you cannot find one, return null."
        )
        if exclude_urls:
            prompt += f"\n\nDO NOT return any of these URLs (already rejected): {', '.join(exclude_urls)}"

        system = (
            "You are a search agent that finds official Goodreads book page URLs.\n"
            "RULES:\n"
            "- Only return URLs matching https://www.goodreads.com/book/show/... or https://www.goodreads.com/en/book/show/...\n"
            "- NEVER guess or construct a URL. Only use URLs you find in search results.\n"
            "- The book title from podcast extraction may be wrong — prioritize finding the correct book by the given author.\n"
            "- If unsure, return null for goodreads_url.\n"
            "- Accuracy over coverage."
        )

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "goodreads_search",
                            "strict": True,
                            "schema": GoodreadsSearchResult.model_json_schema(),
                        }
                    },
                )
                usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
                data = safe_json_loads(resp.choices[0].message.content)
                if data and isinstance(data, dict):
                    return {"result": data, "usage": usage}
                return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": usage}
            except Exception as e:
                if _is_rate_limit(e):
                    wait = _parse_retry_delay(e)
                    print(f"  Rate limit (gemini). Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Gemini search error: {e}")
                    return _empty_search_result()
        return _empty_search_result()


class PerplexitySearchBackend:
    """Uses Perplexity sonar model via OpenRouter for native web search."""

    def __init__(self, api_key: str, model: str):
        api_key = api_key.strip().strip('"').strip("'")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Goodreads Rematcher (Perplexity)",
            }
        )
        self.model_name = model
        self.name = "perplexity"

    def search(self, book_name: str, author_name: Optional[str], exclude_urls: List[str], max_retries: int = 5) -> Dict[str, Any]:
        query_hint = f"'{book_name}' by {author_name}" if author_name else f"'{book_name}'"
        prompt = (
            f"Find the official Goodreads page URL for the book: {query_hint}.\n\n"
            f"IMPORTANT: The title above was extracted from a podcast transcript and may be inaccurate, "
            f"paraphrased, or a subtitle rather than the real title. If you cannot find this exact title, "
            f"search for books by the same author on Goodreads and find the one that best matches the topic "
            f"or keywords in the given title.\n\n"
            f"I need the URL in the format: https://www.goodreads.com/book/show/[ID]-[slug]\n"
            f"Search for it and return the exact URL from the search results."
        )
        if exclude_urls:
            prompt += f"\n\nDO NOT return any of these URLs (already rejected): {', '.join(exclude_urls)}"

        system = (
            "You are a search agent that finds official Goodreads book page URLs.\n"
            "RULES:\n"
            "- Only return URLs matching https://www.goodreads.com/book/show/... or https://www.goodreads.com/en/book/show/...\n"
            "- NEVER guess or fabricate a URL. Only use URLs you actually find.\n"
            "- The book title from podcast extraction may be wrong — prioritize finding the correct book by the given author.\n"
            "- If you cannot find a definitive match, return null for goodreads_url.\n"
            "- Return your answer as JSON with fields: goodreads_url, confidence (0-1), search_query_used."
        )

        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
                raw = resp.choices[0].message.content
                data = safe_json_loads(raw)

                if data and isinstance(data, dict):
                    url = data.get("goodreads_url") or data.get("url")
                    return {
                        "result": {
                            "goodreads_url": url,
                            "confidence": data.get("confidence", 0.5),
                            "search_query_used": data.get("search_query_used", ""),
                        },
                        "usage": usage,
                    }

                # Perplexity may return prose instead of JSON — extract URL with regex
                url_match = re.search(r'https?://(?:www\.)?goodreads\.com/(?:en/)?book/show/[^\s\)\"\']+', raw or "")
                if url_match:
                    return {
                        "result": {"goodreads_url": url_match.group(0).rstrip(".,;)"), "confidence": 0.5, "search_query_used": ""},
                        "usage": usage,
                    }

                return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": usage}
            except Exception as e:
                if _is_rate_limit(e):
                    wait = _parse_retry_delay(e)
                    print(f"  Rate limit (perplexity). Retry {attempt+1}/{max_retries}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Perplexity search error: {e}")
                    return _empty_search_result()
        return _empty_search_result()


# ── Metadata verification via LLM ───────────────────────────────────────

class MetadataVerifier:
    """LLM-based fuzzy comparison of scraped vs expected book metadata. Uses OpenRouter."""

    def __init__(self, api_key: str, model: str):
        api_key = api_key.strip().strip('"').strip("'")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
                "X-OpenRouter-Title": "Goodreads Rematcher Verifier",
            }
        )
        self.model_name = model.replace(":online", "")

    def verify(self, scraped_title: Optional[str], scraped_author: Optional[str],
               expected_title: str, expected_author: Optional[str]) -> Dict[str, Any]:
        if not scraped_title:
            return {
                "result": {"is_match": False, "reason": "Scraped title is empty"},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        prompt = (
            f"EXPECTED (from podcast transcript — title may be inaccurate):\n"
            f"  Title: {expected_title}\n"
            f"  Author: {expected_author or 'Unknown'}\n\n"
            f"SCRAPED (from Goodreads page):\n"
            f"  Title: {scraped_title}\n"
            f"  Author: {scraped_author or 'Unknown'}\n\n"
            f"Is the Goodreads book a plausible match for what was discussed in the podcast?\n\n"
            f"MATCHING RULES:\n"
            f"- The podcast title may be wrong, paraphrased, or a subtitle. Titles do NOT need to match exactly.\n"
            f"- If the AUTHOR matches and the book topic/keywords are related, it IS a match.\n"
            f"- Allow fuzzy matching: subtitles, title variations, transliterations, edition differences.\n"
            f"- Only reject if it is clearly a completely different book (different author AND unrelated topic).\n\n"
            f"Reply as JSON with fields: is_match (bool), reason (string)."
        )
        system = "You are a metadata auditor matching podcast book mentions to Goodreads pages. The podcast-extracted title is often inaccurate."

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "metadata_verification",
                        "strict": True,
                        "schema": MetadataVerification.model_json_schema(),
                    }
                },
            )
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
            data = safe_json_loads(resp.choices[0].message.content)
            if data and isinstance(data, dict):
                return {"result": data, "usage": usage}
            return {"result": {"is_match": False, "reason": "Failed to parse response"}, "usage": usage}
        except Exception as e:
            print(f"  Verifier error: {e}")
            return {"result": {"is_match": False, "reason": str(e)}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


# ── Helpers ──────────────────────────────────────────────────────────────

def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s

def _parse_retry_delay(e: Exception, default: float = 60) -> float:
    m = re.search(r"retry in (\d+\.?\d*)s", str(e).lower())
    return float(m.group(1)) + 2 if m else default

def _empty_search_result() -> Dict[str, Any]:
    return {"result": {"goodreads_url": None, "confidence": 0, "search_query_used": ""}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

def _normalize_goodreads_url(url: str) -> str:
    """Normalize Goodreads URL variants to the canonical /book/show/ format.
    e.g. goodreads.com/en/book/show/123 -> goodreads.com/book/show/123"""
    if not url:
        return url
    return re.sub(r'goodreads\.com/en/book/show/', 'goodreads.com/book/show/', url)

def _clean_author(value: Any) -> Optional[str]:
    """Convert pandas NaN / 'nan' / empty strings to None."""
    if value is None:
        return None
    s = str(value).strip()
    if s.lower() in ("nan", "none", ""):
        return None
    return s


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Rematch and verify Goodreads URLs for extracted book mentions")
    parser.add_argument("--input", required=True, help="Input CSV file from the main pipeline")
    parser.add_argument("--output", help="Output CSV file (default: <input>_rematched.csv)")
    parser.add_argument("--backend", choices=["gemini", "perplexity"], default="gemini", help="Search backend to use")
    parser.add_argument("--mode", choices=["failed", "all"], default="failed",
                        help="'failed' = only rematch rows with missing/unverified URLs; 'all' = rematch every row")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max different URLs to try per book")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay in seconds between books")

    parser.add_argument("--cache_file", default="book_cache.json", help="Path to book cache JSON file (default: book_cache.json)")

    # Backend-specific options
    parser.add_argument("--api_key", help="API key (or set GEMINI_API_KEY in .env)")
    parser.add_argument("--model", help="Model name override")
    args = parser.parse_args()

    # ── Resolve API key & model (both backends use OpenRouter) ──
    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("Error: No API key. Set GEMINI_API_KEY in .env or pass --api_key.")
        sys.exit(1)

    if args.backend == "gemini":
        model = args.model or os.getenv("SEARCHER_MODEL", os.getenv("GEMINI_MODEL", "google/gemini-3-flash-preview"))
        inspector_model = os.getenv("INSPECTOR_MODEL", os.getenv("GEMINI_MODEL", "google/gemini-3-flash-preview"))
    else:
        model = args.model or "perplexity/sonar"
        inspector_model = os.getenv("INSPECTOR_MODEL", os.getenv("GEMINI_MODEL", "google/gemini-3-flash-preview"))

    output_file = args.output or args.input.replace(".csv", "_rematched.csv")

    # ── Load data ──
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {args.input}")

    # ── Filter rows to process ──
    if args.mode == "failed":
        mask = (
            df["goodreads_url"].isna()
            | (df["goodreads_url"] == "")
            | (df["url_verified"] != True)
        )
        target_indices = df.index[mask].tolist()
        print(f"Mode=failed: {len(target_indices)} rows need rematching ({len(df) - len(target_indices)} already verified)")
    else:
        target_indices = df.index.tolist()
        print(f"Mode=all: rematching all {len(target_indices)} rows")

    if not target_indices:
        print("Nothing to rematch. All URLs are already verified.")
        sys.exit(0)

    # ── Initialize components ──
    if args.backend == "gemini":
        searcher = GeminiSearchBackend(api_key, model)
    else:
        searcher = PerplexitySearchBackend(api_key, model)

    verifier = MetadataVerifier(api_key, inspector_model)
    scraper = GoodreadsScraper()
    tracker = TokenTracker(model)
    cache = BookCache(args.cache_file)

    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"Backend:  {args.backend}")
    print(f"Model:    {searcher.model_name}")
    print(f"API Key:  {masked_key}")
    print(f"Output:   {output_file}")
    print(f"Cache:    {args.cache_file} ({len(cache)} entries)")
    print()

    # ── Ensure output columns exist ──
    for col in ["goodreads_url", "url_verified", "scraped_book_name", "scraped_author_name", "rejection_reason", "rematch_backend"]:
        if col not in df.columns:
            df[col] = ""

    # ── Process each target row ──
    success_count = 0
    fail_count = 0
    cache_hit_count = 0

    try:
        for i, idx in enumerate(target_indices):
            row = df.loc[idx]
            book_name = str(row.get("book_name", "")).strip()
            author_name = _clean_author(row.get("author_name"))

            print(f"[{i+1}/{len(target_indices)}] {book_name}" + (f" by {author_name}" if author_name else ""))

            # ── Cache lookup ──
            cached = cache.get(book_name, author_name)
            if cached and cached.get("goodreads_url"):
                url = cached["goodreads_url"]
                print(f"  CACHE HIT: {url}")
                df.at[idx, "goodreads_url"] = url
                df.at[idx, "url_verified"] = True
                df.at[idx, "scraped_book_name"] = cached.get("scraped_book_name", "")
                df.at[idx, "scraped_author_name"] = cached.get("scraped_author_name", "")
                df.at[idx, "rejection_reason"] = ""
                df.at[idx, "rematch_backend"] = "cache"
                success_count += 1
                cache_hit_count += 1
                continue

            # ── Search → Scrape → Verify loop ──
            exclude_urls = []
            matched = False

            for attempt in range(args.max_attempts):
                # 1. Search for Goodreads URL
                search_data = searcher.search(book_name, author_name, exclude_urls)
                tracker.add_usage(
                    search_data["usage"]["prompt_tokens"],
                    search_data["usage"]["completion_tokens"],
                    model_name=searcher.model_name,
                )

                url = search_data["result"].get("goodreads_url")
                if not url:
                    print(f"  Attempt {attempt+1}: No URL returned")
                    df.at[idx, "rejection_reason"] = "Searcher returned no URL"
                    break

                url = _normalize_goodreads_url(url)

                if url in exclude_urls:
                    print(f"  Attempt {attempt+1}: Returned excluded URL, skipping")
                    continue

                print(f"  Attempt {attempt+1}: Scraping {url}")

                # 2. Scrape the Goodreads page
                scrape_res = scraper.scrape_book_metadata(url)
                if scrape_res.get("error"):
                    print(f"  Scrape failed: {scrape_res['error']}")
                    df.at[idx, "rejection_reason"] = f"Scrape error: {scrape_res['error']}"
                    exclude_urls.append(url)
                    continue

                scraped_title = scrape_res.get("title")
                scraped_author = scrape_res.get("author")
                print(f"  Scraped: \"{scraped_title}\" by {scraped_author}")

                # 3. Verify metadata match via LLM
                verify_data = verifier.verify(scraped_title, scraped_author, book_name, author_name)
                tracker.add_usage(
                    verify_data["usage"]["prompt_tokens"],
                    verify_data["usage"]["completion_tokens"],
                    model_name=verifier.model_name,
                )

                if verify_data["result"].get("is_match"):
                    print(f"  VERIFIED: {url}")
                    df.at[idx, "goodreads_url"] = url
                    df.at[idx, "url_verified"] = True
                    df.at[idx, "scraped_book_name"] = scraped_title or ""
                    df.at[idx, "scraped_author_name"] = scraped_author or ""
                    df.at[idx, "rejection_reason"] = ""
                    df.at[idx, "rematch_backend"] = args.backend
                    matched = True
                    success_count += 1

                    # ── Write to cache ──
                    cache.put(book_name, author_name, {
                        "goodreads_url": url,
                        "scraped_book_name": scraped_title or "",
                        "scraped_author_name": scraped_author or "",
                        "book_name": book_name,
                        "author_name": author_name,
                    })
                    break
                else:
                    reason = verify_data["result"].get("reason", "Unknown")
                    print(f"  Rejected: {reason}")
                    df.at[idx, "rejection_reason"] = reason
                    exclude_urls.append(url)

            if not matched:
                fail_count += 1

            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")

    # ── Save results ──
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {output_file}")

    json_output = output_file.replace(".csv", ".json")
    df.to_json(json_output, orient="records", force_ascii=False, indent=2)
    print(f"Saved to {json_output}")

    # ── Report ──
    print(f"\n{'='*50}")
    print(f"REMATCH SUMMARY")
    print(f"{'='*50}")
    print(f"Total processed: {success_count + fail_count}")
    print(f"Cache hits:      {cache_hit_count} (free)")
    print(f"Verified (new):  {success_count - cache_hit_count}")
    print(f"Still failed:    {fail_count}")
    print(f"Cache size:      {len(cache)} entries")
    print(tracker.get_report())


if __name__ == "__main__":
    main()
