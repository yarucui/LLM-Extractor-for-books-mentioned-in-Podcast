"""
benchmark_search.py — Compare web-search LLMs on finding correct Goodreads URLs.

The pipeline's SEARCH stage needs a stronger model than the cheap extractor.
This script benchmarks several web-search-capable models (all via OpenRouter)
against a small, hand-labeled gold set, so you can pick the best one on a
clear accuracy / cost / latency basis.

Scoring uses a HUMAN GOLD STANDARD: you label the correct Goodreads URL for a
small set of books, and each model's returned URL is compared by Goodreads
*book ID* (the number in /book/show/<ID>-slug). ID match is robust to slug,
language-prefix (/en/), query-string and edition-slug differences. No scraping
needed.

Two subcommands
---------------
  make-goldset : sample a diverse set of book mentions from the pipeline's
                 existing CSV into benchmark/goldset.csv. You then fill in the
                 `gold_url` column by hand (leave a row's gold_url blank to skip
                 it — e.g. if you decide it isn't really a book).

  run          : for every labeled book, ask each candidate model for a
                 Goodreads URL, score it against gold_url, and write a summary
                 plus a per-(book,model) detail CSV under benchmark/.

All models are called through OpenRouter using GEMINI_API_KEY from .env.

Usage
-----
  python benchmark_search.py make-goldset --input book_mentions_research.csv -n 25
  #  ...hand-label the gold_url column in benchmark/goldset.csv...
  python benchmark_search.py run
  python benchmark_search.py run --models perplexity/sonar,google/gemini-2.5-pro:online
"""

import argparse
import io
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from pipeline.utils import safe_json_loads

# Force UTF-8 stdout on Windows so non-ASCII book titles don't die on cp936.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
else:  # pragma: no cover
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ── Candidate models (label → OpenRouter model id) ──────────────────────────
# For non-search models (plain Gemini/GPT) append ':online' to enable web search.
# Perplexity sonar models have native web search and need no suffix.
CANDIDATE_MODELS: List[str] = [
    "perplexity/sonar",
    "perplexity/sonar-pro",
    "google/gemini-2.5-pro:online",
    "openai/gpt-4o-search-preview",
]

# Approximate OpenRouter pricing, USD per 1M tokens: (input, output).
# Token-based only — Perplexity also bills a per-request search fee not captured here.
PRICES: Dict[str, tuple] = {
    "perplexity/sonar": (1.0, 1.0),
    "perplexity/sonar-pro": (3.0, 15.0),
    "google/gemini-2.5-pro:online": (1.25, 10.0),
    "google/gemini-2.5-pro": (1.25, 10.0),
    "openai/gpt-4o-search-preview": (2.5, 10.0),
}


# ── Small shared helpers ────────────────────────────────────────────────────

def extract_usage(response) -> Dict[str, int]:
    """Defensively read token counts from an OpenAI/OpenRouter response.

    Inlined from pipeline.utils so this benchmark stays standalone — it does not
    depend on uncommitted pipeline changes.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def _clean_str(value: Any) -> str:
    """pandas NaN / 'nan' / 'none' → '' ; otherwise the trimmed string."""
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _goodreads_id(url: Optional[str]) -> Optional[str]:
    """Extract the canonical book ID from a Goodreads URL.

    Handles /book/show/<ID>-slug and the older /book/show/<ID>.Slug form, plus
    the optional /en/ language prefix. The numeric ID *is* the book's identity.
    """
    if not url:
        return None
    m = re.search(r"goodreads\.com/(?:[a-z]{2}/)?book/show/(\d+)", str(url))
    return m.group(1) if m else None


def _extract_goodreads_url(raw: str) -> Optional[str]:
    """Pull a Goodreads /book/show/ URL out of an LLM response.

    Prefers a structured JSON field; falls back to a regex over free prose
    (Perplexity and the search-preview models often answer in prose).
    """
    if not raw:
        return None

    data = safe_json_loads(raw)
    if isinstance(data, dict):
        for key in ("goodreads_url", "url", "goodreads"):
            val = data.get(key)
            if val and _goodreads_id(val):
                return str(val).strip()

    m = re.search(r"https?://(?:www\.)?goodreads\.com/(?:[a-z]{2}/)?book/show/\d+[^\s\)\"'<>]*", raw)
    return m.group(0).rstrip(".,;)]") if m else None


def _make_client() -> OpenAI:
    load_dotenv()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        print("Error: GEMINI_API_KEY (OpenRouter key) not found in .env.")
        sys.exit(1)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("APP_URL", "https://ai.studio/build"),
            "X-OpenRouter-Title": "Goodreads Search Benchmark",
        },
    )


# ── The thing under test: one search call ───────────────────────────────────

_SEARCH_SYSTEM = (
    "You are a search agent that finds the official Goodreads book page URL for a book.\n"
    "RULES:\n"
    "- Only return a URL of the form https://www.goodreads.com/book/show/<ID>-<slug>.\n"
    "- NEVER guess or fabricate the numeric ID — only use a URL you actually find via search.\n"
    "- The title may be paraphrased or wrong (it came from a podcast transcript). If you can't\n"
    "  find that exact title, find the book by the same author that best matches the topic.\n"
    "- If you cannot find a confident match, return null.\n"
    'Respond as JSON: {"goodreads_url": <url or null>, "confidence": <0-1>}.'
)


def search_once(client: OpenAI, model: str, book_name: str, author_name: str,
                max_retries: int = 4) -> Dict[str, Any]:
    """Run a single search. Returns url / raw / usage / latency / error.

    No response_format is sent: several web-search models reject structured
    outputs, and `_extract_goodreads_url` parses prose or JSON either way.
    """
    who = f"'{book_name}' by {author_name}" if author_name else f"'{book_name}'"
    user = (
        f"Find the official Goodreads page URL for the book: {who}.\n"
        f"Return the URL containing '/book/show/'. If you cannot find a confident match, return null."
    )

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SEARCH_SYSTEM},
                    {"role": "user", "content": user},
                ],
            )
            latency = time.time() - t0
            raw = (resp.choices[0].message.content if resp.choices else "") or ""
            return {
                "url": _extract_goodreads_url(raw),
                "raw": raw,
                "usage": extract_usage(resp),
                "latency": latency,
                "error": "",
            }
        except Exception as e:  # noqa: BLE001 — benchmark must survive any single failure
            s = str(e).lower()
            if "429" in s or "rate limit" in s or "too many requests" in s:
                m = re.search(r"retry in (\d+\.?\d*)s", s)
                wait = (float(m.group(1)) + 2) if m else 30
                print(f"    rate-limit on {model}, retry {attempt+1}/{max_retries} in {wait:.0f}s")
                time.sleep(wait)
                continue
            return {"url": None, "raw": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "latency": 0.0, "error": str(e)[:200]}
    return {"url": None, "raw": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "latency": 0.0, "error": "max retries (rate limit)"}


def score(returned_url: Optional[str], gold_url: str) -> str:
    """correct | wrong | missing  (gold rows with no ID are filtered out earlier)."""
    rid = _goodreads_id(returned_url)
    if rid is None:
        return "missing"
    return "correct" if rid == _goodreads_id(gold_url) else "wrong"


# ── Subcommand: make-goldset ────────────────────────────────────────────────

def make_goldset(args) -> None:
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df["author_name"] = df["author_name"].map(_clean_str)
    df["book_name"] = df["book_name"].map(_clean_str)
    df = df[df["book_name"] != ""].copy()

    # One row per distinct book; prefer rows that already have an author + context.
    df["__key"] = df["book_name"].str.lower()
    df["__has_author"] = (df["author_name"] != "").astype(int)
    df = df.sort_values("__has_author", ascending=False).drop_duplicates("__key")

    is_book = df["is_book"].map(_truthy) if "is_book" in df.columns else pd.Series(True, index=df.index)
    verified = df["url_verified"].map(_truthy) if "url_verified" in df.columns else pd.Series(False, index=df.index)
    real = df[is_book]  # only label things the pipeline already thinks are books

    # Half from currently-verified (easy) + half from currently-failed (hard),
    # so the benchmark can actually discriminate between models.
    half = max(1, args.n // 2)
    easy = real[verified[real.index]].sample(min(half, int(verified[real.index].sum())),
                                             random_state=args.seed) if verified[real.index].any() else real.iloc[0:0]
    hard_pool = real[~verified[real.index]]
    hard = hard_pool.sample(min(args.n - len(easy), len(hard_pool)), random_state=args.seed)
    picked = pd.concat([easy, hard]).drop_duplicates("__key").head(args.n)

    out = pd.DataFrame({
        "book_name": picked["book_name"].values,
        "author_name": picked["author_name"].values,
        # ↓ two columns you fill in by hand:
        "is_book_gold": "",  # yes / no — is this REALLY a book (not a movie/show/doc)?  → validate ground truth
        "gold_url": "",      # correct Goodreads URL (only when is_book_gold == yes)    → search ground truth
        "context_quote": picked.get("context_quote", pd.Series("", index=picked.index)).map(
            lambda s: _clean_str(s)[:400]).values,
        "current_url_hint": picked.get("goodreads_url", pd.Series("", index=picked.index)).map(_clean_str).values,
        "current_verified": picked.get("url_verified", pd.Series("", index=picked.index)).map(_truthy).values,
    })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    n_hint = (out["current_url_hint"] != "").sum()
    print(f"Wrote {len(out)} books to {args.out}")
    print(f"  {len(easy)} currently-verified (easy) + {len(out) - len(easy)} currently-failed (hard)")
    print(f"  {n_hint} rows have a current_url_hint you can verify/correct into gold_url")
    print("\nNext: open the file and, for each row:")
    print("  1. is_book_gold = yes/no  (no for movies, shows, documentaries, fake listings)")
    print("  2. gold_url = correct Goodreads URL  (only when is_book_gold == yes)")
    print("Then run:  python benchmark_search.py run")


# ── Subcommand: run ─────────────────────────────────────────────────────────

def run(args) -> None:
    models = [m.strip() for m in args.models.split(",")] if args.models else CANDIDATE_MODELS

    if not os.path.exists(args.goldset):
        print(f"Error: gold set not found: {args.goldset}\nRun `make-goldset` first.")
        sys.exit(1)

    gold = pd.read_csv(args.goldset, encoding="utf-8-sig")
    gold["gold_url"] = gold.get("gold_url", "").map(_clean_str)
    gold["author_name"] = gold["author_name"].map(_clean_str)
    gold["book_name"] = gold["book_name"].map(_clean_str)

    labeled = gold[gold["gold_url"].map(lambda u: _goodreads_id(u) is not None)].reset_index(drop=True)
    skipped = len(gold) - len(labeled)
    if labeled.empty:
        print(f"No usable gold_url values in {args.goldset}. Fill in the gold_url column first.")
        print("(Each gold_url must be a real Goodreads /book/show/<ID>-... link.)")
        sys.exit(1)

    print(f"Gold set: {len(labeled)} labeled books ({skipped} skipped/blank)")
    print(f"Models:   {', '.join(models)}")
    print(f"Workers:  {args.max_workers} (per model)\n")

    client = _make_client()
    rows: List[Dict[str, Any]] = []  # long-form: one record per (book, model)

    for model in models:
        print(f"── {model} ──")
        results: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            fut_to_i = {
                pool.submit(search_once, client, model,
                            r["book_name"], r["author_name"]): i
                for i, r in labeled.iterrows()
            }
            for fut in as_completed(fut_to_i):
                results[fut_to_i[fut]] = fut.result()

        for i, r in labeled.iterrows():
            res = results[i]
            st = "error" if res["error"] else score(res["url"], r["gold_url"])
            rows.append({
                "book_name": r["book_name"],
                "author_name": r["author_name"],
                "gold_url": r["gold_url"],
                "model": model,
                "returned_url": res["url"] or "",
                "status": st,
                "latency_s": round(res["latency"], 2),
                "prompt_tokens": res["usage"]["prompt_tokens"],
                "completion_tokens": res["usage"]["completion_tokens"],
                "error": res["error"],
            })
            mark = {"correct": "OK ", "wrong": "WR ", "missing": "-- ", "error": "ERR"}.get(st, "?  ")
            print(f"  {mark} {r['book_name'][:48]:48}  {res['url'] or res['error'] or '(none)'}")
        print()

    detail = pd.DataFrame(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    detail_path = os.path.join(args.out_dir, f"results_{ts}_detail.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary = _summarize(detail, models, len(labeled))
    print(summary)
    summary_path = os.path.join(args.out_dir, f"results_{ts}.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Detail:  {detail_path}")
    print(f"Summary: {summary_path}")
    print("\nNote: accuracy is by Goodreads book ID. A 'wrong' row may still be a valid")
    print("alternate edition — eyeball the detail CSV before trusting the headline number.")


def _summarize(detail: pd.DataFrame, models: List[str], n_books: int) -> str:
    lines = []
    lines.append("# Goodreads search benchmark\n")
    lines.append(f"Gold set: **{n_books} labeled books**\n")
    lines.append("| model | accuracy | correct | wrong | missing | err | avg latency | tokens | ~cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for model in models:
        sub = detail[detail["model"] == model]
        if sub.empty:
            continue
        c = (sub["status"] == "correct").sum()
        w = (sub["status"] == "wrong").sum()
        miss = (sub["status"] == "missing").sum()
        err = (sub["status"] == "error").sum()
        acc = c / n_books if n_books else 0.0
        lat = sub["latency_s"].mean()
        ptok = int(sub["prompt_tokens"].sum())
        ctok = int(sub["completion_tokens"].sum())
        price = PRICES.get(model) or PRICES.get(model.replace(":online", ""))
        cost = f"${(ptok/1e6)*price[0] + (ctok/1e6)*price[1]:.3f}" if price else "—"
        lines.append(
            f"| {model} | **{acc:.0%}** | {c} | {w} | {miss} | {err} | "
            f"{lat:.1f}s | {ptok+ctok:,} | {cost} |"
        )
    lines.append("\n_accuracy = correct / total labeled; cost is token-based only "
                 "(excludes Perplexity per-request search fees)._\n")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    mg = sub.add_parser("make-goldset", help="Sample books into a gold-set template to hand-label")
    mg.add_argument("--input", default="book_mentions_research.csv", help="Pipeline CSV to sample from")
    mg.add_argument("--out", default="benchmark/goldset.csv", help="Where to write the template")
    mg.add_argument("-n", type=int, default=25, help="Number of books to sample")
    mg.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    mg.set_defaults(func=make_goldset)

    rn = sub.add_parser("run", help="Run each model against the gold set and score")
    rn.add_argument("--goldset", default="benchmark/goldset.csv", help="Hand-labeled gold set CSV")
    rn.add_argument("--out_dir", default="benchmark", help="Where to write results")
    rn.add_argument("--models", default="", help="Comma-separated OpenRouter model ids (default: built-in candidates)")
    rn.add_argument("--max_workers", type=int, default=4, help="Parallel searches per model")
    rn.set_defaults(func=run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
