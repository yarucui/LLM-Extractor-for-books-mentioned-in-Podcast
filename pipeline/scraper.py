import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Optional

class GoodreadsScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def scrape_book_metadata(self, url: str) -> Dict[str, Optional[str]]:
        """
        Scrapes title and author from a Goodreads book page.
        Returns a dict with 'title' and 'author'.
        """
        if not url or "goodreads.com/book/show/" not in url:
            return {"title": None, "author": None, "error": "Invalid Goodreads URL"}

        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return {"title": None, "author": None, "error": f"HTTP {response.status_code}"}

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Goodreads uses data-testid for stable selection
            title_elem = soup.find("h1", {"data-testid": "bookTitle"})
            if not title_elem:
                # Fallback for older layout
                title_elem = soup.find("h1", id="bookTitle")
            
            author_elem = soup.find("span", {"data-testid": "name"})
            if not author_elem:
                # Fallback for older layout
                author_elem = soup.select_one("div.authorName__container a span")

            return {
                "title": title_elem.get_text(strip=True) if title_elem else None,
                "author": author_elem.get_text(strip=True) if author_elem else None,
                "error": None
            }
        except Exception as e:
            return {"title": None, "author": None, "error": str(e)}
