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

    def scrape_book_metadata(self, url: str, max_retries: int = 3) -> Dict[str, Optional[str]]:
        """
        Scrapes title and author from a Goodreads book page with retry logic for 503 errors.
        """
        if not url or "goodreads.com/book/show/" not in url:
            return {"title": None, "author": None, "error": "Invalid Goodreads URL"}

        retries = 0
        backoff = 2
        while retries <= max_retries:
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Goodreads uses data-testid for stable selection
                    title_elem = soup.find("h1", {"data-testid": "bookTitle"})
                    if not title_elem:
                        title_elem = soup.find("h1", id="bookTitle")
                    
                    # Find ALL authors
                    author_elements = soup.find_all("span", {"data-testid": "name"})
                    if not author_elements:
                        # Fallback for older layout
                        author_elements = soup.select("div.authorName__container a span")
                    
                    authors = [el.get_text(strip=True) for el in author_elements]
                    author_str = ", ".join(authors) if authors else None

                    return {
                        "title": title_elem.get_text(strip=True) if title_elem else None,
                        "author": author_str,
                        "error": None
                    }
                
                if response.status_code == 503:
                    retries += 1
                    if retries <= max_retries:
                        print(f"Goodreads 503 (Service Unavailable). Retrying in {backoff}s... ({retries}/{max_retries})")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                
                return {"title": None, "author": None, "error": f"HTTP {response.status_code}"}
                
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return {"title": None, "author": None, "error": str(e)}
        
        return {"title": None, "author": None, "error": "Max retries reached for 503 error"}
