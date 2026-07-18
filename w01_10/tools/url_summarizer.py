"""
Name: UrlSummarizer

Description: "Fetches a webpage, extracts main text, and returns a concise summary."

Inputs: { "url": str, "max_words": int = 120 }

Outputs: short summary string
"""

from typing import Optional
import re, time
import requests 
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, PositiveInt, ValidationError
from cachetools import cached, TTLCache

# -------- Input schema --------
class UrlSummarizeInput(BaseModel):
    url: HttpUrl
    max_words: PositiveInt = 120

# -------- Utilities --------

def clean_text(txt: str) -> str:
    """
    Clean and normalize whitespace in text.
    
    Collapses multiple consecutive whitespace characters into a single space
    and strips leading/trailing whitespace.
    
    Args:
        txt: The input text to clean.
        
    Returns:
        Cleaned text with normalized whitespace.
    """
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_main_text(html: str) -> str:
    """
    Extract main content text from HTML.
    
    Attempts to find the main article content first, falling back to all
    paragraph tags if no article element is found. Cleans the extracted text.
    
    Args:
        html: Raw HTML content as a string.
        
    Returns:
        Extracted and cleaned main text from the HTML.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Heuristic: prefer <article>, else join <p>
    article = soup.find("article")
    if article:
        text = " ".join(p.get_text(" ", strip=True) for p in article.find_all("p"))
    else:
        text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return clean_text(text)

def http_get(url: str, retries: int = 3, backoff: float = 1.5, timeout: int = 15) -> requests.Response:
    """
    Fetch a URL with exponential backoff retry logic.
    
    Attempts to fetch the URL multiple times with exponential backoff on failure.
    Includes proper error handling and User-Agent headers.
    
    Args:
        url: The URL to fetch.
        retries: Number of retry attempts (default: 3).
        backoff: Backoff multiplier for exponential delays (default: 1.5).
        timeout: Request timeout in seconds (default: 15).
        
    Returns:
        The requests.Response object if successful.
        
    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    last_err = None
    for attempt in range(1, retries+1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent":"AgenticAI/1.0"})
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            if attempt == retries:
                raise
            time.sleep(backoff**attempt)
    raise last_err

 
# -------- Caching --------
cache = TTLCache(maxsize=256, ttl=60*10)  # 10-minute TTL
 
@cached(cache)
def fetch_and_extract(url: str) -> str:
    """
    Fetch a URL and extract its main text content with caching.
    
    Fetches the webpage and extracts main text. Results are cached with a
    10-minute TTL to avoid redundant requests for the same URL.
    
    Args:
        url: The URL to fetch and extract content from.
        
    Returns:
        Extracted main text from the webpage.
        
    Raises:
        requests.RequestException: If the URL fetch fails.
    """
    resp = http_get(url)
    return extract_main_text(resp.text)
 

  
# -------- The tool function --------
def summarize_url(url: str, max_words: int = 120) -> str:
    """
    Fetch and summarize the content of a given URL.
    
    Validates the input URL, fetches the webpage, extracts main content,
    and returns a concise summary truncated to the specified word limit
    while respecting sentence boundaries where possible.
    
    Args:
        url: The URL to summarize (must be a valid HTTP/HTTPS URL).
        max_words: Maximum words in the summary (default: 120, must be positive).
        
    Returns:
        A summary string of the webpage content, or an error message if
        validation, fetching, or parsing fails.
    """
    try:
        args = UrlSummarizeInput(url=url, max_words=max_words)
    except ValidationError as ve:
        return f"Input validation error: {ve}"
 
    try:
        text = fetch_and_extract(str(args.url))
        if not text or len(text.split()) < 30:
            return "Sorry, couldn't find enough readable content on that page."
    except Exception as e:
        return f"Failed to fetch or parse the URL: {e}"
 
    # Lightweight on-device summarizer via heuristic compression:
    # (For production: call an LLM to summarize + guardrails)
    words = text.split()
    if len(words) <= args.max_words:
        return " ".join(words)
 
    # Sentence-aware truncate
    out = []
    count = 0
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        sw = sentence.split()
        if count + len(sw) > args.max_words:
            break
        out.append(sentence)
        count += len(sw)
 
    if not out:
        out = words[:args.max_words]
        return " ".join(out) + " ..."
    return " ".join(out)