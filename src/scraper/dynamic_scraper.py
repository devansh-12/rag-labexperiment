"""
src/scraper/dynamic_scraper.py
─────────────────────────────────────────────────────────────────────────────
Playwright-based dynamic scraper with graceful fallback to static scraper.
"""
from datetime import datetime, timezone
from urllib.parse import urlparse
import time
import trafilatura
from bs4 import BeautifulSoup
from .static_scraper import scrape_static

JS_PAGES = [
    "https://www.spit.ac.in/placements/",
    "https://www.spit.ac.in/events/",
]

def scrape_dynamic(url: str, doc_id: str,
                   wait_selector: str = "body",
                   extra_clicks: list[str] = None) -> dict | None:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"User-Agent": "SPITRagBot/1.0 (academic-research; CE413)"})
            page.goto(url, wait_until="networkidle", timeout=25_000)
            page.wait_for_selector(wait_selector, timeout=8_000)

            if extra_clicks:
                for selector in extra_clicks:
                    try:
                        for el in page.query_selector_all(selector):
                            el.click()
                            time.sleep(0.3)
                    except Exception:
                        pass

            html = page.content()
            browser.close()

            text = trafilatura.extract(html, include_tables=True, favor_recall=True)
            soup = BeautifulSoup(html, "html.parser")
            h1 = soup.find("h1")
            title = soup.find("title")
            title_str = (h1.get_text(strip=True) if h1 else (title.get_text(strip=True) if title else url))

            return {
                "doc_id": doc_id,
                "source_url": url,
                "source_name": _infer_name(url),
                "title": title_str,
                "type": "webpage_dynamic",
                "date_collected": datetime.now(timezone.utc).isoformat(),
                "full_text": text or "",
                "raw_html": html,
            }
    except Exception as e:
        print(f"[INFO] Playwright unavailable or failed for {url} ({e}); falling back to static scrape.")
        return scrape_static(url, doc_id)

def _infer_name(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    return " > ".join(p.replace("-", " ").title() for p in parts if p) or "Home"
