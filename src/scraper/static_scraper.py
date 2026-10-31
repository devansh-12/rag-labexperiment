"""
src/scraper/static_scraper.py
─────────────────────────────────────────────────────────────────────────────
Static HTML page scraper using requests, trafilatura and BeautifulSoup fallback.
"""
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests
import trafilatura
from bs4 import BeautifulSoup

def scrape_static(url: str, doc_id: str) -> dict | None:
    headers = {"User-Agent": "SPITRagBot/1.0 (academic-research; CE413)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        raw_html = resp.text
    except Exception as e:
        print(f"[WARN] Failed fetching {url}: {e}")
        return None

    extracted = trafilatura.extract(
        raw_html,
        include_tables=True,
        include_links=False,
        favor_recall=True,
    )

    if not extracted or len(extracted) < 150:
        extracted = _bs4_extract(raw_html)

    soup = BeautifulSoup(raw_html, "html.parser")
    title = soup.find("title")
    h1 = soup.find("h1")
    sections = _extract_section_map(soup)

    title_str = (h1.get_text(strip=True) if h1 else (title.get_text(strip=True) if title else url))

    return {
        "doc_id": doc_id,
        "source_url": url,
        "source_name": _infer_name(url),
        "title": title_str,
        "type": "webpage",
        "date_collected": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
        "full_text": extracted or "",
        "raw_html": raw_html,
    }

def _bs4_extract(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style", "aside", "form", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator="\n").split())

def _extract_section_map(soup: BeautifulSoup) -> list[dict]:
    sections, current_heading, buffer = [], "Introduction", []
    for el in soup.find_all(["h1", "h2", "h3", "p", "li", "td"]):
        if el.name in ("h1", "h2", "h3"):
            if buffer:
                sections.append({"heading": current_heading, "content": " ".join(buffer)})
            current_heading = el.get_text(strip=True)
            buffer = []
        else:
            t = el.get_text(strip=True)
            if t:
                buffer.append(t)
    if buffer:
        sections.append({"heading": current_heading, "content": " ".join(buffer)})
    return sections

def _infer_name(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    return " > ".join(p.replace("-", " ").title() for p in parts if p) or "Home"
