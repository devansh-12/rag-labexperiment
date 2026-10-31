"""
src/scraper/pdf_downloader.py
─────────────────────────────────────────────────────────────────────────────
PDF discovery and download utility.
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pathlib
import hashlib

def find_linked_pdfs(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if href.lower().endswith(".pdf"):
            pdfs.append(href)
    return list(set(pdfs))

def download_pdf(url: str, save_dir: pathlib.Path, doc_id: str) -> pathlib.Path | None:
    try:
        resp = requests.get(url, timeout=25, headers={"User-Agent": "SPITRagBot/1.0 (academic-research; CE413)"})
        resp.raise_for_status()
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"{doc_id}_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
        fname.write_bytes(resp.content)
        return fname
    except Exception as e:
        print(f"[WARN] PDF download failed: {url} - {e}")
        return None
