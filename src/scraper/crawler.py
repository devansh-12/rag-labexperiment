"""
src/scraper/crawler.py
─────────────────────────────────────────────────────────────────────────────
Full scraper orchestrator for SPIT RAG Knowledge Base.
Discovers pages via sitemap and BFS, deduplicates, strips PII, downloads linked PDFs,
and updates kb/metadata.csv with complete provenance while preserving baseline documents.
"""
from collections import deque
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests

from .dedup import ContentDeduplicator
from .dynamic_scraper import JS_PAGES, scrape_dynamic
from .pdf_downloader import download_pdf, find_linked_pdfs
from .robots_respect import PoliteCrawler
from .static_scraper import scrape_static

BASE_URL = "https://www.spit.ac.in"
KB_DIR = Path(__file__).resolve().parent.parent.parent / "kb"
METADATA = KB_DIR / "metadata.csv"

SEED_URLS = [
    "https://www.spit.ac.in/",
    "https://www.spit.ac.in/about/",
    "https://www.spit.ac.in/about/principals-message/",
    "https://www.spit.ac.in/about/vision-mission/",
    "https://www.spit.ac.in/about/campus/",
    "https://www.spit.ac.in/about/contact/",
    "https://www.spit.ac.in/about/institutional-values/",
    "https://www.spit.ac.in/about/kulpati-dr-k-m-munshi/",
    "https://www.spit.ac.in/about/organogram/",
    "https://www.spit.ac.in/academic-calendar/",
    "https://www.spit.ac.in/academics/departments/",
    "https://www.spit.ac.in/academics/graduate/",
    "https://www.spit.ac.in/academics/post-graduate/",
    "https://www.spit.ac.in/academics/ph-d-programme/",
    "https://www.spit.ac.in/academics/faculty-empowerment-policy/",
    "https://www.spit.ac.in/academics/students-empowerment-policy/",
    "https://www.spit.ac.in/administration/governing-body/",
    "https://www.spit.ac.in/administration/academic-council/",
    "https://www.spit.ac.in/administration/finance-committee/",
    "https://www.spit.ac.in/administration/committees/",
    "https://www.spit.ac.in/administration/faculty/",
    "https://www.spit.ac.in/admissions/",
    "https://www.spit.ac.in/nirf/",
    "https://www.spit.ac.in/naac/",
    "https://www.spit.ac.in/ariia/",
    "https://www.spit.ac.in/best-practices/",
    "https://www.spit.ac.in/achievement/",
    "https://www.spit.ac.in/mandatory-disclosures/",
    "https://www.spit.ac.in/placements/",
    "https://www.spit.ac.in/student-activities/",
    "https://www.spit.ac.in/2026/05/04/cut-off-marks-for-2025-26-b-tech-dse-mca/",
    "https://www.spit.ac.in/2026/08/27/research-in-the-era-of-generative-ai/",
    "https://placements.spit.ac.in/",
    "https://research.spit.ac.in/",
    "https://iic.spit.ac.in/",
    "https://idealab.spit.ac.in/",
]

SENSITIVE = re.compile(
    r'\b[A-Z]{2}\d{2}[A-Z0-9]{6}\b'
    r'|\b[6-9]\d{9}\b'
    r'|\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b'
    r'|\b\d{12}\b'
)

def parse_sitemap(sitemap_url: str) -> list[str]:
    try:
        resp = requests.get(sitemap_url, timeout=10, headers={"User-Agent": "SPITRagBot/1.0"})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.content, "xml")
        locs = [tag.text.strip() for tag in soup.find_all("loc")]
        sitemaps = [tag.text.strip() for tag in soup.find_all("sitemap")]
        for s in sitemaps:
            locs += parse_sitemap(s)
        return locs
    except Exception as e:
        print(f"[WARN] Sitemap parsing failed: {e}")
        return []

def bfs_crawl(seed_urls: list[str], domain: str = "spit.ac.in", max_pages: int = 50,
              polite: PoliteCrawler = None) -> list[str]:
    visited, queue, discovered = set(), deque(seed_urls), []
    while queue and len(discovered) < max_pages:
        url = queue.popleft()
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if clean_url in visited or (polite and not polite.can_fetch(clean_url)):
            continue
        visited.add(clean_url)
        try:
            resp = requests.get(clean_url, timeout=10, headers={"User-Agent": "SPITRagBot/1.0 (academic-research; CE413)"})
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
            discovered.append(clean_url)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                full = urljoin(clean_url, a["href"])
                full_parsed = urlparse(full)
                if ("spit.ac.in" in full_parsed.netloc) and (full not in visited) and not full.endswith(('.jpg', '.png', '.jpeg', '.gif')):
                    queue.append(full)
            if polite:
                polite.polite_sleep()
        except Exception as e:
            print(f"[WARN] {clean_url}: {e}")
    return discovered

def run_scrape(max_pages: int = 40, download_pdfs: bool = True):
    KB_DIR.joinpath("website_pages").mkdir(parents=True, exist_ok=True)
    KB_DIR.joinpath("institutional_pdfs").mkdir(parents=True, exist_ok=True)
    KB_DIR.joinpath("erp_export").mkdir(parents=True, exist_ok=True)

    polite = PoliteCrawler(BASE_URL, crawl_delay=0.8)
    dedup = ContentDeduplicator()
    
    # ── Read existing metadata so baseline docs are preserved ──────────────
    meta_rows = []
    seen_urls = set()
    highest_id_num = -1
    
    if METADATA.exists():
        with open(METADATA, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta_rows.append(row)
                if row.get("source_url"):
                    seen_urls.add(row["source_url"].rstrip("/"))
                doc_id = row.get("doc_id", "")
                if doc_id.startswith("D") and doc_id[1:].isdigit():
                    highest_id_num = max(highest_id_num, int(doc_id[1:]))

        # Seed deduplicator with existing local text to prevent duplicates
        for row in meta_rows:
            lp = Path(row["local_path"])
            if not lp.is_absolute():
                lp = KB_DIR.parent / lp
            if lp.exists() and lp.suffix in (".txt", ".json"):
                try:
                    dedup.is_duplicate(lp.read_text(encoding="utf-8", errors="ignore"), row["doc_id"])
                except Exception:
                    pass

    for path in KB_DIR.rglob("*"):
        match = re.match(r"D(\d+)(?:_|\.|$)", path.name)
        if path.is_file() and match:
            highest_id_num = max(highest_id_num, int(match.group(1)))

    doc_counter = highest_id_num + 1
    print(f"[Scraper] Baseline has {len(meta_rows)} docs. New docs start from D{doc_counter:03d}.")

    sitemap_urls = parse_sitemap(f"{BASE_URL}/sitemap.xml")
    bfs_urls = bfs_crawl(SEED_URLS[:15], "spit.ac.in", max_pages=max_pages, polite=polite)
    all_urls = list(dict.fromkeys(SEED_URLS + sitemap_urls + bfs_urls))
    
    new_scraped_count = 0
    discovered_pdfs = set()

    for url in all_urls:
        if new_scraped_count >= max_pages:
            break
        clean_url = url.rstrip("/")
        if clean_url in seen_urls:
            continue
        if not polite.can_fetch(url):
            continue

        doc_id = f"D{doc_counter:03d}"
        print(f"[Scrape] Fetching {url} as {doc_id}…")
        
        if url in JS_PAGES:
            doc = scrape_dynamic(url, doc_id)
        else:
            doc = scrape_static(url, doc_id)

        if not doc or len(doc.get("full_text", "")) < 80:
            continue

        text = doc["full_text"]
        # Privacy sanitizer
        sanitized_text = SENSITIVE.sub("[REDACTED_PII]", text)

        if dedup.is_duplicate(sanitized_text, doc_id):
            print(f"[Dedup] Skipping duplicate content from {url}")
            continue

        # Look for linked institutional PDFs
        if download_pdfs and doc.get("raw_html"):
            for pdf_url in find_linked_pdfs(url, doc["raw_html"]):
                if "spit.ac.in" in pdf_url and not pdf_url.endswith("/"):
                    discovered_pdfs.add(pdf_url)

        out_path = KB_DIR / "website_pages" / f"{doc_id}.txt"
        out_path.write_text(sanitized_text, encoding="utf-8")

        sections_path = KB_DIR / "website_pages" / f"{doc_id}_sections.json"
        if "sections" in doc and doc["sections"]:
            sections = [
                {
                    key: SENSITIVE.sub("[REDACTED_PII]", value) if isinstance(value, str) else value
                    for key, value in section.items()
                }
                for section in doc["sections"]
            ]
            sections_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2))

        meta_rows.append({
            "doc_id": doc_id,
            "source_name": doc["source_name"],
            "source_url": doc["source_url"],
            "type": doc["type"],
            "title": doc["title"],
            "date_collected": doc["date_collected"],
            "local_path": f"kb/website_pages/{doc_id}.txt",
        })
        seen_urls.add(clean_url)
        doc_counter += 1
        new_scraped_count += 1
        if polite:
            polite.polite_sleep()

    # ── Download discovered institutional PDFs (up to 5 relevant) ─────────────
    if download_pdfs and discovered_pdfs:
        print(f"[PDF] Discovered {len(discovered_pdfs)} linked PDFs. Downloading key institutional ones…")
        pdf_count = 0
        for pdf_url in sorted(discovered_pdfs):
            if pdf_count >= 5:
                break
            clean_pdf = pdf_url.rstrip("/")
            if clean_pdf in seen_urls:
                continue
            doc_id = f"D{doc_counter:03d}"
            save_dir = KB_DIR / "institutional_pdfs"
            downloaded_file = download_pdf(pdf_url, save_dir, doc_id)
            if downloaded_file and downloaded_file.stat().st_size > 500:
                meta_rows.append({
                    "doc_id": doc_id,
                    "source_name": Path(pdf_url).name,
                    "source_url": pdf_url,
                    "type": "pdf",
                    "title": f"Institutional Document: {Path(pdf_url).stem.replace('-', ' ').title()}",
                    "date_collected": datetime.now(timezone.utc).isoformat(),
                    "local_path": f"kb/institutional_pdfs/{downloaded_file.name}",
                })
                seen_urls.add(clean_pdf)
                doc_counter += 1
                pdf_count += 1
                print(f"[PDF] Saved {downloaded_file.name} as {doc_id}")

    # ── Write updated metadata.csv ───────────────────────────────────────────
    if meta_rows:
        fieldnames = ["doc_id", "source_name", "source_url", "type", "title", "date_collected", "local_path"]
        with open(METADATA, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in meta_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"✅ [OK] Updated knowledge base: {len(meta_rows)} total docs ({new_scraped_count} new pages scraped) in {METADATA}")

if __name__ == "__main__":
    run_scrape(max_pages=25, download_pdfs=True)
