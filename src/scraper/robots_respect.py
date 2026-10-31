"""
src/scraper/robots_respect.py
─────────────────────────────────────────────────────────────────────────────
robots.txt parser and polite rate limiter.
"""
import urllib.robotparser
import time
import random

class PoliteCrawler:
    def __init__(self, base_url: str = "https://www.spit.ac.in", crawl_delay: float = 2.0):
        self.rp = urllib.robotparser.RobotFileParser()
        try:
            self.rp.set_url(f"{base_url.rstrip('/')}/robots.txt")
            self.rp.read()
        except Exception as e:
            print(f"[WARN] robots.txt fetch error: {e}")
        self.crawl_delay = crawl_delay

    def can_fetch(self, url: str) -> bool:
        try:
            return self.rp.can_fetch("*", url)
        except Exception:
            return True

    def polite_sleep(self):
        time.sleep(self.crawl_delay + random.uniform(0.1, 0.5))
