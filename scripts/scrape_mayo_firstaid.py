"""
scripts/scrape_mayo_firstaid.py
────────────────────────────────
Run ONCE to populate data/firstaid.sqlite3 with all Mayo Clinic first aid articles.
Uses Playwright to bypass advanced anti-bot protections.
"""

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI
from playwright.sync_api import sync_playwright

# ── Config ─────────────────────────────────────────────────────────────────

INDEX_URL   = "https://www.mayoclinic.org/first-aid"
DB_PATH     = Path("data/firstaid.sqlite3")
RATE_LIMIT  = 2.0   # seconds between requests (be polite)
EMBED_MODEL = "text-embedding-qwen3-embedding-0.6b"
LM_BASE_URL = "http://localhost:1234/v1"

# ── Embedding client ───────────────────────────────────────────────────────

embed_client = OpenAI(base_url=LM_BASE_URL, api_key="lm-studio")

def get_embedding(text: str) -> np.ndarray:
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(response.data[0].embedding)

# ── Database ───────────────────────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            url             TEXT NOT NULL UNIQUE,
            content         TEXT NOT NULL,
            title_embedding TEXT NOT NULL,
            scraped_at      TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON articles(title)")
    conn.commit()
    return conn

def upsert_article(conn, title, url, content, embedding: np.ndarray):
    conn.execute("""
        INSERT INTO articles (title, url, content, title_embedding)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title,
            content=excluded.content,
            title_embedding=excluded.title_embedding,
            scraped_at=datetime('now')
    """, (title, url, content, json.dumps(embedding.tolist())))
    conn.commit()

# ── Scraping ───────────────────────────────────────────────────────────────

def extract_article_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    links = []
    for a in soup.select("a[href*='/first-aid/']"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if not href or not text:
            continue
        if "/basics/" not in href and "/art-" not in href:
            continue
        if href.startswith("/"):
            href = "https://www.mayoclinic.org" + href
        links.append((text, href))
    seen, unique = set(), []
    for title, url in links:
        if url not in seen:
            seen.add(url)
            unique.append((title, url))
    return unique

def extract_article_content(soup: BeautifulSoup, title: str) -> str:
    for tag in soup.find_all(["nav", "header", "footer", "script", "style", "noscript", "iframe", "aside", "form"]):
        tag.decompose()
    for cls in ["advertising", "sponsorship", "related", "breadcrumb", "site-map", "social", "newsletter", "press"]:
        for el in soup.find_all(class_=re.compile(cls, re.I)):
            el.decompose()
    content_div = (
        soup.find("div", class_=re.compile(r"content|article|main", re.I))
        or soup.find("main") or soup.find("article") or soup.body
    )
    if content_div is None: return ""
    parts = []
    for el in content_div.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(separator=" ", strip=True)
        if len(text) < 20: continue
        if any(skip in text.lower() for skip in ["mayo clinic", "advertising", "copyright", "©"]): continue
        parts.append(text)
    deduped = []
    for part in parts:
        if not deduped or part != deduped[-1]: deduped.append(part)
    return "\n".join(deduped).strip()

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Aegis — Mayo Clinic First Aid Scraper (Playwright Edition)")
    print("=" * 60)

    try:
        embed_client.models.list()
    except Exception:
        print("ERROR: LM Studio not reachable at", LM_BASE_URL)
        sys.exit(1)

    conn = init_db(DB_PATH)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a realistic user agent
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        print(f"\nFetching index: {INDEX_URL}")
        try:
            page.goto(INDEX_URL, wait_until="load", timeout=60000)
            # Extra wait for scripts to finish
            page.wait_for_timeout(5000)
            soup = BeautifulSoup(page.content(), "html.parser")
        except Exception as e:
            print(f"ERROR: Could not fetch index page: {e}")
            browser.close()
            sys.exit(1)

        articles = extract_article_links(soup)
        if not articles:
            # Maybe the selector changed or we got a 403 anyway?
            if "Forbidden" in page.title():
                print("ERROR: Still getting 403 Forbidden even with Playwright.")
            else:
                print("ERROR: Found 0 articles. Check site structure or block status.")
            browser.close()
            sys.exit(1)

        print(f"Found {len(articles)} article links.\n")

        ok, skipped, failed = 0, 0, 0
        for i, (title, url) in enumerate(articles, 1):
            print(f"[{i:02d}/{len(articles)}] {title}")
            existing = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()
            if existing:
                print("  ↩  Already in DB — skipping.")
                skipped += 1
                continue

            time.sleep(RATE_LIMIT)

            try:
                page.goto(url, wait_until="load", timeout=60000)
                page.wait_for_timeout(2000)
                article_soup = BeautifulSoup(page.content(), "html.parser")
            except Exception as e:
                print(f"  ✗ Fetch failed: {e}")
                failed += 1
                continue

            content = extract_article_content(article_soup, title)
            if len(content) < 100:
                print(f"  ⚠  Content too short ({len(content)} chars) — skipping.")
                failed += 1
                continue

            try:
                embedding = get_embedding(title)
            except Exception as e:
                print(f"  ✗ Embedding failed: {e}")
                failed += 1
                continue

            upsert_article(conn, title, url, content, embedding)
            print(f"  ✓ Stored ({len(content)} chars)")
            ok += 1

        browser.close()

    conn.close()
    print(f"\nDone.  Stored: {ok}  Skipped: {skipped}  Failed: {failed}")

if __name__ == "__main__":
    main()
