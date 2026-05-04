"""
corpus_builder.py
─────────────────────────────────────────────────────────────────
Builds corpus.txt via web scraping of FREE public financial-news
sources.  No API key, no dataset download required.

Sources used (all free, no login):
  1. Yahoo Finance RSS       — Business/Finance headlines + summaries
  2. Reuters RSS             — World business news
  3. MarketWatch RSS         — Market news
  4. CNBC RSS                — Economy & Finance feeds
  5. Wikipedia               — Financial-topic article prose

Run BEFORE nlp_preprocessing.py:
    python corpus_builder.py
"""

import re
import sys
import time
import logging
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from html.parser import HTMLParser
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── PATHS & SETTINGS ──────────────────────────────────────────────────────────
OUTPUT_FILE = Path("corpus.txt")
ENCODING    = "utf-8"
MIN_WORDS   = 25_000
REQUEST_DELAY = 1.5          # seconds between HTTP requests (be polite)
USER_AGENT    = (
    "Mozilla/5.0 (compatible; NLP-Academic-Corpus-Builder/1.0; "
    "+https://github.com/student/nlp-corpus)"
)


# ── RSS FEED SOURCES ──────────────────────────────────────────────────────────
RSS_FEEDS = [
    # Yahoo Finance
    ("Yahoo Finance — Markets",     "https://finance.yahoo.com/news/rssindex"),
    ("Yahoo Finance — Top Stories", "https://finance.yahoo.com/rss/topstories"),
    # Reuters
    ("Reuters — Business",          "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters — Money",             "https://feeds.reuters.com/news/wealth"),
    # MarketWatch
    ("MarketWatch — Top Stories",   "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("MarketWatch — Markets",       "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
    # CNBC
    ("CNBC — Finance",              "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
    ("CNBC — Economy",              "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("CNBC — Earnings",             "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135"),
    # Investing.com
    ("Investing.com — Stock Market News", "https://www.investing.com/rss/news_25.rss"),
    ("Investing.com — Economy",           "https://www.investing.com/rss/news_14.rss"),
    # Seeking Alpha
    ("Seeking Alpha — Market Currents",   "https://seekingalpha.com/market_currents.xml"),
    # Nasdaq
    ("Nasdaq — Original Articles",        "https://www.nasdaq.com/feed/rssoutbound?category=Original+Articles"),
]

# ── WIKIPEDIA FINANCIAL ARTICLES ──────────────────────────────────────────────
# Plain-text export (action=raw) — no API key required
WIKIPEDIA_ARTICLES = [
    "Stock_market",
    "Bond_(finance)",
    "Inflation",
    "Interest_rate",
    "Gross_domestic_product",
    "Financial_market",
    "Stock_exchange",
    "Central_bank",
    "Monetary_policy",
    "Fiscal_policy",
    "Recession",
    "Bull_market",
    "Bear_market",
    "Hedge_fund",
    "Mutual_fund",
    "Exchange-traded_fund",
    "Dividend",
    "Equity_(finance)",
    "Debt",
    "Credit_rating",
    "Financial_crisis_of_2007%E2%80%932008",
    "Quantitative_easing",
    "Federal_Reserve",
    "European_Central_Bank",
    "Foreign_exchange_market",
    "Cryptocurrency",
    "Bitcoin",
    "Initial_public_offering",
    "Mergers_and_acquisitions",
    "Private_equity",
    "Venture_capital",
    "Portfolio_(finance)",
    "Asset_management",
    "Supply_and_demand",
    "Trade_deficit",
    "Balance_of_trade",
    "Economic_growth",
    "Unemployment",
    "Consumer_price_index",
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/index.php?action=raw&title={}"


# ── HELPERS ───────────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Minimal HTML → plain-text stripper (no external dependencies)."""

    def __init__(self):
        super().__init__()
        self.reset()
        self._parts = []
        self._skip  = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False
        if tag in ("p", "br", "li", "div", "h1", "h2", "h3"):
            self._parts.append(" ")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join("".join(self._parts).split())


def strip_html(html: str) -> str:
    s = _HTMLStripper()
    try:
        s.feed(html)
    except Exception:
        pass
    return s.get_text()


def clean_wiki_markup(raw: str) -> str:
    """Strip MediaWiki markup, leaving readable prose."""
    # Remove file/image links
    raw = re.sub(r"\[\[File:[^\]]*\]\]", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[\[Image:[^\]]*\]\]", "", raw, flags=re.IGNORECASE)
    # Remove templates {{...}}
    # Iteratively remove nested templates
    for _ in range(6):
        raw = re.sub(r"\{\{[^{}]*\}\}", "", raw)
    # Remove table markup
    raw = re.sub(r"\{\|.*?\|\}", "", raw, flags=re.DOTALL)
    # Convert [[link|text]] → text, [[link]] → link
    raw = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", raw)
    # Remove external links [http... text] → text
    raw = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", raw)
    raw = re.sub(r"\[https?://\S+\]", "", raw)
    # Remove section headers (== Heading ==)
    raw = re.sub(r"={2,}[^=]+=*={2,}", "", raw)
    # Remove category lines
    raw = re.sub(r"\[\[Category:[^\]]*\]\]", "", raw, flags=re.IGNORECASE)
    # Remove HTML tags that slip through
    raw = re.sub(r"<[^>]+>", " ", raw)
    # Remove references <ref...>...</ref>
    raw = re.sub(r"<ref[^/]*/?>.*?</ref>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref[^>]*/?>", "", raw)
    # Collapse whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw.strip()


def http_get(url: str, timeout: int = 20) -> bytes | None:
    """Fetch a URL and return raw bytes, or None on error."""
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        log.warning(f"    HTTP {e.code} — {url}")
    except URLError as e:
        log.warning(f"    URLError — {url}: {e.reason}")
    except Exception as e:
        log.warning(f"    Error fetching {url}: {e}")
    return None


def word_count(text: str) -> int:
    return len(text.split())


# ── SCRAPER 1: RSS FEEDS ──────────────────────────────────────────────────────

def scrape_rss_feed(name: str, url: str) -> list[str]:
    """
    Parse an RSS/Atom feed and return a list of text snippets
    (title + description/summary for each item).
    """
    log.info(f"  RSS  ← {name}")
    data = http_get(url)
    if not data:
        return []

    snippets = []
    try:
        root = ET.fromstring(data)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        # RSS 2.0
        for item in root.iter("item"):
            parts = []
            title = item.findtext("title", "").strip()
            desc  = item.findtext("description", "").strip()
            if title:
                parts.append(strip_html(title))
            if desc:
                parts.append(strip_html(desc))
            text = " ".join(parts).strip()
            if len(text.split()) >= 5:
                snippets.append(text)

        # Atom feeds
        if not snippets:
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                parts = []
                title   = entry.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
                summary = entry.findtext("{http://www.w3.org/2005/Atom}summary", "").strip()
                content = entry.findtext("{http://www.w3.org/2005/Atom}content", "").strip()
                for field in (title, summary, content):
                    if field:
                        parts.append(strip_html(field))
                text = " ".join(parts).strip()
                if len(text.split()) >= 5:
                    snippets.append(text)

    except ET.ParseError as e:
        log.warning(f"    XML parse error for {name}: {e}")

    log.info(f"    → {len(snippets)} snippets  (~{sum(word_count(s) for s in snippets):,} words)")
    return snippets


# ── SCRAPER 2: WIKIPEDIA ARTICLES ─────────────────────────────────────────────

def scrape_wikipedia_article(title: str) -> str:
    """
    Fetch the raw wikitext of a Wikipedia article and return cleaned prose.
    Uses action=raw — no API key needed.
    """
    url = WIKIPEDIA_API.format(title)
    log.info(f"  WIKI ← {title.replace('%E2%80%93', '–')}")
    data = http_get(url)
    if not data:
        return ""

    try:
        raw  = data.decode("utf-8", errors="replace")
        text = clean_wiki_markup(raw)
        wc   = word_count(text)
        log.info(f"    → ~{wc:,} words")
        return text
    except Exception as e:
        log.warning(f"    Failed to process {title}: {e}")
        return ""


# ── MAIN BUILD ────────────────────────────────────────────────────────────────

def build_corpus():
    log.info("Corpus Builder — Financial News (Web Scraping)")
    log.info("=" * 60)
    log.info(f"Target minimum: {MIN_WORDS:,} words")
    log.info("")

    corpus_blocks: list[str] = []
    seen_hashes:   set[str]  = set()   # SHA-1 of normalised snippet for dedup
    total_words = 0

    def dedup_add(text: str) -> str:
        """Return text only if not seen before (by content hash)."""
        key = hashlib.sha1(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()
        if key in seen_hashes:
            return ""
        seen_hashes.add(key)
        return text

    # ── 1. RSS FEEDS ──────────────────────────────────────────────────────────
    log.info("── PHASE 1: RSS Feed Scraping ──────────────────────────────────")
    rss_snippets: list[str] = []
    for name, url in RSS_FEEDS:
        snippets = scrape_rss_feed(name, url)
        for s in snippets:
            clean = dedup_add(s)
            if clean:
                rss_snippets.append(clean)
        time.sleep(REQUEST_DELAY)

    if rss_snippets:
        block_text = "\n".join(rss_snippets)
        wc = word_count(block_text)
        total_words += wc
        header = (
            f"\n{'─' * 70}\n"
            f"SOURCE: Financial News RSS Feeds\n"
            f"SNIPPETS: {len(rss_snippets):,}\n"
            f"WORDS: {wc:,}\n"
            f"{'─' * 70}\n"
        )
        corpus_blocks.append(header + block_text)
        log.info(f"\n  ✓ RSS total: {len(rss_snippets):,} snippets  ({wc:,} words)")
    else:
        log.warning("  No RSS snippets collected. Check network connectivity.")

    # ── 2. WIKIPEDIA ──────────────────────────────────────────────────────────
    log.info("\n── PHASE 2: Wikipedia Article Scraping ─────────────────────────")
    wiki_articles: list[tuple[str, str]] = []
    for title in WIKIPEDIA_ARTICLES:
        article_text = scrape_wikipedia_article(title)
        if article_text:
            clean = dedup_add(article_text)
            if clean:
                wiki_articles.append((title.replace("%E2%80%93", "–"), clean))
        time.sleep(REQUEST_DELAY)

    if wiki_articles:
        for art_title, art_text in wiki_articles:
            wc = word_count(art_text)
            total_words += wc
            header = (
                f"\n{'─' * 70}\n"
                f"SOURCE: Wikipedia — {art_title.replace('_', ' ')}\n"
                f"WORDS: {wc:,}\n"
                f"{'─' * 70}\n"
            )
            corpus_blocks.append(header + art_text)

        wiki_wc = sum(word_count(t) for _, t in wiki_articles)
        log.info(f"\n  ✓ Wikipedia total: {len(wiki_articles)} articles  ({wiki_wc:,} words)")
    else:
        log.warning("  No Wikipedia articles collected.")

    # ── Validate ──────────────────────────────────────────────────────────────
    if not corpus_blocks:
        log.error("No data collected at all. Check network connectivity.")
        sys.exit(1)

    # ── Assemble corpus.txt ───────────────────────────────────────────────────
    full_text = "\n".join(corpus_blocks)
    # Recount on full assembled text for accuracy
    total_words = word_count(full_text)

    header = (
        f"CORPUS: Financial News Sentiment Analysis\n"
        f"Method: Web Scraping (RSS Feeds + Wikipedia)\n"
        f"Built : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'─' * 60}\n"
        f"Total sections  : {len(corpus_blocks):,}\n"
        f"  RSS snippets  : {len(rss_snippets):,}\n"
        f"  Wiki articles : {len(wiki_articles):,}\n"
        f"Total words     : {total_words:,}\n"
        f"{'=' * 60}\n\n"
    )

    with open(OUTPUT_FILE, "w", encoding=ENCODING) as fh:
        fh.write(header)
        fh.write(full_text)

    # ── Final report ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info(f"  Total sections written  : {len(corpus_blocks):,}")
    log.info(f"    RSS snippets          : {len(rss_snippets):,}")
    log.info(f"    Wikipedia articles    : {len(wiki_articles):,}")
    log.info(f"  Total words in corpus   : {total_words:,}")
    log.info(f"  Output file             : {OUTPUT_FILE.resolve()}")

    if total_words < MIN_WORDS:
        shortage = MIN_WORDS - total_words
        log.warning(
            f"\n  ⚠  Only {total_words:,} words — need {shortage:,} more "
            f"to meet the {MIN_WORDS:,}-word minimum.\n"
            f"  → Add more WIKIPEDIA_ARTICLES entries in corpus_builder.py,\n"
            f"    or extend the RSS_FEEDS list."
        )
    else:
        log.info(f"\n  ✓  Corpus meets the {MIN_WORDS:,}-word minimum requirement.")

    log.info("\nNext step: python nlp_preprocessing.py")


if __name__ == "__main__":
    build_corpus()
