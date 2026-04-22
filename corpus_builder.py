"""
corpus_builder.py
─────────────────────────────────────────────────────────────────
Helper script to MERGE all your source text files into one corpus.txt
Run this BEFORE nlp_preprocessing.py.

Usage:
    1. Place all your source .txt files in the sources/ folder.
    2. Run:  python corpus_builder.py
    3. A validated corpus.txt is created in the working directory.

Source files for Financial News Sentiment Analysis corpus:
  ─ financial_news_1.txt      (Reuters financial articles)
  ─ financial_news_2.txt      (Bloomberg opinion pieces)
  ─ annual_reports.txt        (SEC 10-K filing excerpts)
  ─ economic_forecasts.txt    (IMF / World Bank reports)
  ─ earnings_calls.txt        (Earnings call transcripts)
  ─ market_commentary.txt     (Analyst notes and commentary)
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── CONFIGURE SOURCE FOLDER ───────────────────────────────────────────────────
SOURCES_DIR    = Path("sources")
OUTPUT_FILE    = Path("corpus.txt")
ENCODING       = "utf-8"
MIN_WORD_COUNT = 25_000

def build_corpus():
    log.info("Corpus Builder — Financial News Sentiment Analysis")
    log.info("=" * 60)

    if not SOURCES_DIR.exists():
        SOURCES_DIR.mkdir()
        log.warning(
            f"Created empty 'sources/' folder.\n"
            f"  Please add your .txt source files to: {SOURCES_DIR.resolve()}\n"
            f"  Then re-run this script."
        )
        sys.exit(0)

    txt_files = sorted(SOURCES_DIR.glob("*.txt"))
    if not txt_files:
        log.error(f"No .txt files found in '{SOURCES_DIR}'. Add your source files first.")
        sys.exit(1)

    log.info(f"Found {len(txt_files)} source file(s):")
    combined_parts = []
    total_words    = 0

    for fpath in txt_files:
        try:
            text = fpath.read_text(encoding=ENCODING, errors="replace").strip()
            wc   = len(text.split())
            total_words += wc
            log.info(f"  ✓ {fpath.name:<35}  {wc:>8,} words")
            # Add a clear separator between sources
            combined_parts.append(f"\n\n{'─' * 60}\n")
            combined_parts.append(f"SOURCE: {fpath.name}\n")
            combined_parts.append(f"{'─' * 60}\n\n")
            combined_parts.append(text)
        except Exception as e:
            log.error(f"  ✗ Failed to read {fpath.name}: {e}")

    # ── Write merged corpus ───────────────────────────────────────────────────
    header = (
        f"CORPUS: Financial News Sentiment Analysis\n"
        f"Built : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Files : {len(txt_files)}\n"
        f"Words : {total_words:,}\n"
        f"{'=' * 60}\n\n"
    )

    with open(OUTPUT_FILE, "w", encoding=ENCODING) as fh:
        fh.write(header)
        fh.write("\n".join(combined_parts))

    log.info(f"\nTotal merged words : {total_words:,}")
    log.info(f"Output file        : {OUTPUT_FILE.resolve()}")

    if total_words < MIN_WORD_COUNT:
        log.warning(
            f"WARNING: {total_words:,} words is below the {MIN_WORD_COUNT:,} minimum.\n"
            f"  Add more source files to reach the requirement."
        )
    else:
        log.info(f"✓ Corpus meets the {MIN_WORD_COUNT:,}-word minimum requirement.")

    log.info("\nNext step: run  python nlp_preprocessing.py")

if __name__ == "__main__":
    build_corpus()
