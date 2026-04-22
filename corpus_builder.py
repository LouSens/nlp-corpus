"""
corpus_builder.py
─────────────────────────────────────────────────────────────────
Builds corpus.txt from the Kaggle Financial PhraseBank dataset.

Sources read:
  sources/*.txt           ← format: sentence@label

Run BEFORE nlp_preprocessing.py:
    python corpus_builder.py
"""

import csv
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── PATHS ─────────────────────────────────────────────────────────────────────
SOURCES_DIR  = Path("sources")
OUTPUT_FILE  = Path("corpus.txt")
ENCODING     = "utf-8"
MIN_WORDS    = 25_000

# Which FinancialPhraseBank agreement-level files to include
# AllAgree = only sentences all annotators agreed on (highest quality)
# 50Agree  = broader coverage (more sentences, some label disagreement)
PHRASEBANK_FILES = [
    "Sentences_AllAgree.txt",
    "Sentences_50Agree.txt",
]


# ── READERS ───────────────────────────────────────────────────────────────────



def read_phrasebank_txt(path: Path) -> list[tuple[str, str]]:
    """
    Reads a FinancialPhraseBank .txt file.
    Format: sentence@label  (latin-1 encoded)
    Returns list of (label, sentence) tuples.
    """
    records = []
    try:
        with open(path, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if "@" not in line:
                    continue
                sentence, _, label = line.rpartition("@")
                sentence = sentence.strip()
                label    = label.strip().lower()
                if sentence:
                    records.append((label, sentence))
        log.info(f"  ✓ {path.name:<40}  {len(records):>6,} sentences  (sentence@label TXT)")
    except Exception as e:
        log.error(f"  ✗ Failed to read {path}: {e}")
    return records


def format_block(records: list[tuple[str, str]], source_name: str) -> str:
    """
    Formats a list of (label, sentence) records into a readable corpus block.
    Each line: [LABEL]  sentence
    """
    lines = [
        f"\n{'─' * 70}",
        f"SOURCE: {source_name}",
        f"SENTENCES: {len(records):,}",
        f"{'─' * 70}\n",
    ]
    for label, sentence in records:
        lines.append(f"[{label.upper():8}]  {sentence}")
    return "\n".join(lines)


# ── MAIN BUILD ────────────────────────────────────────────────────────────────

def build_corpus():
    log.info("Corpus Builder — Financial News Sentiment Analysis")
    log.info("=" * 60)

    all_records: list[tuple[str, str]] = []
    corpus_blocks: list[str] = []
    seen_sentences: set[str] = set()   # deduplication

    # ── 1. Read sources/*.txt ────────────────────────────────────────────────
    for fname in PHRASEBANK_FILES:
        fpath = SOURCES_DIR / fname
        if not fpath.exists():
            log.warning(f"  {fpath} not found — skipping.")
            continue
        records = read_phrasebank_txt(fpath)
        unique = [(l, s) for l, s in records if s not in seen_sentences]
        seen_sentences.update(s for _, s in unique)
        if unique:
            all_records.extend(unique)
            corpus_blocks.append(format_block(unique, f"sources/{fname}"))
        else:
            log.info(f"  (all sentences in {fname} already seen — skipped as duplicates)")

    # ── Validate ──────────────────────────────────────────────────────────────
    if not all_records:
        log.error("No data collected. Check that sources/ contains the Kaggle files.")
        sys.exit(1)

    # ── Corpus text ───────────────────────────────────────────────────────────
    # For NLP preprocessing we only need the sentence text (labels are metadata).
    # We embed them as readable tags so the preprocessing pipeline stays clean.
    full_corpus_text = "\n".join(corpus_blocks)
    word_count = len(full_corpus_text.split())

    # Breakdown by sentiment label
    from collections import Counter
    label_counts = Counter(label for label, _ in all_records)

    # ── Write corpus.txt ──────────────────────────────────────────────────────
    header = (
        f"CORPUS: Financial News Sentiment Analysis\n"
        f"Source: Kaggle — Financial PhraseBank (Malo et al., 2014)\n"
        f"Built : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'─' * 60}\n"
        f"Total sentences : {len(all_records):,}\n"
        f"  Positive      : {label_counts.get('positive', 0):,}\n"
        f"  Negative      : {label_counts.get('negative', 0):,}\n"
        f"  Neutral       : {label_counts.get('neutral',  0):,}\n"
        f"Total words     : {word_count:,}\n"
        f"{'=' * 60}\n\n"
    )

    with open(OUTPUT_FILE, "w", encoding=ENCODING) as fh:
        fh.write(header)
        fh.write(full_corpus_text)

    # ── Report ────────────────────────────────────────────────────────────────
    log.info("")
    log.info(f"  Total sentences collected : {len(all_records):,}")
    log.info(f"    Positive                : {label_counts.get('positive', 0):,}")
    log.info(f"    Negative                : {label_counts.get('negative', 0):,}")
    log.info(f"    Neutral                 : {label_counts.get('neutral',  0):,}")
    log.info(f"  Total words in corpus.txt : {word_count:,}")
    log.info(f"  Output file               : {OUTPUT_FILE.resolve()}")

    if word_count < MIN_WORDS:
        log.warning(
            f"\n  ⚠  Only {word_count:,} words — need {MIN_WORDS - word_count:,} more "
            f"to meet the {MIN_WORDS:,}-word minimum.\n"
            f"  → Include Sentences_50Agree.txt or Sentences_75Agree.txt in PHRASEBANK_FILES."
        )
    else:
        log.info(f"\n  ✓  Corpus meets the {MIN_WORDS:,}-word minimum requirement.")

    log.info("\nNext step: python nlp_preprocessing.py")


if __name__ == "__main__":
    build_corpus()
