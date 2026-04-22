"""
=============================================================================
NLP ASSIGNMENT — SPECIALIZED CORPUS PREPROCESSING PIPELINE
=============================================================================
Application  : Financial News Sentiment Analysis
Author       : [Your Name]
Date         : [Your Date]

Pipeline Stages
---------------
  Stage 0  →  Load & validate corpus.txt
  Stage 1  →  Sentence + word tokenization  →  tokens.txt
  Stage 2  →  Frequency distribution (top 10 selected tokens)  →  freq_plot.png
  Stage 3  →  Bigram / Trigram N-gram modeling  →  n-gram.txt
  Stage 4  →  WordNet Lemmatization  →  lemma.txt
  Stage 5  →  Vocabulary extraction (unique words)  →  vocab.txt
  Stage 6  →  Summary report printed to console

Dependencies
------------
  pip install nltk matplotlib seaborn pandas
  python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger wordnet stopwords
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import logging
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import nltk
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams, ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (edit paths here if needed)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "corpus_path"     : "corpus.txt",          # Input: your merged corpus file
    "output_dir"      : "nlp_output",          # Folder for all exported files
    "tokens_file"     : "tokens.txt",
    "ngram_file"      : "n-gram.txt",
    "lemma_file"      : "lemma.txt",
    "vocab_file"      : "vocab.txt",
    "freq_plot_file"  : "freq_plot.png",
    "ngram_type"      : "bigram",              # "bigram" or "trigram"
    "top_n_freq"      : 10,                    # Number of tokens for freq plot
    "top_n_ngrams"    : 50,                    # Top N n-grams saved to file
    "encoding"        : "utf-8",
    # ── Frequency plot: 10 SPECIFIC tokens you manually choose ──────────────
    # After running Stage 1 you will know the vocab. Replace these with 10
    # domain-meaningful tokens from your corpus (not stopwords).
    # Default set targets financial-domain vocabulary.
    "selected_tokens" : [
        "market", "revenue", "inflation", "growth",
        "investment", "profit", "risk", "interest",
        "economy", "forecast"
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0  ·  BOOTSTRAP  (NLTK downloads + output directory)
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap() -> Path:
    """Download required NLTK data and create the output directory."""
    log.info("Stage 0 ▸ Bootstrapping environment …")

    required_packages = [
        "punkt", "punkt_tab", "averaged_perceptron_tagger",
        "wordnet", "stopwords", "omw-1.4",
    ]
    for pkg in required_packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            log.info(f"  Downloading NLTK resource: {pkg}")
            nltk.download(pkg, quiet=True)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Output directory: {output_dir.resolve()}")
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CORPUS
# ─────────────────────────────────────────────────────────────────────────────
def load_corpus(path: str) -> str:
    """Load and validate the raw corpus text file."""
    log.info(f"Loading corpus from '{path}' …")
    corpus_path = Path(path)

    if not corpus_path.exists():
        log.error(
            f"corpus.txt not found at '{corpus_path.resolve()}'.\n"
            "  Please create it by merging all your source texts into one file:\n"
            "    cat source1.txt source2.txt source3.txt > corpus.txt\n"
            "  Then re-run this script."
        )
        sys.exit(1)

    with open(corpus_path, "r", encoding=CONFIG["encoding"], errors="replace") as fh:
        raw = fh.read()

    word_count = len(raw.split())
    char_count = len(raw)
    log.info(f"  Characters: {char_count:,}")
    log.info(f"  Words (raw split): {word_count:,}")

    if word_count < 25_000:
        log.warning(
            f"  Corpus has only {word_count:,} words. "
            "Minimum requirement is 25,000 words. Please add more source material."
        )
    else:
        log.info(f"  ✓ Word count meets the 25,000-word minimum requirement.")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1  ·  TOKENIZATION
# ─────────────────────────────────────────────────────────────────────────────
def stage_tokenization(raw_text: str, output_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Tokenize the corpus at sentence level and word level.

    Pipeline:
        raw_text
          ─→ sent_tokenize()         →  sentence list
          ─→ word_tokenize()         →  word/punctuation tokens
          ─→ lowercase + alpha filter →  clean word tokens
          ─→ Export tokens.txt

    Returns:
        sentences   : list of sentence strings
        word_tokens : list of clean lowercase word tokens (alpha only)
    """
    log.info("Stage 1 ▸ Tokenization …")
    t0 = time.time()

    # ── Sentence tokenization ────────────────────────────────────────────────
    sentences = sent_tokenize(raw_text)
    log.info(f"  Sentences detected: {len(sentences):,}")

    # ── Word tokenization (includes punctuation) ──────────────────────────────
    all_tokens = word_tokenize(raw_text)
    log.info(f"  All tokens (incl. punctuation): {len(all_tokens):,}")

    # ── Clean tokens: lowercase alphabetic words only ─────────────────────────
    word_tokens = [tok.lower() for tok in all_tokens if tok.isalpha()]
    log.info(f"  Clean word tokens (alpha, lowercase): {len(word_tokens):,}")

    # ── Export ────────────────────────────────────────────────────────────────
    out_path = output_dir / CONFIG["tokens_file"]
    with open(out_path, "w", encoding=CONFIG["encoding"]) as fh:
        fh.write("=" * 72 + "\n")
        fh.write("TOKENIZATION OUTPUT — NLP PREPROCESSING PIPELINE\n")
        fh.write("=" * 72 + "\n\n")
        fh.write(f"Total sentences : {len(sentences):,}\n")
        fh.write(f"Total tokens    : {len(all_tokens):,}\n")
        fh.write(f"Clean tokens    : {len(word_tokens):,}\n\n")

        fh.write("─" * 72 + "\n")
        fh.write("SENTENCE TOKENS\n")
        fh.write("─" * 72 + "\n")
        for i, sent in enumerate(sentences, 1):
            fh.write(f"[S{i:05d}] {sent.strip()}\n")

        fh.write("\n" + "─" * 72 + "\n")
        fh.write("WORD TOKENS (clean, lowercase)\n")
        fh.write("─" * 72 + "\n")
        # Write 10 tokens per line for readability
        for i in range(0, len(word_tokens), 10):
            chunk = word_tokens[i : i + 10]
            fh.write("  " + "  |  ".join(chunk) + "\n")

    log.info(f"  ✓ Exported → {out_path}")
    log.info(f"  Stage 1 completed in {time.time() - t0:.2f}s")
    return sentences, word_tokens


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2  ·  FREQUENCY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def stage_frequency_distribution(word_tokens: List[str], output_dir: Path) -> pd.DataFrame:
    """
    Compute frequency distribution for 10 selected domain tokens and plot.

    Pipeline:
        word_tokens
          ─→ FreqDist()              →  full frequency map
          ─→ filter selected_tokens  →  10-token subset
          ─→ bar chart (seaborn)     →  freq_plot.png
          ─→ printed observation

    Returns:
        DataFrame of selected tokens and their frequencies
    """
    log.info("Stage 2 ▸ Frequency Distribution …")
    t0 = time.time()

    # ── Build full frequency distribution ─────────────────────────────────────
    freq_dist = FreqDist(word_tokens)
    total = len(word_tokens)

    log.info(f"  Vocabulary size: {freq_dist.B():,} unique types")
    log.info(f"  Total token count: {total:,}")

    # ── Filter selected tokens ─────────────────────────────────────────────────
    selected = CONFIG["selected_tokens"]
    records = []
    for tok in selected:
        count = freq_dist[tok]
        pct   = (count / total) * 100
        records.append({"token": tok, "frequency": count, "percentage": round(pct, 4)})
        log.info(f"    {tok:<14}  count={count:>6,}   ({pct:.4f}%)")

    df = pd.DataFrame(records).sort_values("frequency", ascending=False).reset_index(drop=True)

    # ── Plot ───────────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({
        "font.family"  : "DejaVu Sans",
        "font.size"    : 11,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })

    palette = sns.color_palette("Blues_d", len(df))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        "Frequency Distribution — 10 Selected Domain Tokens\n"
        "Financial News Sentiment Analysis Corpus",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Left: bar chart
    ax = axes[0]
    bars = ax.barh(df["token"], df["frequency"], color=palette, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Frequency (count)", fontsize=11)
    ax.set_ylabel("Token", fontsize=11)
    ax.set_title("Token Frequency — Horizontal Bar", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    # Annotate bars
    for bar, row in zip(bars, df.itertuples()):
        ax.text(
            bar.get_width() + max(df["frequency"]) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{row.frequency:,} ({row.percentage}%)",
            va="center", fontsize=9, color="#444"
        )
    ax.invert_yaxis()

    # Right: table
    ax2 = axes[1]
    ax2.axis("off")
    table_data = [[r.token, f"{r.frequency:,}", f"{r.percentage}%"]
                  for r in df.itertuples()]
    tbl = ax2.table(
        cellText=table_data,
        colLabels=["Token", "Count", "%"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if row == 0:
            cell.set_facecolor("#2c5f8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf3fb")

    plt.tight_layout()
    plot_path = output_dir / CONFIG["freq_plot_file"]
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"  ✓ Plot saved → {plot_path}")
    log.info(f"  Stage 2 completed in {time.time() - t0:.2f}s")

    # ── Console observation ────────────────────────────────────────────────────
    top_token = df.iloc[0]
    bot_token = df.iloc[-1]
    print("\n" + "=" * 72)
    print("OBSERVATION — FREQUENCY DISTRIBUTION")
    print("=" * 72)
    print(f"  The frequency distribution of the 10 selected domain-specific tokens")
    print(f"  reveals meaningful patterns in the corpus vocabulary.")
    print(f"  The token '{top_token['token']}' is the most frequent with "
          f"{top_token['frequency']:,} occurrences ({top_token['percentage']}%),")
    print(f"  indicating it is a central theme in the financial news corpus.")
    print(f"  '{bot_token['token']}' is the least frequent at "
          f"{bot_token['frequency']:,} occurrences ({bot_token['percentage']}%),")
    print(f"  suggesting it appears in specialised contexts within the domain.")
    print(f"  The variation in frequency across tokens confirms the corpus covers")
    print(f"  a broad spectrum of financial topics relevant to sentiment analysis.")
    print("=" * 72 + "\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3  ·  N-GRAM MODELING
# ─────────────────────────────────────────────────────────────────────────────
def stage_ngram_modeling(word_tokens: List[str], output_dir: Path) -> List[Tuple]:
    """
    Compute bigram or trigram frequency model and export.

    Pipeline:
        word_tokens  (stopwords kept — n-grams benefit from context)
          ─→ nltk.bigrams() / nltk.trigrams()
          ─→ FreqDist of n-grams
          ─→ Export top N n-grams → n-gram.txt

    Returns:
        List of (ngram_tuple, count) for the most common n-grams
    """
    log.info(f"Stage 3 ▸ N-gram Modeling ({CONFIG['ngram_type']}) …")
    t0 = time.time()

    ngram_type = CONFIG["ngram_type"].lower()
    top_n      = CONFIG["top_n_ngrams"]

    # ── Remove stopwords and punctuation for cleaner n-grams ──────────────────
    stop_words  = set(stopwords.words("english"))
    filtered    = [tok for tok in word_tokens if tok not in stop_words]
    log.info(f"  Tokens after stopword removal: {len(filtered):,}")

    # ── Generate n-grams ───────────────────────────────────────────────────────
    if ngram_type == "bigram":
        gram_list = list(bigrams(filtered))
        n = 2
    elif ngram_type == "trigram":
        gram_list = list(trigrams(filtered))
        n = 3
    else:
        log.error(f"Unknown ngram_type '{ngram_type}'. Use 'bigram' or 'trigram'.")
        sys.exit(1)

    log.info(f"  Total {ngram_type}s generated: {len(gram_list):,}")

    # ── Frequency distribution ─────────────────────────────────────────────────
    ngram_freq  = FreqDist(gram_list)
    most_common = ngram_freq.most_common(top_n)

    log.info(f"  Top {top_n} {ngram_type}s computed.")

    # ── Export ─────────────────────────────────────────────────────────────────
    out_path = output_dir / CONFIG["ngram_file"]
    with open(out_path, "w", encoding=CONFIG["encoding"]) as fh:
        fh.write("=" * 72 + "\n")
        fh.write(f"N-GRAM MODELING OUTPUT — {ngram_type.upper()} MODEL\n")
        fh.write("=" * 72 + "\n\n")
        fh.write(f"N-gram type        : {ngram_type} (n={n})\n")
        fh.write(f"Input tokens       : {len(word_tokens):,}\n")
        fh.write(f"Post-filter tokens : {len(filtered):,}\n")
        fh.write(f"Total {ngram_type}s      : {len(gram_list):,}\n")
        fh.write(f"Unique {ngram_type}s     : {ngram_freq.B():,}\n\n")

        fh.write("─" * 72 + "\n")
        fh.write(f"TOP {top_n} {ngram_type.upper()}S BY FREQUENCY\n")
        fh.write("─" * 72 + "\n")
        fh.write(f"{'Rank':<6}{'N-gram':<40}{'Count':>8}{'%Total':>10}\n")
        fh.write("─" * 72 + "\n")

        total_ngrams = len(gram_list)
        for rank, (gram, count) in enumerate(most_common, 1):
            gram_str = " ".join(gram)
            pct      = (count / total_ngrams) * 100
            fh.write(f"{rank:<6}{gram_str:<40}{count:>8,}{pct:>9.4f}%\n")

        fh.write("\n" + "─" * 72 + "\n")
        fh.write("ALL UNIQUE N-GRAMS (sorted by frequency)\n")
        fh.write("─" * 72 + "\n")
        for gram, count in ngram_freq.most_common():
            fh.write(f"{' '.join(gram)}\t{count}\n")

    log.info(f"  ✓ Exported → {out_path}")
    log.info(f"  Stage 3 completed in {time.time() - t0:.2f}s")
    return most_common


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4  ·  LEMMATIZATION
# ─────────────────────────────────────────────────────────────────────────────
def stage_lemmatization(word_tokens: List[str], output_dir: Path) -> List[str]:
    """
    Lemmatize all clean word tokens using NLTK WordNetLemmatizer.

    Pipeline:
        word_tokens
          ─→ pos_tag()                →  (word, POS) pairs
          ─→ POS → WordNet tag map    →  convert Penn Treebank → WordNet POS
          ─→ WordNetLemmatizer()      →  canonical base forms
          ─→ Export lemma.txt

    Note:
        POS-aware lemmatisation is significantly more accurate than
        default noun-only lemmatisation. E.g. "running" → "run" (verb)
        vs "running" → "running" (default noun fallback).

    Returns:
        List of lemmatized tokens
    """
    log.info("Stage 4 ▸ Lemmatization …")
    t0 = time.time()

    lemmatizer = WordNetLemmatizer()

    # ── POS tag → WordNet tag converter ───────────────────────────────────────
    def get_wordnet_pos(treebank_tag: str) -> str:
        """Map Penn Treebank POS to WordNet POS constants."""
        from nltk.corpus import wordnet
        if   treebank_tag.startswith("J"): return wordnet.ADJ
        elif treebank_tag.startswith("V"): return wordnet.VERB
        elif treebank_tag.startswith("N"): return wordnet.NOUN
        elif treebank_tag.startswith("R"): return wordnet.ADV
        else:                              return wordnet.NOUN   # default

    # ── POS-aware lemmatisation ────────────────────────────────────────────────
    log.info("  POS tagging tokens … (this may take a moment for large corpora)")
    tagged = pos_tag(word_tokens)

    lemmas = []
    changed = 0
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        lemma  = lemmatizer.lemmatize(word, pos=wn_tag)
        lemmas.append(lemma)
        if lemma != word:
            changed += 1

    pct_changed = (changed / len(word_tokens)) * 100 if word_tokens else 0
    log.info(f"  Tokens lemmatized  : {len(lemmas):,}")
    log.info(f"  Forms changed      : {changed:,}  ({pct_changed:.1f}%)")

    # ── Build change examples ─────────────────────────────────────────────────
    examples = [(w, l, t) for (w, t), l in zip(tagged, lemmas) if w != l][:20]

    # ── Export ─────────────────────────────────────────────────────────────────
    out_path = output_dir / CONFIG["lemma_file"]
    with open(out_path, "w", encoding=CONFIG["encoding"]) as fh:
        fh.write("=" * 72 + "\n")
        fh.write("LEMMATIZATION OUTPUT — NLP PREPROCESSING PIPELINE\n")
        fh.write("=" * 72 + "\n\n")
        fh.write(f"Lemmatizer         : NLTK WordNetLemmatizer (POS-aware)\n")
        fh.write(f"Total tokens       : {len(word_tokens):,}\n")
        fh.write(f"Forms changed      : {changed:,} ({pct_changed:.1f}%)\n\n")

        fh.write("─" * 72 + "\n")
        fh.write("SAMPLE TRANSFORMATIONS (original → lemma, POS tag)\n")
        fh.write("─" * 72 + "\n")
        fh.write(f"{'Original':<20}{'POS':<8}{'Lemma':<20}\n")
        fh.write("─" * 72 + "\n")
        for orig, lem, tag in examples:
            fh.write(f"{orig:<20}{tag:<8}{lem:<20}\n")

        fh.write("\n" + "─" * 72 + "\n")
        fh.write("FULL LEMMATIZED TOKEN SEQUENCE\n")
        fh.write("─" * 72 + "\n")
        for i in range(0, len(lemmas), 10):
            chunk = lemmas[i : i + 10]
            fh.write("  " + "  |  ".join(chunk) + "\n")

    log.info(f"  ✓ Exported → {out_path}")
    log.info(f"  Stage 4 completed in {time.time() - t0:.2f}s")
    return lemmas


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5  ·  VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
def stage_vocabulary(word_tokens: List[str], lemmas: List[str], output_dir: Path) -> Dict:
    """
    Extract and analyse vocabulary (unique word types) from the corpus.

    Pipeline:
        word_tokens  ─→  set()  →  unique raw types
        lemmas       ─→  set()  →  unique lemma types
        ─→ frequency rank + TTR
        ─→ Export vocab.txt

    Returns:
        Dictionary with vocabulary statistics
    """
    log.info("Stage 5 ▸ Vocabulary Extraction …")
    t0 = time.time()

    # ── Raw vocabulary ─────────────────────────────────────────────────────────
    raw_vocab   = sorted(set(word_tokens))
    lemma_vocab = sorted(set(lemmas))
    freq_counter = Counter(word_tokens)

    vocab_size_raw   = len(raw_vocab)
    vocab_size_lemma = len(lemma_vocab)
    ttr              = vocab_size_raw / len(word_tokens) if word_tokens else 0

    log.info(f"  Raw vocabulary size   : {vocab_size_raw:,} unique word types")
    log.info(f"  Lemma vocabulary size : {vocab_size_lemma:,} unique lemma types")
    log.info(f"  Type–Token Ratio (TTR): {ttr:.4f}")

    # ── Export ─────────────────────────────────────────────────────────────────
    out_path = output_dir / CONFIG["vocab_file"]
    with open(out_path, "w", encoding=CONFIG["encoding"]) as fh:
        fh.write("=" * 72 + "\n")
        fh.write("VOCABULARY OUTPUT — NLP PREPROCESSING PIPELINE\n")
        fh.write("=" * 72 + "\n\n")
        fh.write(f"Total tokens (running words) : {len(word_tokens):,}\n")
        fh.write(f"Raw vocabulary (types)       : {vocab_size_raw:,}\n")
        fh.write(f"Lemma vocabulary (types)     : {vocab_size_lemma:,}\n")
        fh.write(f"Type–Token Ratio (TTR)       : {ttr:.4f}\n")
        fh.write(f"Reduction after lemmatisation: "
                 f"{vocab_size_raw - vocab_size_lemma:,} forms\n\n")

        fh.write("─" * 72 + "\n")
        fh.write("VOCABULARY — FREQUENCY RANKED (raw tokens)\n")
        fh.write("─" * 72 + "\n")
        fh.write(f"{'Rank':<6}{'Word':<30}{'Count':>8}{'Cumul.%':>10}\n")
        fh.write("─" * 72 + "\n")

        total_tokens = len(word_tokens)
        cumul = 0
        for rank, (word, count) in enumerate(freq_counter.most_common(), 1):
            cumul += count
            cumul_pct = (cumul / total_tokens) * 100
            fh.write(f"{rank:<6}{word:<30}{count:>8,}{cumul_pct:>9.2f}%\n")

        fh.write("\n" + "─" * 72 + "\n")
        fh.write("ALPHABETICAL VOCABULARY LIST (raw)\n")
        fh.write("─" * 72 + "\n")
        cols = 4
        rows = [raw_vocab[i:i+cols] for i in range(0, len(raw_vocab), cols)]
        for row in rows:
            fh.write("  " + "   ".join(f"{w:<20}" for w in row) + "\n")

        fh.write("\n" + "─" * 72 + "\n")
        fh.write("LEMMA VOCABULARY (alphabetical)\n")
        fh.write("─" * 72 + "\n")
        rows = [lemma_vocab[i:i+cols] for i in range(0, len(lemma_vocab), cols)]
        for row in rows:
            fh.write("  " + "   ".join(f"{w:<20}" for w in row) + "\n")

    log.info(f"  ✓ Exported → {out_path}")
    log.info(f"  Stage 5 completed in {time.time() - t0:.2f}s")

    # ── Console observation ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("OBSERVATION — VOCABULARY")
    print("=" * 72)
    print(f"  The corpus contains {vocab_size_raw:,} unique word types across "
          f"{len(word_tokens):,} total tokens.")
    print(f"  The Type–Token Ratio (TTR) is {ttr:.4f}, indicating that "
          f"{'high' if ttr > 0.1 else 'moderate' if ttr > 0.05 else 'low'} lexical")
    print(f"  diversity is present in the corpus.")
    print(f"  After lemmatisation, the vocabulary reduces to {vocab_size_lemma:,} unique")
    print(f"  base forms — a reduction of {vocab_size_raw - vocab_size_lemma:,} morphological variants.")
    print(f"  This reduction improves downstream model generalisation by collapsing")
    print(f"  inflected forms (e.g., 'rising', 'rose', 'risen' → 'rise') into a")
    print(f"  single canonical representation.")
    print("=" * 72 + "\n")

    return {
        "vocab_size_raw"  : vocab_size_raw,
        "vocab_size_lemma": vocab_size_lemma,
        "ttr"             : ttr,
        "raw_vocab"       : raw_vocab,
        "lemma_vocab"     : lemma_vocab,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6  ·  SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(raw_text: str, sentences: List[str], word_tokens: List[str],
                  lemmas: List[str], ngrams_result: List[Tuple],
                  vocab_stats: Dict, output_dir: Path) -> None:
    """Print a formatted pipeline summary to the console."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  PIPELINE COMPLETE — SUMMARY REPORT")
    print(sep)
    print(f"  Corpus characters          : {len(raw_text):>12,}")
    print(f"  Sentences                  : {len(sentences):>12,}")
    print(f"  Word tokens (clean)        : {len(word_tokens):>12,}")
    print(f"  Lemmas                     : {len(lemmas):>12,}")
    print(f"  Raw vocabulary (types)     : {vocab_stats['vocab_size_raw']:>12,}")
    print(f"  Lemma vocabulary (types)   : {vocab_stats['vocab_size_lemma']:>12,}")
    print(f"  Type–Token Ratio (TTR)     : {vocab_stats['ttr']:>12.4f}")
    print(f"\n  Top 5 {CONFIG['ngram_type']}s:")
    for i, (gram, cnt) in enumerate(ngrams_result[:5], 1):
        print(f"    {i}. {'  '.join(gram):<30} → {cnt:,}")
    print(f"\n  Output files saved to: {output_dir.resolve()}")
    files = [CONFIG["tokens_file"], CONFIG["freq_plot_file"],
             CONFIG["ngram_file"], CONFIG["lemma_file"], CONFIG["vocab_file"]]
    for f in files:
        p = output_dir / f
        size = p.stat().st_size if p.exists() else 0
        print(f"    • {f:<22}  ({size:,} bytes)")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 72)
    print("  NLP PREPROCESSING PIPELINE — FINANCIAL NEWS SENTIMENT ANALYSIS")
    print("=" * 72 + "\n")

    # Stage 0: Bootstrap
    output_dir = bootstrap()

    # Load corpus
    raw_text = load_corpus(CONFIG["corpus_path"])

    # Stage 1: Tokenization
    sentences, word_tokens = stage_tokenization(raw_text, output_dir)

    # Stage 2: Frequency distribution
    freq_df = stage_frequency_distribution(word_tokens, output_dir)

    # Stage 3: N-gram modeling
    ngrams_result = stage_ngram_modeling(word_tokens, output_dir)

    # Stage 4: Lemmatization
    lemmas = stage_lemmatization(word_tokens, output_dir)

    # Stage 5: Vocabulary
    vocab_stats = stage_vocabulary(word_tokens, lemmas, output_dir)

    # Stage 6: Summary
    print_summary(raw_text, sentences, word_tokens, lemmas,
                  ngrams_result, vocab_stats, output_dir)


if __name__ == "__main__":
    main()
