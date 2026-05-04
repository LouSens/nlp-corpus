# NLP PREPROCESSING PIPELINE — SETUP & RUN GUIDE

**Application:** Financial News Sentiment Analysis  
**Assignment:** Specialized Corpus & NLP Preprocessing  
**Corpus Method:** Web Scraping (no Kaggle dataset required)

---

## Folder Structure

```text
nlp-corpus/
├── corpus.txt             ← AUTO-BUILT by Step 3 (web scraping)
├── corpus_builder.py      ← Step 3: Scrapes & builds corpus.txt
├── nlp_preprocessing.py   ← Step 4: Full preprocessing pipeline
├── requirements.txt       ← Python dependencies
└── nlp_output/            ← AUTO-CREATED by pipeline
    ├── tokens.txt
    ├── freq_plot.png
    ├── n-gram.txt
    ├── lemma.txt
    └── vocab.txt
```

---

## Corpus Sources (All Free, No API Key Required)

| Source | Type | How accessed |
|--------|------|-------------|
| Yahoo Finance | RSS feed headlines + summaries | Public RSS URL |
| Reuters Business | RSS feed | Public RSS URL |
| MarketWatch | RSS feed | Public RSS URL |
| CNBC Finance / Economy / Earnings | RSS feed | Public RSS URL |
| Investing.com | RSS feed | Public RSS URL |
| Seeking Alpha | RSS feed | Public RSS URL |
| Nasdaq | RSS feed | Public RSS URL |
| Wikipedia (~39 finance articles) | Article prose | `action=raw` (no key) |

> **Total target: ≥ 25,000 words** — met primarily by Wikipedia articles with RSS supplementing.

---

## STEP 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

> No extra packages are needed beyond what's already in `requirements.txt`.  
> The scraper uses only Python standard library (`urllib`, `xml.etree.ElementTree`, `html.parser`).

---

## STEP 2 — Download NLTK Data

```bash
python -c "
import nltk
for pkg in ['punkt','punkt_tab','averaged_perceptron_tagger','wordnet','stopwords','omw-1.4']:
    nltk.download(pkg)
"
```

---

## STEP 3 — Build `corpus.txt` via Web Scraping

```bash
python corpus_builder.py
```

This script will:
1. **Phase 1** — Fetch headlines + summaries from ~13 financial RSS feeds
2. **Phase 2** — Scrape ~39 Wikipedia finance articles (raw wikitext, cleaned to prose)
3. Deduplicate all text (SHA-1 content hashing)
4. Assemble and write `corpus.txt`
5. Validate ≥ 25,000 words

Expected runtime: **3–5 minutes** (polite 1.5 s delay between requests).

> **If some RSS feeds are blocked** on your network, Wikipedia articles alone provide well over 25,000 words.

---

## STEP 4 — Run Preprocessing Pipeline

```bash
python nlp_preprocessing.py
```

**Stages executed by the script:**
- **Stage 0** → Bootstrap (NLTK check, output dir)
- **Stage 1** → Tokenization       → `nlp_output/tokens.txt`
- **Stage 2** → Freq. Distribution → `nlp_output/freq_plot.png`
- **Stage 3** → N-gram Modeling    → `nlp_output/n-gram.txt`
- **Stage 4** → Lemmatization      → `nlp_output/lemma.txt`
- **Stage 5** → Vocabulary         → `nlp_output/vocab.txt`
- **Stage 6** → Summary report printed to console

---

## Customizing Selected Tokens (Stage 2)

After **Stage 1** runs, check `nlp_output/tokens.txt` to see what words appear in your corpus. Then edit `nlp_preprocessing.py`:

```python
CONFIG = {
    ...
    "selected_tokens": [
        "market", "revenue", "inflation", "growth",
        "investment", "profit", "risk", "interest",
        "economy", "forecast"
    ],
}
```

Replace these 10 words with domain-relevant tokens from **YOUR** specific corpus before re-running.

---

## Switching BIGRAM ↔ TRIGRAM

In `nlp_preprocessing.py`, simply change:
```python
"ngram_type": "bigram"   # change to "trigram"
```

---

## If You Need More Words

Add more entries to `WIKIPEDIA_ARTICLES` in `corpus_builder.py`:

```python
WIKIPEDIA_ARTICLES = [
    ...
    "Capital_market",
    "Commodity_market",
    "Derivatives_market",
    ...
]
```

Or add extra RSS feed URLs to `RSS_FEEDS`. Each Wikipedia article adds ~2,000–8,000 words.

---

## Report Screenshots Guide

Task 3 requires code screenshots. Recommended approach:

1. Open `nlp_preprocessing.py` in VS Code or any IDE.
2. Screenshot the relevant function for each step:
   - **Stage 1 function** → screenshot for Tokenization
   - **Stage 2 function** → screenshot for Freq. Distribution
   - **Stage 3 function** → screenshot for N-gram
   - **Stage 4 function** → screenshot for Lemmatization
3. Run the script and screenshot the terminal output per stage.
4. Insert `freq_plot.png` directly into your report (Task 3, Step 2).
