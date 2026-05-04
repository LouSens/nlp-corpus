# NLP PREPROCESSING PIPELINE — SETUP & RUN GUIDE

**Application:** Financial News Sentiment Analysis  
**Assignment:** Specialized Corpus & NLP Preprocessing  
**Corpus Method:** Web Scraping (no Kaggle dataset required)

---

## Task 1: Specialized Corpus

### 1.1 Description of the Corpus

This corpus is a **Financial News Sentiment Corpus** — a domain-specific, English-language text collection compiled from publicly available financial news sources and encyclopaedia-style financial articles. The corpus was assembled through automated web scraping of real-time RSS news feeds (Yahoo Finance, Reuters Business, MarketWatch, CNBC, Investing.com, Seeking Alpha, and Nasdaq) as well as structured prose articles from Wikipedia covering core finance and economics topics.

The corpus contains a minimum of **25,000 words** of clean, deduplicated text spanning topics such as market movements, corporate earnings, macroeconomic indicators, investment strategies, inflation, interest rates, and financial risk. All text is in formal written English, reflecting the register and vocabulary typical of professional financial reporting and economic discourse. Duplicate content is removed using SHA-1 content hashing to ensure unique documents.

### 1.2 Description of the Intended Application

The intended application for this corpus is **Financial News Sentiment Analysis** — a Natural Language Processing (NLP) task that automatically classifies the sentiment (positive, negative, or neutral) expressed in financial news articles, headlines, and summaries. The system is designed to assist investors, analysts, and automated trading systems in gauging market sentiment from news content without manually reading large volumes of daily financial reports.

The application works by training or fine-tuning a machine learning model (e.g., a Naive Bayes classifier, LSTM, or a transformer-based model such as FinBERT) on labelled financial text. Once trained, the model can process new financial news in real time and output a sentiment label, which can be used to inform investment decisions, monitor market risk, or trigger trading signals.

### 1.3 Justification for Using a Specialized Corpus

A general-purpose corpus (e.g., Wikipedia dumps or books corpora) is **insufficient** for financial sentiment analysis for the following reasons:

1. **Domain-specific vocabulary:** Financial text contains highly specialised terminology — terms such as *quantitative easing*, *yield curve inversion*, *EBITDA*, *bearish divergence*, and *margin call* carry precise meanings within finance that are absent or misrepresented in general corpora. A model trained on general text will fail to correctly interpret these terms.

2. **Sentiment is context-dependent:** The word *"growth"* in everyday language is almost always positive. In financial reporting, however, *"slowing growth"* or *"growth below expectations"* carries a negative sentiment. Sentiment polarity in finance is driven by contextual comparison against forecasts, benchmarks, and market expectations — nuances that only appear in domain-specific text.

3. **Writing style and register:** Financial news is written in a formal, precise, hedged register distinct from general journalism or social media. NLP models must be exposed to this style during training to generalise correctly to unseen financial articles.

4. **Topic coverage:** A general corpus would not provide sufficient coverage of financial events (earnings reports, central bank decisions, commodity movements) that are the primary subjects a sentiment analysis model must understand.

5. **Noise reduction:** Sourcing text directly from reputable financial outlets and encyclopaedic finance articles ensures the corpus has low noise, high factual density, and consistent domain relevance — all properties critical for building a reliable sentiment analysis model.

For these reasons, a specialized financial news corpus is not merely preferable — it is a **prerequisite** for building a sentiment analysis model that performs meaningfully in a financial context.

---

## Task 2: Corpus Materials and Sources

### 2.1 Target Materials

The target materials for this corpus are:

- **Financial news headlines and summaries** sourced from major financial news RSS feeds
- **Encyclopaedic financial articles** from Wikipedia covering foundational concepts in economics, investing, and financial markets

These material types were chosen because they collectively represent both **current market discourse** (news) and **stable domain knowledge** (encyclopaedic articles), providing the model with both the vocabulary of real-world financial events and the conceptual grounding of financial theory.

### 2.2 Why These Materials Are Necessary

Financial sentiment analysis requires text that is:

- **Representative of real analyst and journalist language** — so the model learns the actual vocabulary and phrasing used in professional financial communication.
- **Topically diverse within the financial domain** — covering stocks, macroeconomics, commodities, corporate performance, and monetary policy — so the model generalises across different types of financial news rather than overfitting to one sub-domain.
- **Consistently formal** — so the model is not confused by the informal registers of social media or consumer forums that are absent from the real-world deployment environment.

RSS news feeds fulfil the first two criteria by providing high-frequency, multi-topic, professionally written financial reporting. Wikipedia finance articles fulfil the third criterion and additionally ensure the corpus exceeds the 25,000-word minimum, providing stable, high-density domain vocabulary that enriches the model's lexical coverage.

### 2.3 Sources and Per-Source Justification

| # | Source | Material Type | Justification |
|---|--------|---------------|---------------|
| 1 | **Yahoo Finance** (RSS) | Headlines & summaries | Yahoo Finance is one of the most widely read English-language financial news aggregators globally. Its RSS feed delivers a high volume of timely market news covering stocks, commodities, and macroeconomic events. It provides broad market coverage that reflects the type of text a sentiment model would encounter in real-world deployment. |
| 2 | **Reuters Business** (RSS) | Headlines & summaries | Reuters is a globally trusted, editorially rigorous wire service. Its business and finance reporting is widely cited by institutional investors and regulators. Including Reuters ensures the corpus contains authoritative, neutral-register financial text representative of institutional-grade reporting. |
| 3 | **MarketWatch** (RSS) | Headlines & summaries | MarketWatch specialises in real-time financial markets coverage, including stock market updates, earnings announcements, and economic indicators. It is particularly valuable because it covers fast-moving market events — exactly the type of content where sentiment classification adds the most analytical value. |
| 4 | **CNBC Finance / Economy / Earnings** (RSS) | Headlines & summaries | CNBC is a leading financial broadcast and digital news organisation. Its three sub-feeds (Finance, Economy, Earnings) provide topically segmented coverage, ensuring the corpus captures sentiment expressed across different financial contexts — general markets, macroeconomic policy, and corporate performance — rather than a single topic category. |
| 5 | **Investing.com** (RSS) | Headlines & summaries | Investing.com covers global financial markets including forex, commodities, indices, and cryptocurrencies. This broadens the corpus beyond US equity markets, making the resulting sentiment model more generalisable to multi-asset financial contexts. |
| 6 | **Seeking Alpha** (RSS) | Headlines & summaries | Seeking Alpha publishes both professional analyst commentary and investor opinion pieces. This introduces a range of sentiment-expressive writing styles — from cautiously hedged professional analysis to more opinionated investor perspectives — increasing the diversity of sentiment expression patterns in the corpus. |
| 7 | **Nasdaq** (RSS) | Headlines & summaries | Nasdaq's news feed is directly tied to the performance and listings of technology and growth-oriented equities. Given that technology stocks are among the most sentiment-sensitive assets in modern markets, Nasdaq's feed contributes a valuable and distinct sector-specific subset of financial language to the corpus. |
| 8 | **Wikipedia Finance Articles** (~39 articles, prose) | Encyclopaedic article prose | Wikipedia finance articles (e.g., *Stock market*, *Inflation*, *Monetary policy*, *Bond market*) provide structured, authoritative definitional text covering the core conceptual vocabulary of financial analysis. These articles contribute the bulk of the word count (each article typically contains 2,000–8,000 words) and ensure the corpus is lexically rich in established financial terminology, grounding the model's vocabulary in stable domain knowledge rather than purely event-driven news language. |

### 2.4 Summary of Material Necessity

Together, these eight sources provide:

- **Breadth** — coverage across stocks, bonds, commodities, forex, macroeconomics, corporate earnings, and monetary policy
- **Depth** — from high-frequency real-time news summaries to long-form encyclopaedic prose
- **Linguistic diversity** — from neutral wire-service reporting (Reuters) to opinion-inflected analysis (Seeking Alpha)
- **Sufficient volume** — collectively exceeding 25,000 words of clean, deduplicated domain-specific text

No single source alone would satisfy all these criteria. The combination of RSS news feeds and Wikipedia articles is therefore deliberate and necessary to produce a corpus that is representative, diverse, and of sufficient size for effective NLP model training.

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
