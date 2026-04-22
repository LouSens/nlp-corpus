# NLP PREPROCESSING PIPELINE — SETUP & RUN GUIDE

**Application:** Financial News Sentiment Analysis  
**Assignment:** Specialized Corpus & NLP Preprocessing

---

## Folder Structure

```text
nlp-corpus/
├── corpus.txt             ← Built by Step 1 (or pre-existing)
├── corpus_builder.py      ← Step 1: Merges source files
├── nlp_preprocessing.py   ← Step 2: Full preprocessing pipeline
├── requirements.txt       ← Python dependencies
├── sources/               ← Kaggle Financial PhraseBank dataset
│   ├── Sentences_AllAgree.txt
│   └── Sentences_50Agree.txt
└── nlp_output/            ← AUTO-CREATED by pipeline
    ├── tokens.txt
    ├── freq_plot.png
    ├── n-gram.txt
    ├── lemma.txt
    └── vocab.txt
```

---

## STEP 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

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

## STEP 3 — Add Source Files

1. Download the **Kaggle Financial PhraseBank** dataset.
2. Place the following files inside the `sources/` folder:
   - `Sentences_AllAgree.txt`
   - `Sentences_50Agree.txt`
3. *Note: Combined total must exceed 25,000 words.*

---

## STEP 4 — Build `corpus.txt`

```bash
python corpus_builder.py
```

This merges the PhraseBank source files into a single `corpus.txt` and prints a word count validation (must be >= 25,000 words).

---

## STEP 5 — Run Preprocessing Pipeline

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
