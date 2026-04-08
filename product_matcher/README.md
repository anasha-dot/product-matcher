# Product Matcher — General-Purpose Product Deduplicator

## The Assignment

> **Challenge:** You receive a list of products with duplicates and non-uniform names
> (e.g. `Samsung S23` and `סמסונג גלקסי 23`).
> Write a script that merges duplicates and ensures the **lowest price** is displayed to the customer.

## What This Program Does

This tool takes a messy product catalogue — where the same product may appear
under different names, languages, or with typos — and **groups all duplicates
together**, picking the lowest price from each group.

It works for **any product category** (phones, laptops, headphones, TVs, etc.)
in **any language** without hardcoded brand or model dictionaries.

### How It Works (High Level)

1. **Normalize** every product name: lowercase, strip diacritics, unify
   Arabic/Hebrew script forms, clean separators, normalise storage units.
2. **Block** products into candidate pairs using fingerprint-based hash keys
   so we don't compare every row against every other row.
3. **Score** each candidate pair with a weighted combination of:
   - Exact normalised name match
   - Fuzzy string similarity (token-sort ratio + partial ratio)
   - Token Jaccard overlap
   - Storage match signal (extracted from text)
   - Semantic similarity (sentence-transformer embeddings or TF-IDF fallback)
   - Trade-ID / MPN / SKU exact match
4. **Decide**: if the score exceeds a *match threshold* the pair is merged;
   if it falls in a *review zone* it is sent to the **LLM resolver** (if enabled)
   or flagged for human review; otherwise rejected.
5. **Cluster** matched pairs with Union-Find and output the groups with the
   lowest price highlighted.

### Two Modes of Operation

| Mode | Description |
|---|---|
| **Without LLM** | Uses fuzzy string matching, token overlap, semantic similarity (sentence-transformers or TF-IDF). No API key needed. Works well for most cases. |
| **With LLM resolver** (`--llm-resolve`) | Sends uncertain pairs (review zone) to an LLM for automatic resolution. Most accurate for edge cases. Requires an OpenAI API key. |

## Project Structure

```
product_matcher/
├── __init__.py        # Package exports
├── __main__.py        # Entry point for `python -m product_matcher`
├── config.py          # MatcherConfig dataclass (thresholds, weights, flags)
├── models.py          # Data classes: ProductRecord, PairDecision
├── normalize.py       # General text normalization, tokenization, blocking keys
├── embeddings.py      # Semantic similarity backends (sentence-transformers + TF-IDF)
├── llm_resolver.py    # LLM fallback for uncertain pairs
├── matcher.py         # Core matching engine: scoring, blocking, clustering
├── io_utils.py        # File I/O helpers and built-in sample data
├── cli.py             # Command-line interface and main()
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### What Each File Does

| File | Purpose |
|---|---|
| **config.py** | `MatcherConfig` dataclass with all thresholds, feature weights, and LLM resolver flags. No hardcoded brand or product dictionaries. |
| **models.py** | `ProductRecord` and `PairDecision` (match/review/reject verdict). |
| **normalize.py** | Language-level text processing: Unicode normalization, Arabic/Hebrew script unification, separator cleanup, storage-unit normalization. Generic tokenization and fingerprint-based blocking keys. |
| **embeddings.py** | Two semantic similarity backends — multilingual sentence-transformer with SQLite cache, and character n-gram TF-IDF fallback. |
| **llm_resolver.py** | LLM fallback for uncertain pairs: asks "are these the same product?" and merges confirmed matches. |
| **matcher.py** | Core `ProductMatcher` class: loads records, generates candidate pairs, computes weighted scores, clusters with Union-Find, builds output. |
| **io_utils.py** | File I/O (CSV, Excel, JSON) and built-in sample data (phones + laptops + headphones). |
| **cli.py** | Full CLI with column mapping, tuning, and LLM resolver flags. |

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` and `openai` are optional.  Without them the
> program falls back to TF-IDF similarity and skips the LLM resolver.

## Usage

### Demo (built-in sample data, no API key needed)

```bash
python -m product_matcher --demo --disable-semantic
```

### With LLM resolver (best accuracy for edge cases)

```bash
export OPENAI_API_KEY="sk-..."
python -m product_matcher \
    --input products.csv \
    --llm-resolve
```

### Real data with column mapping

```bash
python -m product_matcher \
    --input products.csv \
    --output matched_products.json \
    --reviews-output review_pairs.json \
    --name-col "product_name" \
    --price-col "price" \
    --seller-col "seller"
```

### As a library

```python
import pandas as pd
from product_matcher import ProductMatcher, MatcherConfig

df = pd.read_csv("products.csv")
config = MatcherConfig(
    name_col="product_name",
    price_col="price",
    seller_col="seller",
    match_mode="exact",
    llm_resolve=True,          # use LLM for uncertain pairs
    llm_api_key="sk-...",
)
matcher = ProductMatcher(config)
clusters, reviews = matcher.run(df)
matcher.close()
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to CSV / XLSX / JSON file |
| `--output` | `matched_products.json` | Where to write matched clusters |
| `--reviews-output` | `review_pairs.json` | Where to write uncertain pairs |
| `--mode` | `exact` | `exact` = same SKU, `family` = same product line |
| `--semantic-backend` | `local` | `local` (sentence-transformers) or `tfidf` |
| `--disable-semantic` | off | Skip semantic similarity entirely |
| `--match-threshold` | `0.82` | Score above this = automatic match |
| `--review-threshold` | `0.68` | Score in review-match range = flagged |
| `--llm-resolve` | off | Use LLM to resolve uncertain pairs |
| `--llm-api-key` | `$OPENAI_API_KEY` | OpenAI API key |
| `--llm-model` | `gpt-4o-mini` | Which OpenAI model to use |
| `--demo` | off | Run on built-in sample data |

## LLM Pair Resolution

When `--llm-resolve` is enabled, pairs in the "review zone" (score 0.68-0.82)
are sent to the LLM with a yes/no question.  Confirmed matches are merged;
rejected pairs stay in the review file.

## Output Format

`matched_products.json` contains a list of clusters:

```json
[
  {
    "cluster_id": 1,
    "canonical_name": "Samsung Galaxy S23",
    "lowest_price": 3100,
    "offer_count": 3,
    "offers": [
      {"name": "סמסונג גלקסי 23", "price": 3100, "brand": "samsung", "model": "galaxy s23", "...": "..."},
      {"name": "Samsung Galaxy S23", "price": 3150, "brand": "samsung", "model": "galaxy s23", "...": "..."}
    ]
  }
]
```

## Supported Product Types

The matcher works for **any** product category using fuzzy matching and
semantic similarity: phones, laptops, tablets, headphones, TVs, smartwatches,
cameras, monitors, etc.

## Supported Languages

Multilingual out of the box — the normalizer handles Arabic and Hebrew script,
and the multilingual sentence-transformer model understands 100+ languages.
