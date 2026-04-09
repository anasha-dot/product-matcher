# Product Matcher

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
2. **Extract features** (optional, with `--llm-extract`): send product names
   to an LLM which returns structured fields — brand, model, variant, specs
   (storage, RAM, color, …), and product category. Results are cached in
   SQLite so the same name is never extracted twice.
3. **Block** products into candidate pairs using fingerprint-based hash keys
   so we don't compare every row against every other row.
4. **Score** each candidate pair with a weighted combination of:
   - Exact normalised name match
   - Fuzzy string similarity (token-sort ratio + partial ratio)
   - Token Jaccard overlap
   - Brand / model / variant / storage match signals (from LLM extraction)
   - Specs overlap (how many extracted attributes agree)
   - Semantic similarity (multilingual sentence-transformer embeddings)
   - Trade-ID / MPN / SKU exact match
5. **Apply hard conflict rules**: different brands, different models,
   different storage sizes, different numeric tokens, or different categories
   → automatic rejection, regardless of similarity score.
6. **Decide**: if the score exceeds a _match threshold_ the pair is merged;
   if it falls in a _review zone_ it is sent to the **LLM resolver** (if enabled)
   or flagged for human review; otherwise rejected.
7. **Cluster** matched pairs with Union-Find and output the groups with the
   lowest price highlighted.

### Three Modes of Operation

| Mode                                                                 | Description                                                                                                                      |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Without LLM** (default)                                            | Uses fuzzy string matching, token overlap, and semantic similarity. No API key needed. Works well for most cases.                |
| **With LLM extraction** (`--llm-extract`)                            | LLM extracts structured fields per product. Most accurate, works across languages and product types. Requires an OpenAI API key. |
| **With LLM extraction + resolution** (`--llm-extract --llm-resolve`) | Full pipeline: extraction + uncertain pair resolution. Maximum accuracy.                                                         |

## Design : Accuracy First, Cost-Efficiently

**Accuracy is the top priority.** An incorrect match — like merging
"Samsung S23" with "Samsung S23 Ultra" — shows the customer a wrong price
and damages trust. The program achieves maximum accuracy through a
combination of **AI-powered extraction** and **hard deterministic rules**,
while keeping costs minimal through caching and batching.

### How Accuracy Is Achieved

The program uses multiple layers that all work together on every pair:

1. **LLM extraction** — an AI model parses every product name and extracts
   structured fields (brand, model, variant, specs, category) in any
   language. This is what makes the system general-purpose: no hardcoded
   brand dictionaries, no product-specific regex. The LLM understands
   that "סמסונג גלקסי S23 אולטרה" means `{brand: "samsung", model:
"galaxy s23", variant: "ultra"}` without any manual configuration.

2. **Semantic similarity** — a multilingual sentence-transformer model
   [`intfloat/multilingual-e5-large`]
   computes embeddings for every product name locally — no API calls
   needed. This is a 560M-parameter model from Microsoft/INTFLOAT that
   supports 100+ languages, including Hebrew and Arabic.
   Embeddings are cached in SQLite so they are only computed once per unique name.

3. **Fuzzy string matching** — token-sort ratio and partial ratio
   (via `rapidfuzz`) catch typos, spacing differences, and minor naming
   variations (e.g. "Dell XPS15" vs "Dell XPS 15"). Token Jaccard overlap
   measures how many words two names share.

4. **Hard deterministic rules** — after all scores are computed, the
   program applies strict conflict checks that **cannot be overridden**
   by any similarity score. These rules act as a safety net against both
   LLM hallucinations and edge cases:

   | Rule                       | What it catches                     |
   | -------------------------- | ----------------------------------- |
   | **brand_conflict**         | Samsung ≠ Apple → reject, always    |
   | **model_conflict**         | Galaxy S23 ≠ Galaxy S24 → reject    |
   | **variant_conflict**       | Pro ≠ Ultra → reject                |
   | **storage_conflict**       | 128GB ≠ 256GB → reject              |
   | **category_conflict**      | phone ≠ headphones → reject         |
   | **numeric_token_conflict** | different numbers in names → reject |
   | **specs_conflict**         | RAM 8GB ≠ RAM 16GB → reject         |
   | **trade_id / MPN / SKU**   | different identifiers → reject      |

### Why Not Use a Full LLM-Only Solution?

- **Hallucination** — LLMs can return incorrect results: a wrong brand, a
  missed variant word, or inconsistent specs. With a full LLM solution
  there is no safety net to catch these errors. In our approach, the hard
  rules run **after** the LLM and override any hallucinated output.
- **Cost at scale** — sending full product lists (with pairwise comparison
  context) to an LLM is far more expensive than extracting structured
  fields once per product and comparing them locally.
- **Determinism** — the same input should always produce the same output.
  LLMs are non-deterministic by nature; hard rules are not. By using the
  LLM only for extraction and letting deterministic rules make the final
  match/reject decision, the system produces consistent, reproducible
  results.

### Cost Efficiency Through Caching

Accuracy does not have to be expensive. Every LLM extraction result is
**cached in SQLite** — each unique product name is sent to the API exactly
**once, ever**. Re-runs, daily updates, and incremental imports only pay
for genuinely new names.

| Dataset size       | Unique names | One-time cost | Re-run cost |
| ------------------ | ------------ | ------------- | ----------- |
| 100 products       | ~100         | ~$0.003       | $0 (cached) |
| 10,000 products    | ~4,000       | ~$0.08        | $0 (cached) |
| 100,000 products   | ~30,000      | ~$0.60        | $0 (cached) |
| 1,000,000 products | ~200,000     | ~$4.00        | $0 (cached) |

Costs are based on `gpt-4o-mini` pricing with batched extraction (20
products per API call).

## Project Structure

```
product_matcher/
├── __init__.py        # Package exports
├── __main__.py        # Entry point for `python -m product_matcher`
├── config.py          # MatcherConfig dataclass (thresholds, weights, flags)
├── models.py          # Data classes: ProductRecord, PairDecision
├── normalize.py       # General text normalization, tokenization, blocking keys
├── embeddings.py      # Semantic similarity (multilingual sentence-transformers)
├── llm_extractor.py   # LLM-based structured feature extraction (brand, model, specs)
├── llm_resolver.py    # LLM fallback for uncertain pairs
├── matcher.py         # Core matching engine: scoring, blocking, clustering
├── io_utils.py        # File I/O helpers and built-in sample data
├── cli.py             # Command-line interface and main()
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### What Each File Does

| File                 | Purpose                                                                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **config.py**        | `MatcherConfig` dataclass with all thresholds, feature weights, and LLM flags. No hardcoded brand or product dictionaries.                                                                        |
| **models.py**        | `ProductRecord` (enriched row with brand, model, variant, specs, category) and `PairDecision` (match/review/reject verdict).                                                                      |
| **normalize.py**     | Language-level text processing: Unicode normalization, Arabic/Hebrew script unification, separator cleanup, storage-unit normalization. Generic tokenization and fingerprint-based blocking keys. |
| **embeddings.py**    | Multilingual sentence-transformer semantic similarity with SQLite embedding cache.                                                                                                                |
| **llm_extractor.py** | Batch LLM extraction: sends product names to an OpenAI model, gets back structured JSON (brand, model, variant, specs, category). Caches results in SQLite.                                       |
| **llm_resolver.py**  | LLM fallback for uncertain pairs: asks "are these the same product?" and merges confirmed matches.                                                                                                |
| **matcher.py**       | Core `ProductMatcher` class: loads records, runs LLM extraction, generates candidate pairs, computes weighted scores (including specs overlap), clusters with Union-Find, builds output.          |
| **io_utils.py**      | File I/O (CSV, Excel, JSON) and built-in sample data (phones + laptops + headphones).                                                                                                             |
| **cli.py**           | Full CLI with column mapping, tuning, and LLM flags.                                                                                                                                              |

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` is required for multilingual matching.
> `openai` is only needed when using `--llm-extract` or `--llm-resolve`.

## Usage

### Demo (built-in sample data, no API key needed)

```bash
python -m product_matcher --demo --disable-semantic
```

### With LLM extraction (best accuracy)

```bash
export OPENAI_API_KEY="sk-..."
python -m product_matcher \
    --input products.csv \
    --llm-extract \
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
    --seller-col "seller" \
    --llm-extract
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
    llm_extract=True,          # use LLM for feature extraction
    llm_resolve=True,          # use LLM for uncertain pairs
    llm_api_key="sk-...",
)
matcher = ProductMatcher(config)
clusters, reviews = matcher.run(df)
matcher.close()
```

### Key CLI Flags

| Flag                 | Default                 | Description                                      |
| -------------------- | ----------------------- | ------------------------------------------------ |
| `--input`            | _(required)_            | Path to CSV / XLSX / JSON file                   |
| `--output`           | `matched_products.json` | Where to write matched clusters                  |
| `--reviews-output`   | `review_pairs.json`     | Where to write uncertain pairs                   |
| `--mode`             | `exact`                 | `exact` = same SKU, `family` = same product line |
| `--disable-semantic` | off                     | Skip semantic similarity entirely                |
| `--match-threshold`  | `0.82`                  | Score above this = automatic match               |
| `--review-threshold` | `0.68`                  | Score in review-match range = flagged            |
| `--llm-extract`      | off                     | Use LLM to extract brand/model/specs from names  |
| `--llm-resolve`      | off                     | Use LLM to resolve uncertain pairs               |
| `--llm-api-key`      | `$OPENAI_API_KEY`       | OpenAI API key                                   |
| `--llm-model`        | `gpt-4o-mini`           | Which OpenAI model to use                        |
| `--demo`             | off                     | Run on built-in sample data                      |

## LLM Feature Extraction

When `--llm-extract` is enabled, product names are sent in batches to an
OpenAI model which returns structured JSON for each:

```json
{
  "brand": "dell",
  "model": "xps 15",
  "variant": null,
  "specs": { "storage": "512gb", "ram": "16gb" },
  "category": "laptop"
}
```

These fields are used for:

- **Hard conflicts**: different brand/model/storage/specs = different products
- **Feature scoring**: brand match, model match, specs overlap
- **Blocking**: products are grouped by brand+model for efficient comparison

Results are cached in SQLite, so re-running on the same data costs nothing.

## LLM Pair Resolution

When `--llm-resolve` is enabled, pairs in the "review zone" (score 0.68-0.82)
are sent to the LLM with a yes/no question. Confirmed matches are merged;
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
      { "name": "סמסונג גלקסי 23", "price": 3100, "brand": "samsung", "model": "galaxy s23", "...": "..." },
      { "name": "Samsung Galaxy S23", "price": 3150, "brand": "samsung", "model": "galaxy s23", "...": "..." }
    ]
  }
]
```

## Supported Product Types

With `--llm-extract`, the matcher works for **any** product category:
phones, laptops, tablets, headphones, TVs, smartwatches, cameras, monitors, etc.

## Supported Languages

Multilingual out of the box — the normalizer handles Arabic and Hebrew script,
and the multilingual sentence-transformer model understands 100+ languages.
With `--llm-extract`, the LLM translates non-English names to English
structured fields automatically.
