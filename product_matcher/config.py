"""
config.py
---------
Central configuration for the product matcher.

Contains:
- MatcherConfig dataclass with all tunable thresholds and weights

This is the **general-purpose** version: no domain-specific brand aliases,
stopwords, or phone-specific constants.  Feature extraction is handled by
the LLM extractor (when enabled) or left to fuzzy + semantic similarity.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_LOCAL_MODEL: str = os.getenv(
    "PRODUCT_MATCHER_LOCAL_MODEL",
    "intfloat/multilingual-e5-large",
)


@dataclass
class MatcherConfig:
    """All tuneable knobs for the matching pipeline."""

    # Column mapping
    name_col: str = "name"
    price_col: str = "price"
    seller_col: Optional[str] = None
    id_col: Optional[str] = None
    currency_col: Optional[str] = None
    gtin_col: Optional[str] = None
    upc_col: Optional[str] = None
    ean_col: Optional[str] = None
    mpn_col: Optional[str] = None
    sku_col: Optional[str] = None

    # Matching behaviour
    match_mode: str = "exact"           # "exact" | "family"
    use_semantic: bool = True
    local_model_name: str = DEFAULT_LOCAL_MODEL
    semantic_weight: float = 0.50
    semantic_match_threshold: float = 0.90

    # Decision thresholds
    match_threshold: float = 0.82
    review_threshold: float = 0.68

    # Feature weights (general-purpose, heavier on fuzzy + semantic)
    exact_name_weight: float = 0.10
    fuzzy_weight: float = 0.25
    token_weight: float = 0.10
    brand_weight: float = 0.10
    model_weight: float = 0.10
    variant_weight: float = 0.05
    storage_weight: float = 0.05
    specs_weight: float = 0.10
    trade_id_weight: float = 0.35
    mpn_weight: float = 0.18
    sku_weight: float = 0.12

    # Exact-mode strictness
    strict_storage_for_exact: bool = True
    strict_variant_for_exact: bool = True

    # Blocking
    max_bucket_size: int = 200

    # LLM feature extraction (general-purpose, replaces hardcoded constants)
    llm_extract: bool = False
    llm_extract_model: str = "gpt-4o-mini"

    # LLM fallback for uncertain pairs
    llm_resolve: bool = False
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_pairs: int = 50

    # Cache
    cache_path: str = ".embedding_cache.sqlite3"
