"""
normalize.py
------------
General-purpose text normalization and tokenization helpers.

This module contains ONLY language-level normalization (Unicode, Arabic/Hebrew
script, separators, storage units).

Contains:
- Unicode / diacritic / script normalization (Arabic, Hebrew)
- Separator and unit normalization (GB, TB, MB — multilingual)
- Generic tokenization (no hardcoded stopwords)
- Fingerprint-based blocking-key generation
- Row-to-ProductRecord conversion
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Set

import pandas as pd

from .config import MatcherConfig
from .models import ProductRecord

# ---------------------------------------------------------------------------
# Low-level text helpers (linguistic, not domain-specific)
# ---------------------------------------------------------------------------

ARABIC_CHAR_NORMALIZATION = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي",
})

HEBREW_FINAL_FORMS = str.maketrans({
    "ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ",
})


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def basic_script_normalize(text: str) -> str:
    text = text.translate(ARABIC_CHAR_NORMALIZATION)
    text = text.translate(HEBREW_FINAL_FORMS)
    return text


def normalize_separators(text: str) -> str:
    text = re.sub(r"[\-_/|]+", " ", text)
    text = re.sub(r"[\[\](){}:;,+]+", " ", text)
    text = re.sub(r"[^\w\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_units(text: str) -> str:
    """Normalise storage/memory units to GB across English, Hebrew, Arabic."""
    # GB variants
    text = re.sub(r"(\d+)\s*(gb|ג[י׳']?ב|جيجا|جيجابايت)", r"\1gb", text)
    # TB -> GB
    text = re.sub(
        r"(\d+)\s*(tb|ט[י׳']?ב|تيرا|تيرابايت)",
        lambda m: f"{int(m.group(1)) * 1024}gb",
        text,
    )
    # MB variants (keep as-is but normalize spelling)
    text = re.sub(r"(\d+)\s*(mb|מ[י׳']?ב|ميجا|ميجابايت)", r"\1mb", text)
    return text


def normalize_name(name: str) -> str:
    """General-purpose normalization pipeline for a product name."""
    text = (name or "").lower().strip()
    text = strip_diacritics(text)
    text = basic_script_normalize(text)
    text = normalize_separators(text)
    text = normalize_units(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Identifier normalization
# ---------------------------------------------------------------------------

def normalize_identifier(value: object, digits_only: bool = False) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if digits_only:
        digits = re.sub(r"\D", "", text)
        return digits or None
    text = strip_diacritics(text).lower().strip()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text or None


def get_trade_id_from_row(row: pd.Series, config: MatcherConfig) -> Optional[str]:
    for col_name in (config.gtin_col, config.upc_col, config.ean_col):
        if col_name and col_name in row and pd.notna(row[col_name]):
            normalized = normalize_identifier(row[col_name], digits_only=True)
            if normalized:
                return normalized
    return None


def get_optional_identifier(
    row: pd.Series, col_name: Optional[str], digits_only: bool = False,
) -> Optional[str]:
    if not col_name or col_name not in row or pd.isna(row[col_name]):
        return None
    return normalize_identifier(row[col_name], digits_only=digits_only)


# ---------------------------------------------------------------------------
# Tokenization (no hardcoded stopwords — all tokens kept)
# ---------------------------------------------------------------------------

def get_tokens(text: str) -> Set[str]:
    """Split normalised text into token set.  No stopword filtering."""
    return {tok for tok in text.split() if tok}


# ---------------------------------------------------------------------------
# Blocking keys — general fingerprint-based approach
# ---------------------------------------------------------------------------

def build_block_keys(
    normalized_name: str,
    brand: Optional[str],
    model: Optional[str],
    tokens: Set[str],
    trade_id: Optional[str] = None,
    mpn: Optional[str] = None,
    sku: Optional[str] = None,
    seller: Optional[str] = None,
    category: Optional[str] = None,
) -> Set[str]:
    keys: Set[str] = set()
    significant = sorted(tokens)

    # Identifier-based keys (always strong)
    if trade_id:
        keys.add(f"trade_id:{trade_id}")
    if mpn:
        keys.add(f"mpn:{mpn}")
    if sku and seller:
        keys.add(f"seller_sku:{seller.lower()}:{sku}")

    # LLM-extracted structured keys
    if brand:
        keys.add(f"brand:{brand}")
    if brand and model:
        keys.add(f"brand_model:{brand}:{model}")
    if model:
        keys.add(f"model:{model}")
    if category:
        keys.add(f"category:{category}")
    if brand and category:
        keys.add(f"brand_cat:{brand}:{category}")

    # Token-based keys (work without LLM extraction)
    if brand and significant:
        keys.add(f"brand_sig:{brand}:{significant[0]}")
    if len(significant) >= 2:
        keys.add(f"token_pair:{significant[0]}:{significant[1]}")
    elif significant:
        keys.add(f"token:{significant[0]}")

    # Short fingerprint from first few significant tokens
    fingerprint = "".join(sorted(tok[:3] for tok in significant[:4]))
    if fingerprint:
        keys.add(f"fp:{fingerprint}")

    # Character n-gram fingerprint for fuzzy blocking
    alpha_only = re.sub(r"[^a-z0-9]", "", normalized_name)
    if len(alpha_only) >= 6:
        keys.add(f"ngram:{alpha_only[:6]}")
    if len(alpha_only) >= 8:
        keys.add(f"ngram:{alpha_only[:8]}")

    # Number-based blocking — numbers are script-independent and bridge
    # languages (e.g. "Galaxy S23" and "גלקסי 23" both contain "23").
    numbers = sorted(set(re.findall(r"\d+", normalized_name)))
    for num in numbers:
        keys.add(f"num:{num}")
    if len(numbers) >= 2:
        keys.add(f"nums:{numbers[0]}:{numbers[1]}")

    return keys


# ---------------------------------------------------------------------------
# Storage extraction from specs or normalized text
# ---------------------------------------------------------------------------

def _extract_storage_from_text(text: str) -> Optional[int]:
    """Try to pull a storage value from normalised text as a fallback."""
    match = re.search(r"\b(\d+)gb\b", text)
    if match:
        val = int(match.group(1))
        if val in {16, 32, 64, 128, 256, 512, 1024, 2048}:
            return val
    return None


def _storage_from_specs(specs: Dict[str, str]) -> Optional[int]:
    """Pull storage_gb from the specs dict if present."""
    raw = specs.get("storage", "")
    match = re.search(r"(\d+)\s*gb", raw)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*tb", raw)
    if match:
        return int(match.group(1)) * 1024
    return None


# ---------------------------------------------------------------------------
# Row -> ProductRecord
# ---------------------------------------------------------------------------

def row_to_record(
    row: pd.Series,
    index: int,
    config: MatcherConfig,
    llm_fields: Optional[Dict[str, Any]] = None,
) -> ProductRecord:
    """Convert a single DataFrame row into an enriched ProductRecord.

    If *llm_fields* is provided (from LLMExtractor), those are used for brand,
    model, variant, specs, and category.  Otherwise those fields are None and
    the matcher relies on fuzzy + semantic similarity.
    """
    raw_name = str(row[config.name_col])
    normalized = normalize_name(raw_name)
    price = float(row[config.price_col])

    seller = (
        str(row[config.seller_col])
        if config.seller_col and pd.notna(row.get(config.seller_col))
        else None
    )
    product_id = (
        str(row[config.id_col])
        if config.id_col and pd.notna(row.get(config.id_col))
        else None
    )
    currency = (
        str(row[config.currency_col])
        if config.currency_col and pd.notna(row.get(config.currency_col))
        else None
    )

    trade_id = get_trade_id_from_row(row, config)
    mpn = get_optional_identifier(row, config.mpn_col, digits_only=False)
    sku = get_optional_identifier(row, config.sku_col, digits_only=False)

    if llm_fields:
        brand = llm_fields.get("brand")
        model = llm_fields.get("model")
        variant = llm_fields.get("variant")
        specs = llm_fields.get("specs") or {}
        category = llm_fields.get("category")
        color = specs.get("color")
        storage_gb = _storage_from_specs(specs) or _extract_storage_from_text(normalized)
    else:
        brand = None
        model = None
        variant = None
        specs = {}
        category = None
        color = None
        storage_gb = _extract_storage_from_text(normalized)

    tokens = get_tokens(normalized)
    block_keys = build_block_keys(
        normalized, brand, model, tokens,
        trade_id=trade_id, mpn=mpn, sku=sku, seller=seller,
        category=category,
    )

    return ProductRecord(
        index=index,
        raw_name=raw_name,
        normalized_name=normalized,
        price=price,
        seller=seller,
        product_id=product_id,
        currency=currency,
        brand=brand,
        model=model,
        variant=variant,
        storage_gb=storage_gb,
        color=color,
        category=category,
        specs=specs,
        trade_id=trade_id,
        mpn=mpn,
        sku=sku,
        tokens=tokens,
        block_keys=block_keys,
    )
