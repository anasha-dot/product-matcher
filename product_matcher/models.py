"""
models.py
---------
Data-classes that flow through the matching pipeline.

Contains:
- ProductRecord  – one row from the input, enriched with extracted features
- PairDecision   – the verdict for a pair of products (match / review / reject)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ProductRecord:
    """A single product listing enriched with normalized fields."""

    index: int
    raw_name: str
    normalized_name: str
    price: float
    seller: Optional[str] = None
    product_id: Optional[str] = None
    currency: Optional[str] = None

    brand: Optional[str] = None
    model: Optional[str] = None
    variant: Optional[str] = None
    storage_gb: Optional[int] = None
    color: Optional[str] = None
    category: Optional[str] = None
    specs: Dict[str, str] = field(default_factory=dict)

    trade_id: Optional[str] = None
    mpn: Optional[str] = None
    sku: Optional[str] = None
    tokens: Set[str] = field(default_factory=set)
    block_keys: Set[str] = field(default_factory=set)


@dataclass
class PairDecision:
    """Result of comparing two ProductRecords."""

    same: bool
    score: float
    status: str                                     # "match" | "review" | "reject"
    reasons: List[str]
    source: Optional[str] = None                    # e.g. "trade_id", "mpn", "sku"
    features: Optional[Dict[str, float]] = None     # per-signal breakdown
