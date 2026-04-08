"""
product_matcher
===============
General-purpose multilingual product-listing deduplicator.

Given a list of product names (any category, any language, with typos and
inconsistent formatting), this package groups duplicate listings and surfaces
the lowest price for each unique product.

Quick start (library)::

    import pandas as pd
    from product_matcher import ProductMatcher, MatcherConfig

    df = pd.read_csv("products.csv")
    matcher = ProductMatcher(MatcherConfig(name_col="name", price_col="price"))
    clusters, reviews = matcher.run(df)

Quick start (CLI)::

    python -m product_matcher --input products.csv --output results.json
"""
from .config import MatcherConfig
from .llm_resolver import LLMResolver, LLMResolverConfig
from .matcher import ProductMatcher
from .models import PairDecision, ProductRecord

__all__ = [
    "MatcherConfig",
    "LLMResolver",
    "LLMResolverConfig",
    "ProductMatcher",
    "ProductRecord",
    "PairDecision",
]
