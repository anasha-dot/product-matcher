"""
matcher.py
----------
Core matching engine (general-purpose, no domain-specific constants).

Contains:
- UnionFind             – disjoint-set structure for clustering
- ProductMatcher        – orchestrates blocking, scoring, and clustering
  - Hard-conflict detection (brand / model / storage / variant / specs)
  - Weighted feature scoring (fuzzy, token Jaccard, semantic, specs, IDs …)
  - Pair decision logic (match / review / reject)
  - Candidate-pair generation via blocking keys
  - Union-Find clustering and canonical-name selection
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz

from .config import MatcherConfig
from .embeddings import (
    LocalEmbeddingSemanticSimilarity,
    SemanticSimilarity,
)
from .llm_extractor import LLMExtractor, LLMExtractorConfig
from .llm_resolver import LLMResolver, LLMResolverConfig
from .models import PairDecision, ProductRecord
from .normalize import row_to_record

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Union-Find (disjoint set)
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def specs_overlap(a: Dict[str, str], b: Dict[str, str]) -> float:
    """Score how well two specs dicts agree. 1.0 = perfect, 0.0 = no overlap."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common_keys = set(a) & set(b)
    if not common_keys:
        return 0.5
    matches = sum(1 for k in common_keys if a[k] == b[k])
    return matches / len(common_keys)


# ---------------------------------------------------------------------------
# Main matcher
# ---------------------------------------------------------------------------

class ProductMatcher:
    """End-to-end product deduplication engine (general-purpose)."""

    def __init__(self, config: Optional[MatcherConfig] = None) -> None:
        self.config = config or MatcherConfig()
        self.semantic: Optional[SemanticSimilarity] = None
        self._llm_extractor: Optional[LLMExtractor] = None

    # ---- LLM extractor initialization ----

    def _build_llm_extractor(self) -> Optional[LLMExtractor]:
        cfg = self.config
        if not cfg.llm_extract:
            return None
        try:
            return LLMExtractor(LLMExtractorConfig(
                enabled=True,
                api_key=cfg.llm_api_key,
                model=cfg.llm_extract_model,
                cache_path=cfg.cache_path,
            ))
        except Exception as exc:
            logger.warning("Could not initialise LLM extractor: %s", exc)
            return None

    # ---- semantic backend initialization ----

    def _build_semantic_backend(self, texts: Sequence[str]) -> None:
        if not self.config.use_semantic:
            self.semantic = None
            return

        self.semantic = LocalEmbeddingSemanticSimilarity(
            model_name=self.config.local_model_name,
            cache_path=self.config.cache_path,
        )
        self.semantic.fit(texts)

    # ---- record loading ----

    def load_records(self, frame: pd.DataFrame) -> List[ProductRecord]:
        required = {self.config.name_col, self.config.price_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        frame = frame.reset_index(drop=True)

        # LLM extraction pass (if enabled)
        llm_fields_list = None
        self._llm_extractor = self._build_llm_extractor()
        if self._llm_extractor is not None:
            raw_names = [str(row[self.config.name_col]) for _, row in frame.iterrows()]
            llm_fields_list = self._llm_extractor.extract_batch(raw_names)
            logger.info("LLM extraction complete for %d products.", len(raw_names))

        records = []
        for idx, row in frame.iterrows():
            llm_fields = llm_fields_list[idx] if llm_fields_list else None
            records.append(row_to_record(row, idx, self.config, llm_fields=llm_fields))

        self._build_semantic_backend([r.normalized_name for r in records])
        return records

    # ---- hard conflicts ----

    def _hard_conflict(self, a: ProductRecord, b: ProductRecord) -> Optional[List[str]]:
        reasons: List[str] = []

        if a.trade_id and b.trade_id and a.trade_id != b.trade_id:
            reasons.append("trade_id_conflict")
        if a.brand and b.brand and a.brand != b.brand:
            reasons.append("brand_conflict")
        if (a.brand and b.brand and a.brand == b.brand
                and a.mpn and b.mpn and a.mpn != b.mpn):
            reasons.append("mpn_conflict")
        if (a.seller and b.seller and a.seller == b.seller
                and a.sku and b.sku and a.sku != b.sku
                and self.config.match_mode == "exact"):
            reasons.append("seller_sku_conflict")

        # Category conflict: different product types are never the same
        if a.category and b.category and a.category != b.category:
            reasons.append("category_conflict")

        # Numeric token conflict: if both names contain numbers and the
        # number sets differ, the products likely differ in model version,
        # capacity, or generation (e.g. XM5 vs XM4, 128GB vs 256GB).
        nums_a = set(re.findall(r"\d+", a.normalized_name))
        nums_b = set(re.findall(r"\d+", b.normalized_name))
        if nums_a and nums_b and nums_a != nums_b:
            reasons.append("numeric_token_conflict")

        if self.config.match_mode == "exact":
            if a.model and b.model and a.model != b.model:
                reasons.append("model_conflict")
            if (self.config.strict_variant_for_exact
                    and a.variant and b.variant and a.variant != b.variant):
                reasons.append("variant_conflict")
            if (self.config.strict_storage_for_exact
                    and a.storage_gb and b.storage_gb
                    and a.storage_gb != b.storage_gb):
                reasons.append("storage_conflict")
            if a.specs and b.specs:
                for key in set(a.specs) & set(b.specs):
                    if a.specs[key] != b.specs[key]:
                        reasons.append(f"specs_conflict:{key}")
        else:
            if a.brand == b.brand and a.model and b.model and a.model != b.model:
                a_letters = re.sub(r"\d", "", a.model)
                b_letters = re.sub(r"\d", "", b.model)
                a_digits = re.sub(r"\D", "", a.model)
                b_digits = re.sub(r"\D", "", b.model)
                if a_letters == b_letters and a_digits and b_digits and a_digits != b_digits:
                    reasons.append("model_conflict")

        return reasons if reasons else None

    # ---- feature scoring ----

    def _feature_scores(
        self, a: ProductRecord, b: ProductRecord,
    ) -> Tuple[float, Dict[str, float], List[str]]:
        reasons: List[str] = []
        cfg = self.config

        # --- compute individual feature scores ---
        exact_norm = 1.0 if a.normalized_name == b.normalized_name else 0.0
        fuzzy_score = fuzz.token_sort_ratio(a.normalized_name, b.normalized_name) / 100.0
        partial_score = fuzz.partial_ratio(a.normalized_name, b.normalized_name) / 100.0
        fuzzy_score = max(fuzzy_score, 0.6 * fuzzy_score + 0.4 * partial_score)
        token_score = jaccard(a.tokens, b.tokens)

        brand_known = bool(a.brand and b.brand)
        brand_score = 1.0 if brand_known and a.brand == b.brand else 0.0
        if not brand_known:
            reasons.append("brand_unknown")

        model_known = bool(a.model and b.model)
        if model_known:
            model_score = 1.0 if a.model == b.model else 0.0
            if a.model == b.model:
                reasons.append("same_model")
        else:
            model_score = 0.0
            reasons.append("model_missing")

        variant_known = a.variant is not None or b.variant is not None
        variant_score = (
            1.0 if (a.variant is None and b.variant is None) or a.variant == b.variant
            else 0.0
        )
        storage_known = a.storage_gb is not None or b.storage_gb is not None
        storage_score = (
            1.0 if (a.storage_gb is None and b.storage_gb is None) or a.storage_gb == b.storage_gb
            else 0.0
        )

        specs_known = bool(a.specs or b.specs)
        specs_score_val = specs_overlap(a.specs, b.specs)

        semantic_score = self.semantic.similarity(a.index, b.index) if self.semantic else 0.0
        if semantic_score >= 0.90:
            reasons.append("very_high_semantic_similarity")

        trade_id_known = bool(a.trade_id and b.trade_id)
        trade_id_score = 1.0 if trade_id_known and a.trade_id == b.trade_id else 0.0
        mpn_known = bool(a.mpn and b.mpn)
        mpn_score = 1.0 if mpn_known and a.mpn == b.mpn else 0.0
        sku_known = bool(a.sku and b.sku and a.seller and b.seller)
        sku_score = (
            1.0 if sku_known and a.sku == b.sku and a.seller == b.seller
            else 0.0
        )

        if trade_id_score:
            reasons.append("same_trade_id")
        if mpn_score:
            reasons.append("same_mpn")
        if sku_score:
            reasons.append("same_seller_sku")

        features = {
            "exact_norm": exact_norm,
            "fuzzy": fuzzy_score,
            "token": token_score,
            "brand": brand_score,
            "model": model_score,
            "variant": variant_score,
            "storage": storage_score,
            "specs": specs_score_val,
            "semantic": semantic_score,
            "trade_id": trade_id_score,
            "mpn": mpn_score,
            "sku": sku_score,
        }

        # --- adaptive weight normalisation ---
        # Always-active features (text similarity signals are always computable).
        weighted_pairs: List[Tuple[float, float]] = [
            (cfg.exact_name_weight, features["exact_norm"]),
            (cfg.fuzzy_weight,      features["fuzzy"]),
            (cfg.token_weight,      features["token"]),
            (cfg.semantic_weight,   features["semantic"]),
        ]
        # Structured features only participate when at least one side has data;
        # otherwise their weight is redistributed to the active signals above,
        # preventing cross-language / low-data pairs from being penalised.
        if brand_known:
            weighted_pairs.append((cfg.brand_weight, features["brand"]))
        if model_known:
            weighted_pairs.append((cfg.model_weight, features["model"]))
        if variant_known:
            weighted_pairs.append((cfg.variant_weight, features["variant"]))
        if storage_known:
            weighted_pairs.append((cfg.storage_weight, features["storage"]))
        if specs_known:
            weighted_pairs.append((cfg.specs_weight, features["specs"]))
        if trade_id_known:
            weighted_pairs.append((cfg.trade_id_weight, features["trade_id"]))
        if mpn_known:
            weighted_pairs.append((cfg.mpn_weight, features["mpn"]))
        if sku_known:
            weighted_pairs.append((cfg.sku_weight, features["sku"]))

        total_weight = sum(w for w, _ in weighted_pairs)
        raw_score = sum(w * s for w, s in weighted_pairs)
        final_score = (raw_score / total_weight) if total_weight > 0 else 0.0

        return min(1.0, final_score), features, reasons

    # ---- pair decision ----

    def decide_pair(self, a: ProductRecord, b: ProductRecord) -> PairDecision:
        if a.trade_id and b.trade_id and a.trade_id == b.trade_id:
            return PairDecision(
                same=True, score=1.0, status="match",
                reasons=["same_trade_id"], source="trade_id",
                features={"trade_id": 1.0},
            )
        if (a.mpn and b.mpn and a.mpn == b.mpn
                and ((not a.brand or not b.brand) or a.brand == b.brand)):
            return PairDecision(
                same=True, score=0.97, status="match",
                reasons=["same_mpn"], source="mpn",
                features={"mpn": 1.0},
            )
        if (a.sku and b.sku and a.sku == b.sku
                and a.seller and b.seller and a.seller == b.seller):
            return PairDecision(
                same=True, score=0.95, status="match",
                reasons=["same_seller_sku"], source="sku",
                features={"sku": 1.0},
            )

        conflict = self._hard_conflict(a, b)
        if conflict:
            return PairDecision(same=False, score=0.0, status="reject", reasons=conflict)

        final_score, features, reasons = self._feature_scores(a, b)

        # High-confidence semantic shortcut: when the multilingual embedding
        # model is very confident the names are equivalent (e.g. cross-language
        # pairs) and there are no hard conflicts, promote to match — but only
        # if the numeric tokens agree (different numbers often mean different
        # product versions, e.g. XM5 vs XM4).
        sem = features.get("semantic", 0.0)
        if sem >= self.config.semantic_match_threshold:
            nums_a = set(re.findall(r"\d+", a.normalized_name))
            nums_b = set(re.findall(r"\d+", b.normalized_name))
            if nums_a == nums_b:
                final_score = max(final_score, self.config.match_threshold)

        if a.model and b.model and a.model == b.model and final_score >= self.config.review_threshold:
            final_score = max(final_score, self.config.match_threshold)

        if final_score >= self.config.match_threshold:
            return PairDecision(
                same=True, score=final_score, status="match",
                reasons=reasons, features=features,
            )
        if final_score >= self.config.review_threshold:
            return PairDecision(
                same=False, score=final_score, status="review",
                reasons=reasons, features=features,
            )
        return PairDecision(
            same=False, score=final_score, status="reject",
            reasons=reasons, features=features,
        )

    # ---- blocking ----

    def candidate_pairs(self, records: Sequence[ProductRecord]) -> Set[Tuple[int, int]]:
        buckets: Dict[str, List[int]] = defaultdict(list)
        for rec in records:
            for key in rec.block_keys:
                buckets[key].append(rec.index)

        limit = self.config.max_bucket_size
        pairs: Set[Tuple[int, int]] = set()
        for idxs in buckets.values():
            if len(idxs) > limit:
                idxs = idxs[:limit]
            for i, j in combinations(sorted(set(idxs)), 2):
                pairs.add((i, j))
        return pairs

    # ---- LLM resolver ----

    def _build_llm_resolver(self) -> Optional[LLMResolver]:
        cfg = self.config
        if not cfg.llm_resolve:
            return None
        try:
            return LLMResolver(LLMResolverConfig(
                enabled=True,
                api_key=cfg.llm_api_key,
                model=cfg.llm_model,
                temperature=cfg.llm_temperature,
                max_pairs=cfg.llm_max_pairs,
            ))
        except Exception as exc:
            logger.warning("Could not initialise LLM resolver: %s", exc)
            return None

    # ---- clustering ----

    def cluster(
        self, records: Sequence[ProductRecord],
    ) -> Tuple[List[List[ProductRecord]], List[Dict[str, object]]]:
        pairs = self.candidate_pairs(records)
        uf = UnionFind(len(records))
        reviews: List[Dict[str, object]] = []

        by_index = {r.index: r for r in records}
        for i, j in sorted(pairs):
            decision = self.decide_pair(by_index[i], by_index[j])
            if decision.status == "match":
                uf.union(i, j)
            elif decision.status == "review":
                reviews.append({
                    "left_index": i,
                    "right_index": j,
                    "left_name": by_index[i].raw_name,
                    "right_name": by_index[j].raw_name,
                    "normalized_left": by_index[i].normalized_name,
                    "normalized_right": by_index[j].normalized_name,
                    "score": round(decision.score, 4),
                    "reasons": decision.reasons,
                })

        # --- LLM fallback: resolve uncertain pairs ---
        if reviews:
            resolver = self._build_llm_resolver()
            if resolver is not None:
                llm_matches, reviews = resolver.resolve_batch(reviews)
                for pair in llm_matches:
                    uf.union(pair["left_index"], pair["right_index"])
                logger.info(
                    "LLM merged %d additional pairs.", len(llm_matches),
                )

        clusters: Dict[int, List[ProductRecord]] = defaultdict(list)
        for rec in records:
            clusters[uf.find(rec.index)].append(rec)

        cluster_list = [
            sorted(group, key=lambda r: (r.price, len(r.raw_name), r.raw_name))
            for group in clusters.values()
        ]
        cluster_list.sort(key=lambda group: min(r.index for r in group))
        return cluster_list, reviews

    # ---- output formatting ----

    def build_output(
        self, clusters: Sequence[Sequence[ProductRecord]],
    ) -> List[Dict[str, object]]:
        output = []
        for cluster_id, group in enumerate(clusters, start=1):
            lowest = min(group, key=lambda r: r.price)
            canonical = self.choose_canonical_name(group)
            output.append({
                "cluster_id": cluster_id,
                "canonical_name": canonical,
                "lowest_price": lowest.price,
                "currency": lowest.currency,
                "offer_count": len(group),
                "offers": [
                    {
                        "index": r.index,
                        "product_id": r.product_id,
                        "name": r.raw_name,
                        "normalized_name": r.normalized_name,
                        "seller": r.seller,
                        "price": r.price,
                        "currency": r.currency,
                        "brand": r.brand,
                        "model": r.model,
                        "variant": r.variant,
                        "storage_gb": r.storage_gb,
                        "category": r.category,
                        "specs": r.specs,
                        "trade_id": r.trade_id,
                        "mpn": r.mpn,
                        "sku": r.sku,
                    }
                    for r in sorted(group, key=lambda r: (r.price, r.raw_name))
                ],
            })
        return output

    @staticmethod
    def choose_canonical_name(group: Sequence[ProductRecord]) -> str:
        """Pick the richest, most common normalized form as the display name."""
        norm_counts = Counter(r.normalized_name for r in group)
        best_norm = norm_counts.most_common(1)[0][0]
        candidates = [r for r in group if r.normalized_name == best_norm]
        candidates.sort(
            key=lambda r: (-(len(set(r.raw_name.split()))), -len(r.raw_name), r.price),
        )
        return candidates[0].raw_name

    # ---- public API ----

    def run(
        self, frame: pd.DataFrame,
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        records = self.load_records(frame)
        clusters, reviews = self.cluster(records)
        return self.build_output(clusters), reviews

    def close(self) -> None:
        if self.semantic and hasattr(self.semantic, "close"):
            self.semantic.close()
        if self._llm_extractor:
            self._llm_extractor.close()
