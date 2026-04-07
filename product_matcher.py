from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


# -----------------------------
# Configuration
# -----------------------------

BRAND_ALIASES: Dict[str, str] = {
    # English
    "samsung": "samsung",
    "galaxy": "galaxy",
    "apple": "apple",
    "iphone": "apple iphone",
    "xiaomi": "xiaomi",
    "redmi": "xiaomi redmi",
    "poco": "xiaomi poco",
    "oneplus": "oneplus",
    "google": "google",
    "pixel": "google pixel",
    "huawei": "huawei",
    "honor": "honor",
    "oppo": "oppo",
    "realme": "realme",
    "motorola": "motorola",
    "moto": "motorola",
    "nokia": "nokia",

    # Hebrew
    "סמסונג": "samsung",
    "גלקסי": "galaxy",
    "אפל": "apple",
    "אייפון": "apple iphone",
    "אייפונ": "apple iphone",
    "שיאומי": "xiaomi",
    "רדמי": "xiaomi redmi",
    "פיקסל": "google pixel",
    "גוגל": "google",
    "וואווי": "huawei",
    "הונור": "honor",
    "אופו": "oppo",
    "מוטורולה": "motorola",
    "נוקיה": "nokia",
    "פרו": "pro",
    "פלוס": "plus",
    "אולטרה": "ultra",
    "מקס": "max",

    # Arabic
    "سامسونج": "samsung",
    "جالكسي": "galaxy",
    "آبل": "apple",
    "ابل": "apple",
    "ايفون": "apple iphone",
    "آيفون": "apple iphone",
    "شاومي": "xiaomi",
    "شياومي": "xiaomi",
    "ريدمي": "xiaomi redmi",
    "بيكسل": "google pixel",
    "جوجل": "google",
    "هواوي": "huawei",
    "هونر": "honor",
    "اوبو": "oppo",
    "موتورولا": "motorola",
    "نوكيا": "nokia",
    "برو": "pro",
    "بلس": "plus",
    "الترا": "ultra",
    "ماكس": "max",
}

TYPO_ALIASES: Dict[str, str] = {
    "smsung": "samsung",
    "samung": "samsung",
    "smasung": "samsung",
    "galaxi": "galaxy",
    "galxy": "galaxy",
    "galx": "galaxy",
    "iphon": "iphone",
    "iphne": "iphone",
    "xiaome": "xiaomi",
    "redme": "redmi",
}

VARIANT_WORDS = {
    "pro",
    "plus",
    "ultra",
    "max",
    "mini",
    "fe",
    "lite",
    "note",
    "prime",
    "air",
}

STOPWORDS = {
    "new",
    "smartphone",
    "phone",
    "mobile",
    "cellphone",
    "series",
    "edition",
    "דור",
    "טלפון",
    "الهاتف",
    "هاتف",
}

DEFAULT_LOCAL_MODEL = os.getenv("PRODUCT_MATCHER_LOCAL_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")


@dataclass
class MatcherConfig:
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

    match_mode: str = "exact"  # exact | family
    use_semantic: bool = True
    semantic_backend: str = "local"  # local | tfidf
    local_model_name: str = DEFAULT_LOCAL_MODEL
    semantic_weight: float = 0.20

    match_threshold: float = 0.82
    review_threshold: float = 0.68

    exact_name_weight: float = 0.10
    fuzzy_weight: float = 0.15
    token_weight: float = 0.10
    brand_weight: float = 0.15
    model_weight: float = 0.25
    variant_weight: float = 0.10
    storage_weight: float = 0.10
    semantic_extra_weight: float = 0.05
    trade_id_weight: float = 0.35
    mpn_weight: float = 0.18
    sku_weight: float = 0.12

    # exact mode: different storage / variant means different product
    strict_storage_for_exact: bool = True
    strict_variant_for_exact: bool = True

    cache_path: str = ".embedding_cache.sqlite3"


@dataclass
class ProductRecord:
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
    trade_id: Optional[str] = None
    mpn: Optional[str] = None
    sku: Optional[str] = None
    tokens: Set[str] = field(default_factory=set)
    block_keys: Set[str] = field(default_factory=set)


@dataclass
class PairDecision:
    same: bool
    score: float
    status: str  # match | review | reject
    reasons: List[str]


# -----------------------------
# Normalization / extraction
# -----------------------------


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


ARABIC_CHAR_NORMALIZATION = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
})


HEBREW_FINAL_FORMS = str.maketrans({
    "ך": "כ",
    "ם": "מ",
    "ן": "נ",
    "ף": "פ",
    "ץ": "צ",
})


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
    # Storage
    text = re.sub(r"(\d+)\s*(gb|ג[י׳']?ב|جيجا|جيجابايت)", r"\1gb", text)
    text = re.sub(r"(\d+)\s*(tb|ט[י׳']?ב|تيرا|تيرابايت)", lambda m: f"{int(m.group(1)) * 1024}gb", text)
    return text


def normalize_model_patterns(text: str) -> str:
    # s 23 -> s23, a 54 -> a54
    text = re.sub(r"\b([a-z])\s+(\d{1,3})\b", r"\1\2", text)
    # galaxy 23 -> galaxy s23 (smartphone-specific heuristic)
    text = re.sub(r"\bgalaxy\s+(\d{1,3})\b", r"galaxy s\1", text)
    text = re.sub(r"\bgalaxy(\d{1,3})\b", r"galaxy s\1", text)
    text = re.sub(r"\bgalx(\d{1,3})\b", r"galaxy s\1", text)
    # iphone 15pro -> iphone 15 pro
    text = re.sub(r"\b(\d{1,2})(pro|max|plus|mini)\b", r"\1 \2", text)
    return text


def apply_aliases(text: str) -> str:
    words = []
    for word in text.split():
        base = TYPO_ALIASES.get(word, word)
        mapped = BRAND_ALIASES.get(base, base)
        words.extend(mapped.split())
    return " ".join(words)


def normalize_name(name: str) -> str:
    text = (name or "").lower().strip()
    text = strip_diacritics(text)
    text = basic_script_normalize(text)
    text = normalize_separators(text)
    text = normalize_units(text)
    text = apply_aliases(text)
    text = normalize_model_patterns(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def get_optional_identifier(row: pd.Series, col_name: Optional[str], digits_only: bool = False) -> Optional[str]:
    if not col_name or col_name not in row or pd.isna(row[col_name]):
        return None
    return normalize_identifier(row[col_name], digits_only=digits_only)


BRAND_CANONICALS = {
    "samsung",
    "apple",
    "xiaomi",
    "oneplus",
    "google",
    "huawei",
    "honor",
    "oppo",
    "realme",
    "motorola",
    "nokia",
}

COLOR_WORDS = {
    "black", "white", "blue", "green", "red", "pink", "purple", "gold", "silver", "gray", "grey",
    "שחור", "לבן", "כחול", "ירוק", "אדום", "ורוד", "סגול", "זהב", "כסוף", "אפור",
    "اسود", "ابيض", "ازرق", "اخضر", "احمر", "وردي", "بنفسجي", "ذهبي", "فضي", "رمادي",
}


def extract_brand(text: str) -> Optional[str]:
    tokens = text.split()
    for i, token in enumerate(tokens):
        if token in BRAND_CANONICALS:
            return token
        if token == "iphone" or (token == "apple" and i + 1 < len(tokens) and tokens[i + 1] == "iphone"):
            return "apple"
        if token == "pixel" or (token == "google" and i + 1 < len(tokens) and tokens[i + 1] == "pixel"):
            return "google"
        if token == "redmi" or token == "poco":
            return "xiaomi"
    return None


def extract_model(text: str, brand: Optional[str]) -> Optional[str]:
    # Prefer phone-like model patterns: s23, a54, 15, note 13, pixel 8, oneplus 12
    text = text.replace("apple iphone", "iphone")
    patterns = [
        r"\b(s\d{1,3})\b",
        r"\b(a\d{1,3})\b",
        r"\b(note\s*\d{1,2})\b",
        r"\b(pixel\s*\d{1,2})\b",
        r"\b(oneplus\s*\d{1,2})\b",
        r"\b(iphone\s*\d{1,2})\b",
        r"\b(redmi\s*note\s*\d{1,2})\b",
        r"\b(\d{1,2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = re.sub(r"\s+", " ", match.group(1).strip())
            if pattern == r"\b(\d{1,2})\b" and brand not in {"apple", "google", "oneplus"}:
                continue
            return value
    return None


def extract_variant(text: str) -> Optional[str]:
    tokens = text.split()
    found = [token for token in tokens if token in VARIANT_WORDS]
    if not found:
        return None
    # if both note and pro exist, keep quality variant over series word
    for preferred in ["ultra", "max", "pro", "plus", "mini", "lite", "fe", "prime", "air", "note"]:
        if preferred in found:
            return preferred
    return found[0]


def extract_storage_gb(text: str) -> Optional[int]:
    match = re.search(r"\b(64|128|256|512|1024|2048)gb\b", text)
    if match:
        return int(match.group(1))
    return None


def extract_color(text: str) -> Optional[str]:
    for token in text.split():
        if token in COLOR_WORDS:
            return token
    return None


def get_tokens(text: str) -> Set[str]:
    return {tok for tok in text.split() if tok and tok not in STOPWORDS}


def build_block_keys(
    normalized_name: str,
    brand: Optional[str],
    model: Optional[str],
    tokens: Set[str],
    trade_id: Optional[str] = None,
    mpn: Optional[str] = None,
    sku: Optional[str] = None,
    seller: Optional[str] = None,
) -> Set[str]:
    keys: Set[str] = set()
    significant = sorted([tok for tok in tokens if tok not in VARIANT_WORDS])
    if trade_id:
        keys.add(f"trade_id:{trade_id}")
    if mpn:
        keys.add(f"mpn:{mpn}")
    if sku and seller:
        keys.add(f"seller_sku:{seller.lower()}:{sku}")
    if brand:
        keys.add(f"brand:{brand}")
    if brand and model:
        keys.add(f"brand_model:{brand}:{model}")
    if model:
        keys.add(f"model:{model}")
        keys.add(f"model_prefix:{model[:2]}")
    if brand and significant:
        keys.add(f"brand_sig:{brand}:{significant[0]}")
    if len(significant) >= 2:
        keys.add(f"token_pair:{significant[0]}:{significant[1]}")
    elif significant:
        keys.add(f"token:{significant[0]}")
    # Add a generic short fingerprint so obvious typo variants can still meet.
    fingerprint = "".join(sorted(tok[:3] for tok in significant[:3]))
    if fingerprint:
        keys.add(f"fp:{fingerprint}")
    return keys


def row_to_record(row: pd.Series, index: int, config: MatcherConfig) -> ProductRecord:
    raw_name = str(row[config.name_col])
    normalized_name = normalize_name(raw_name)
    price = float(row[config.price_col])
    seller = str(row[config.seller_col]) if config.seller_col and pd.notna(row.get(config.seller_col)) else None
    product_id = str(row[config.id_col]) if config.id_col and pd.notna(row.get(config.id_col)) else None
    currency = str(row[config.currency_col]) if config.currency_col and pd.notna(row.get(config.currency_col)) else None
    trade_id = get_trade_id_from_row(row, config)
    mpn = get_optional_identifier(row, config.mpn_col, digits_only=False)
    sku = get_optional_identifier(row, config.sku_col, digits_only=False)

    brand = extract_brand(normalized_name)
    model = extract_model(normalized_name, brand)
    variant = extract_variant(normalized_name)
    storage_gb = extract_storage_gb(normalized_name)
    color = extract_color(normalized_name)
    tokens = get_tokens(normalized_name)
    block_keys = build_block_keys(
        normalized_name,
        brand,
        model,
        tokens,
        trade_id=trade_id,
        mpn=mpn,
        sku=sku,
        seller=seller,
    )

    return ProductRecord(
        index=index,
        raw_name=raw_name,
        normalized_name=normalized_name,
        price=price,
        seller=seller,
        product_id=product_id,
        currency=currency,
        brand=brand,
        model=model,
        variant=variant,
        storage_gb=storage_gb,
        color=color,
        trade_id=trade_id,
        mpn=mpn,
        sku=sku,
        tokens=tokens,
        block_keys=block_keys,
    )


# -----------------------------
# Embedding backends
# -----------------------------


class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[np.ndarray]:
        row = self.conn.execute("SELECT value FROM embeddings WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        return np.array(json.loads(row[0]), dtype=np.float32)

    def set(self, key: str, value: np.ndarray) -> None:
        data = json.dumps(value.astype(float).tolist(), ensure_ascii=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (key, value) VALUES (?, ?)",
            (key, data),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class SemanticSimilarity:
    def fit(self, texts: Sequence[str]) -> None:
        raise NotImplementedError

    def similarity(self, a_index: int, b_index: int) -> float:
        raise NotImplementedError


class TfidfSemanticSimilarity(SemanticSimilarity):
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.matrix = None

    def fit(self, texts: Sequence[str]) -> None:
        self.matrix = self.vectorizer.fit_transform(texts)

    def similarity(self, a_index: int, b_index: int) -> float:
        if self.matrix is None:
            return 0.0
        value = cosine_similarity(self.matrix[a_index], self.matrix[b_index])[0, 0]
        return float(max(0.0, min(1.0, value)))


class LocalEmbeddingSemanticSimilarity(SemanticSimilarity):
    def __init__(self, model_name: str, cache_path: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it or switch semantic_backend to 'tfidf'."
            )
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cache = EmbeddingCache(cache_path)
        self.embeddings: List[np.ndarray] = []

    def fit(self, texts: Sequence[str]) -> None:
        self.embeddings = []
        missing_texts: List[str] = []
        missing_positions: List[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(f"{self.model_name}::{text}")
            if cached is not None:
                self.embeddings.append(cached)
            else:
                self.embeddings.append(np.array([], dtype=np.float32))
                missing_texts.append(text)
                missing_positions.append(i)

        if missing_texts:
            new_embeddings = self.model.encode(missing_texts, normalize_embeddings=True)
            for pos, emb in zip(missing_positions, new_embeddings):
                emb_arr = np.asarray(emb, dtype=np.float32)
                self.embeddings[pos] = emb_arr
                self.cache.set(f"{self.model_name}::{texts[pos]}", emb_arr)

    def similarity(self, a_index: int, b_index: int) -> float:
        a = self.embeddings[a_index]
        b = self.embeddings[b_index]
        if a.size == 0 or b.size == 0:
            return 0.0
        value = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return max(0.0, min(1.0, value))

    def close(self) -> None:
        self.cache.close()


# -----------------------------
# Matching
# -----------------------------


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ProductMatcher:
    def __init__(self, config: Optional[MatcherConfig] = None) -> None:
        self.config = config or MatcherConfig()
        self.semantic: Optional[SemanticSimilarity] = None

    def _build_semantic_backend(self, texts: Sequence[str]) -> None:
        if not self.config.use_semantic:
            self.semantic = None
            return

        if self.config.semantic_backend == "local":
            try:
                self.semantic = LocalEmbeddingSemanticSimilarity(
                    model_name=self.config.local_model_name,
                    cache_path=self.config.cache_path,
                )
                self.semantic.fit(texts)
                return
            except Exception as exc:
                print(
                    f"[warning] Could not initialize local semantic model ({exc}). Falling back to TF-IDF.",
                    file=sys.stderr,
                )

        self.semantic = TfidfSemanticSimilarity()
        self.semantic.fit(texts)

    def load_records(self, frame: pd.DataFrame) -> List[ProductRecord]:
        required = {self.config.name_col, self.config.price_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        records = [row_to_record(row, idx, self.config) for idx, row in frame.reset_index(drop=True).iterrows()]
        self._build_semantic_backend([r.normalized_name for r in records])
        return records

    def _hard_conflict(self, a: ProductRecord, b: ProductRecord) -> Optional[List[str]]:
        reasons: List[str] = []
        if a.trade_id and b.trade_id and a.trade_id != b.trade_id:
            reasons.append("trade_id_conflict")
        if a.brand and b.brand and a.brand != b.brand:
            reasons.append("brand_conflict")
        if a.brand and b.brand and a.brand == b.brand and a.mpn and b.mpn and a.mpn != b.mpn:
            reasons.append("mpn_conflict")
        if a.seller and b.seller and a.seller == b.seller and a.sku and b.sku and a.sku != b.sku and self.config.match_mode == "exact":
            reasons.append("seller_sku_conflict")

        if self.config.match_mode == "exact":
            if a.model and b.model and a.model != b.model:
                reasons.append("model_conflict")
            if self.config.strict_variant_for_exact and a.variant and b.variant and a.variant != b.variant:
                reasons.append("variant_conflict")
            if self.config.strict_storage_for_exact and a.storage_gb and b.storage_gb and a.storage_gb != b.storage_gb:
                reasons.append("storage_conflict")
        else:
            # family mode: only reject large numeric model drift within the same series when both are strong.
            if a.brand == b.brand and a.model and b.model and a.model != b.model:
                a_letters = re.sub(r"\d", "", a.model)
                b_letters = re.sub(r"\d", "", b.model)
                a_digits = re.sub(r"\D", "", a.model)
                b_digits = re.sub(r"\D", "", b.model)
                if a_letters == b_letters and a_digits and b_digits and a_digits != b_digits:
                    reasons.append("model_conflict")

        return reasons if reasons else None

    def _feature_scores(self, a: ProductRecord, b: ProductRecord) -> Tuple[float, Dict[str, float], List[str]]:
        reasons: List[str] = []

        exact_norm = 1.0 if a.normalized_name == b.normalized_name else 0.0
        fuzzy_score = fuzz.token_sort_ratio(a.normalized_name, b.normalized_name) / 100.0
        partial_score = fuzz.partial_ratio(a.normalized_name, b.normalized_name) / 100.0
        fuzzy_score = max(fuzzy_score, 0.6 * fuzzy_score + 0.4 * partial_score)
        token_score = jaccard(a.tokens, b.tokens)

        brand_score = 1.0 if a.brand and b.brand and a.brand == b.brand else 0.0
        if not a.brand or not b.brand:
            reasons.append("brand_unknown")

        if a.model and b.model:
            model_score = 1.0 if a.model == b.model else 0.0
            if a.model == b.model:
                reasons.append("same_model")
        else:
            model_score = 0.0
            reasons.append("model_missing")

        if a.variant is None and b.variant is None:
            variant_score = 1.0
        else:
            variant_score = 1.0 if a.variant == b.variant else 0.0

        if a.storage_gb is None and b.storage_gb is None:
            storage_score = 1.0
        else:
            storage_score = 1.0 if a.storage_gb == b.storage_gb else 0.0

        semantic_score = self.semantic.similarity(a.index, b.index) if self.semantic else 0.0
        if semantic_score >= 0.90:
            reasons.append("very_high_semantic_similarity")

        trade_id_score = 1.0 if a.trade_id and b.trade_id and a.trade_id == b.trade_id else 0.0
        mpn_score = 1.0 if a.mpn and b.mpn and a.mpn == b.mpn else 0.0
        sku_score = 1.0 if a.sku and b.sku and a.sku == b.sku and a.seller and b.seller and a.seller == b.seller else 0.0
        if trade_id_score:
            reasons.append("same_trade_id")
        if mpn_score:
            reasons.append("same_mpn")
        if sku_score:
            reasons.append("same_seller_sku")

        scores = {
            "exact_norm": exact_norm,
            "fuzzy": fuzzy_score,
            "token": token_score,
            "brand": brand_score,
            "model": model_score,
            "variant": variant_score,
            "storage": storage_score,
            "semantic": semantic_score,
            "trade_id": trade_id_score,
            "mpn": mpn_score,
            "sku": sku_score,
        }

        final_score = (
            self.config.exact_name_weight * scores["exact_norm"]
            + self.config.fuzzy_weight * scores["fuzzy"]
            + self.config.token_weight * scores["token"]
            + self.config.brand_weight * scores["brand"]
            + self.config.model_weight * scores["model"]
            + self.config.variant_weight * scores["variant"]
            + self.config.storage_weight * scores["storage"]
            + (self.config.semantic_weight + self.config.semantic_extra_weight) * scores["semantic"]
            + self.config.trade_id_weight * scores["trade_id"]
            + self.config.mpn_weight * scores["mpn"]
            + self.config.sku_weight * scores["sku"]
        )
        return min(1.0, final_score), scores, reasons

    def decide_pair(self, a: ProductRecord, b: ProductRecord) -> PairDecision:
        if a.trade_id and b.trade_id and a.trade_id == b.trade_id:
            return PairDecision(same=True, score=1.0, status="match", reasons=["same_trade_id"], source="trade_id", features={"trade_id": 1.0})
        if a.mpn and b.mpn and a.mpn == b.mpn and ((not a.brand or not b.brand) or a.brand == b.brand):
            return PairDecision(same=True, score=0.97, status="match", reasons=["same_mpn"], source="mpn", features={"mpn": 1.0})
        if a.sku and b.sku and a.sku == b.sku and a.seller and b.seller and a.seller == b.seller:
            return PairDecision(same=True, score=0.95, status="match", reasons=["same_seller_sku"], source="sku", features={"sku": 1.0})

        conflict = self._hard_conflict(a, b)
        if conflict:
            return PairDecision(same=False, score=0.0, status="reject", reasons=conflict)

        final_score, features, reasons = self._feature_scores(a, b)

        if a.model and b.model and a.model == b.model and final_score >= self.config.review_threshold:
            final_score = max(final_score, self.config.match_threshold)

        if final_score >= self.config.match_threshold:
            return PairDecision(same=True, score=final_score, status="match", reasons=reasons, features=features)
        if final_score >= self.config.review_threshold:
            return PairDecision(same=False, score=final_score, status="review", reasons=reasons, features=features)
        return PairDecision(same=False, score=final_score, status="reject", reasons=reasons, features=features)

    def candidate_pairs(self, records: Sequence[ProductRecord]) -> Set[Tuple[int, int]]:
        buckets: Dict[str, List[int]] = defaultdict(list)
        for rec in records:
            for key in rec.block_keys:
                buckets[key].append(rec.index)

        pairs: Set[Tuple[int, int]] = set()
        for _, idxs in buckets.items():
            # Avoid giant buckets blowing up comparisons.
            if len(idxs) > 200:
                idxs = idxs[:200]
            for i, j in combinations(sorted(set(idxs)), 2):
                pairs.add((i, j))
        return pairs

    def cluster(self, records: Sequence[ProductRecord]) -> Tuple[List[List[ProductRecord]], List[Dict[str, object]]]:
        pairs = self.candidate_pairs(records)
        parent = list(range(len(records)))
        rank = [0] * len(records)
        reviews: List[Dict[str, object]] = []

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1

        by_index = {r.index: r for r in records}
        for i, j in sorted(pairs):
            decision = self.decide_pair(by_index[i], by_index[j])
            if decision.status == "match":
                union(i, j)
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

        clusters: Dict[int, List[ProductRecord]] = defaultdict(list)
        for rec in records:
            clusters[find(rec.index)].append(rec)

        cluster_list = [sorted(group, key=lambda r: (r.price, len(r.raw_name), r.raw_name)) for group in clusters.values()]
        cluster_list.sort(key=lambda group: min(r.index for r in group))
        return cluster_list, reviews

    def build_output(self, clusters: Sequence[Sequence[ProductRecord]]) -> List[Dict[str, object]]:
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
        # Pick the richest clean-looking original title with the majority normalized form.
        norm_counts = Counter(r.normalized_name for r in group)
        best_norm = norm_counts.most_common(1)[0][0]
        candidates = [r for r in group if r.normalized_name == best_norm]
        candidates.sort(
            key=lambda r: (
                -(len(set(r.raw_name.split()))),
                -len(r.raw_name),
                r.price,
            )
        )
        return candidates[0].raw_name

    def run(self, frame: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        records = self.load_records(frame)
        clusters, reviews = self.cluster(records)
        return self.build_output(clusters), reviews

    def close(self) -> None:
        if hasattr(self.semantic, "close"):
            self.semantic.close()  # type: ignore[attr-defined]


# -----------------------------
# CLI / helpers
# -----------------------------


def load_input(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_obj)
    if suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError("Supported input formats: .csv, .xlsx, .xls, .json")


def save_json(path: str, data: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


SAMPLE_DATA = [
    {"name": "Samsung S23", "price": 3200, "seller": "A"},
    {"name": "Samsung Galaxy S23", "price": 3150, "seller": "B"},
    {"name": "סמסונג גלקסי 23", "price": 3100, "seller": "C"},
    {"name": "S-smsung Galx23", "price": 3250, "seller": "D"},
    {"name": "iPhone 15 Pro 128GB", "price": 4700, "seller": "X"},
    {"name": "אייפון 15 פרו 128GB", "price": 4650, "seller": "Y"},
    {"name": "ايفون 15 برو 256GB", "price": 4990, "seller": "Z"},
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deduplicate multilingual product listings and keep the lowest price.")
    parser.add_argument("--input", help="Path to CSV/XLSX/JSON input file.")
    parser.add_argument("--output", default="matched_products.json", help="Path to output JSON file.")
    parser.add_argument("--reviews-output", default="review_pairs.json", help="Path to uncertain-pairs JSON file.")
    parser.add_argument("--name-col", default="name")
    parser.add_argument("--price-col", default="price")
    parser.add_argument("--seller-col", default=None)
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--currency-col", default=None)
    parser.add_argument("--gtin-col", default=None)
    parser.add_argument("--upc-col", default=None)
    parser.add_argument("--ean-col", default=None)
    parser.add_argument("--mpn-col", default=None)
    parser.add_argument("--sku-col", default=None)
    parser.add_argument("--mode", choices=["exact", "family"], default="exact")
    parser.add_argument("--semantic-backend", choices=["local", "tfidf"], default="local")
    parser.add_argument("--disable-semantic", action="store_true")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--match-threshold", type=float, default=0.82)
    parser.add_argument("--review-threshold", type=float, default=0.68)
    parser.add_argument("--demo", action="store_true", help="Run on built-in sample data.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.demo:
        frame = pd.DataFrame(SAMPLE_DATA)
    elif args.input:
        frame = load_input(args.input)
    else:
        raise SystemExit("Provide --input <file> or use --demo.")

    config = MatcherConfig(
        name_col=args.name_col,
        price_col=args.price_col,
        seller_col=args.seller_col,
        id_col=args.id_col,
        currency_col=args.currency_col,
        gtin_col=args.gtin_col,
        upc_col=args.upc_col,
        ean_col=args.ean_col,
        mpn_col=args.mpn_col,
        sku_col=args.sku_col,
        match_mode=args.mode,
        use_semantic=not args.disable_semantic,
        semantic_backend=args.semantic_backend,
        local_model_name=args.local_model,
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
    )

    matcher = ProductMatcher(config)
    try:
        matched, reviews = matcher.run(frame)
        save_json(args.output, matched)
        save_json(args.reviews_output, reviews)
        print(f"Saved matched clusters to: {args.output}")
        print(f"Saved review pairs to: {args.reviews_output}")
        print(json.dumps(matched[:3], ensure_ascii=False, indent=2))
        if reviews:
            print(f"Review pairs: {len(reviews)}")
    finally:
        matcher.close()


if __name__ == "__main__":
    main()
