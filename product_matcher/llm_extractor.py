"""
llm_extractor.py
----------------
LLM-based structured feature extraction for product names.

Instead of hardcoded brand/model dictionaries, this module sends product names
to an LLM and asks it to return structured fields (brand, model, variant,
specs, category).  Works for any product category in any language.

Results are cached in SQLite so each unique product name is only extracted
once — re-runs and daily updates cost nothing for previously seen names.

Contains:
- LLMExtractor            – batch extraction with SQLite cache
- build_extraction_prompt  – prompt that asks the LLM to parse a product name
- parse_extraction_result  – parses the JSON array the LLM returns
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = (
    "You are a product-data parser. Given a list of product names (possibly in "
    "different languages, with typos), extract structured fields for EACH name.\n\n"
    "Return a JSON **array** with one object per product, in the same order.\n"
    "Each object MUST have exactly these keys:\n"
    '  "brand"    – manufacturer name in English (e.g. "Samsung", "Dell", "Sony") or null\n'
    '  "model"    – model identifier (e.g. "Galaxy S23", "XPS 15", "WH-1000XM5") or null\n'
    '  "variant"  – sub-variant (e.g. "Pro", "Ultra", "Plus", "Touch") or null\n'
    '  "specs"    – dict of notable specs like {"storage": "256GB", "ram": "16GB", '
    '"color": "black"} — only include specs that are explicitly stated in the name\n'
    '  "category" – one of: "phone", "laptop", "tablet", "headphones", "tv", '
    '"smartwatch", "camera", "speaker", "monitor", "other"\n\n'
    "Rules:\n"
    "- Translate non-English brand/model names to their English equivalent.\n"
    "- Normalize storage to GB (e.g. 1TB = 1024GB).\n"
    "- If a field cannot be determined, use null (for specs use {}).\n"
    "- Return ONLY the JSON array. No markdown, no explanation.\n"
)


def build_extraction_prompt(names: List[str]) -> str:
    """Build the user message listing product names to extract."""
    lines = [f"{i+1}. {name}" for i, name in enumerate(names)]
    return "Extract structured fields for these products:\n\n" + "\n".join(lines)


def parse_extraction_result(text: str, expected_count: int) -> List[Dict[str, Any]]:
    """Parse the LLM JSON array response into a list of dicts."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -len("```")]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM extraction response as JSON")
        return [_empty_extraction() for _ in range(expected_count)]

    if not isinstance(result, list):
        result = [result]

    while len(result) < expected_count:
        result.append(_empty_extraction())

    return [_normalize_extracted(item) for item in result[:expected_count]]


def _empty_extraction() -> Dict[str, Any]:
    return {"brand": None, "model": None, "variant": None, "specs": {}, "category": None}


def _normalize_extracted(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return _empty_extraction()
    brand = item.get("brand")
    if isinstance(brand, str):
        brand = brand.strip().lower() or None
    else:
        brand = None
    model = item.get("model")
    if isinstance(model, str):
        model = model.strip().lower() or None
    else:
        model = None
    variant = item.get("variant")
    if isinstance(variant, str):
        variant = variant.strip().lower() or None
    else:
        variant = None
    specs = item.get("specs")
    if not isinstance(specs, dict):
        specs = {}
    specs = {
        str(k).strip().lower(): str(v).strip().lower()
        for k, v in specs.items()
        if k and v
    }
    category = item.get("category")
    if isinstance(category, str):
        category = category.strip().lower() or None
    else:
        category = None
    return {
        "brand": brand,
        "model": model,
        "variant": variant,
        "specs": specs,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _ExtractionCache:
    """SQLite key-value cache for extraction results."""

    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS extractions "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT value FROM extractions WHERE key = ?", (key,),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False)
        self.conn.execute(
            "INSERT OR REPLACE INTO extractions (key, value) VALUES (?, ?)",
            (key, data),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

BATCH_SIZE = 20


@dataclass
class LLMExtractorConfig:
    enabled: bool = True
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    cache_path: str = ".embedding_cache.sqlite3"


class LLMExtractor:
    """Batch-extracts structured product fields via an LLM, with caching."""

    def __init__(self, config: LLMExtractorConfig) -> None:
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM extractor requires an OpenAI API key. "
                "Set OPENAI_API_KEY env var or pass --llm-api-key."
            )
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Run: pip install openai"
            )
        self.client = OpenAI(api_key=api_key)
        self.cache = _ExtractionCache(config.cache_path)

    def extract_batch(self, names: List[str]) -> List[Dict[str, Any]]:
        """Extract structured fields for a list of product names.

        Cached results are reused; only uncached names hit the API.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(names)
        uncached_indices: List[int] = []
        uncached_names: List[str] = []

        cache_prefix = f"extract::{self.config.model}::"
        for i, name in enumerate(names):
            cached = self.cache.get(f"{cache_prefix}{name}")
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_names.append(name)

        if uncached_names:
            logger.info(
                "LLM extracting features for %d/%d products (%d cached)",
                len(uncached_names), len(names), len(names) - len(uncached_names),
            )
            extracted = self._call_llm_batched(uncached_names)
            for idx, ext in zip(uncached_indices, extracted):
                results[idx] = ext
                self.cache.set(f"{cache_prefix}{names[idx]}", ext)

        return [r if r is not None else _empty_extraction() for r in results]

    def _call_llm_batched(self, names: List[str]) -> List[Dict[str, Any]]:
        """Split into chunks of BATCH_SIZE and call the LLM for each chunk."""
        all_results: List[Dict[str, Any]] = []
        for start in range(0, len(names), BATCH_SIZE):
            chunk = names[start: start + BATCH_SIZE]
            all_results.extend(self._call_llm(chunk))
        return all_results

    def _call_llm(self, names: List[str]) -> List[Dict[str, Any]]:
        user_msg = build_extraction_prompt(names)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=300 * len(names),
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            answer = response.choices[0].message.content or "[]"
            return parse_extraction_result(answer, len(names))
        except Exception as exc:
            logger.warning("LLM extraction API call failed: %s", exc)
            return [_empty_extraction() for _ in names]

    def close(self) -> None:
        self.cache.close()
