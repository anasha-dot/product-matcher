"""
llm_resolver.py
---------------
LLM-based fallback for uncertain product pairs.

When the scoring engine lands a pair in the "review" zone (score between
review_threshold and match_threshold), this module sends both product names
to an LLM and asks: "Are these the same product?"

Contains:
- LLMResolver            – calls the OpenAI chat-completions API
- build_llm_prompt()     – constructs the system + user prompt
- parse_llm_answer()     – extracts yes / no from the model response
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a product-matching expert. Your task is to decide whether two "
    "product listings refer to the **exact same product** (same brand, model, "
    "variant, and storage capacity) or not.\n\n"
    "Rules:\n"
    "- Different storage sizes (e.g. 128GB vs 256GB) = DIFFERENT products.\n"
    "- Different variants (e.g. Pro vs Ultra) = DIFFERENT products.\n"
    "- Same product in different languages or with typos = SAME product.\n"
    "- If you are unsure, lean towards 'no'.\n\n"
    "Respond with ONLY a JSON object: {\"same\": true} or {\"same\": false}. "
    "No explanation."
)


def build_llm_prompt(
    name_a: str,
    name_b: str,
    norm_a: str,
    norm_b: str,
    score: float,
    reasons: List[str],
) -> str:
    """Build the user message sent to the LLM."""
    return (
        f"Product A: {name_a}\n"
        f"Product B: {name_b}\n"
        f"Normalized A: {norm_a}\n"
        f"Normalized B: {norm_b}\n"
        f"Algorithm score: {score:.4f}\n"
        f"Signals: {', '.join(reasons) if reasons else 'none'}\n\n"
        "Are these the exact same product? Respond {\"same\": true} or {\"same\": false}."
    )


def parse_llm_answer(text: str) -> Optional[bool]:
    """Extract the boolean verdict from the LLM response text."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "same" in obj:
            return bool(obj["same"])
    except json.JSONDecodeError:
        pass
    lower = text.lower()
    if '"same": true' in lower or '"same":true' in lower:
        return True
    if '"same": false' in lower or '"same":false' in lower:
        return False
    if lower.startswith("yes") or lower == "true":
        return True
    if lower.startswith("no") or lower == "false":
        return False
    return None


# ---------------------------------------------------------------------------
# Resolver class
# ---------------------------------------------------------------------------

@dataclass
class LLMResolverConfig:
    """Settings for the LLM fallback."""
    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 32
    max_pairs: int = 50


class LLMResolver:
    """Resolves uncertain product pairs by asking an LLM."""

    def __init__(self, config: LLMResolverConfig) -> None:
        self.config = config
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM resolver requires an OpenAI API key. "
                "Set OPENAI_API_KEY env var or pass --llm-api-key."
            )
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "Run: pip install openai"
            )
        self.client = OpenAI(api_key=api_key)

    def resolve_pair(
        self,
        name_a: str,
        name_b: str,
        norm_a: str,
        norm_b: str,
        score: float,
        reasons: List[str],
    ) -> Optional[bool]:
        """Ask the LLM whether two products are the same.

        Returns True (same), False (different), or None (could not parse).
        """
        user_msg = build_llm_prompt(name_a, name_b, norm_a, norm_b, score, reasons)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            answer_text = response.choices[0].message.content or ""
            verdict = parse_llm_answer(answer_text)
            logger.info(
                "LLM verdict for [%s] vs [%s]: %s (raw: %s)",
                name_a, name_b, verdict, answer_text.strip(),
            )
            return verdict
        except Exception as exc:
            logger.warning("LLM API call failed: %s", exc)
            return None

    def resolve_batch(
        self,
        review_pairs: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process a list of review pairs through the LLM.

        Returns (resolved_matches, still_uncertain) where:
        - resolved_matches: pairs the LLM confirmed as same product
        - still_uncertain:  pairs the LLM rejected or could not decide
        """
        matches: List[Dict] = []
        uncertain: List[Dict] = []

        pairs_to_check = review_pairs[: self.config.max_pairs]
        skipped = review_pairs[self.config.max_pairs:]

        logger.info(
            "Sending %d uncertain pairs to LLM (%s)…",
            len(pairs_to_check), self.config.model,
        )

        for pair in pairs_to_check:
            verdict = self.resolve_pair(
                name_a=pair["left_name"],
                name_b=pair["right_name"],
                norm_a=pair["normalized_left"],
                norm_b=pair["normalized_right"],
                score=pair["score"],
                reasons=pair.get("reasons", []),
            )
            pair_with_llm = {**pair, "llm_verdict": verdict}
            if verdict is True:
                matches.append(pair_with_llm)
            else:
                uncertain.append(pair_with_llm)

        uncertain.extend(skipped)
        logger.info(
            "LLM resolved %d matches, %d still uncertain.",
            len(matches), len(uncertain),
        )
        return matches, uncertain
