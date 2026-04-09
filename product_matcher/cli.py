"""
cli.py
------
Command-line interface for the product matcher.

Contains:
- build_arg_parser()  – defines all CLI flags (input, output, column names,
                        thresholds, semantic backend, LLM resolution, demo, etc.)
- main()              – parses arguments, runs the matcher, and writes output
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys

import pandas as pd

from .config import DEFAULT_LOCAL_MODEL, MatcherConfig
from .io_utils import SAMPLE_DATA, load_input, save_json
from .matcher import ProductMatcher

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "General-purpose product deduplicator.  Groups duplicate listings "
            "(any product, any language) and surfaces the lowest price."
        ),
    )
    parser.add_argument("--input", help="Path to CSV / XLSX / JSON input file.")
    parser.add_argument("--output", default="matched_products.json",
                        help="Path to output JSON file.")
    parser.add_argument("--reviews-output", default="review_pairs.json",
                        help="Path to uncertain-pairs JSON file.")

    col = parser.add_argument_group("column mapping")
    col.add_argument("--name-col",     default="name")
    col.add_argument("--price-col",    default="price")
    col.add_argument("--seller-col",   default=None)
    col.add_argument("--id-col",       default=None)
    col.add_argument("--currency-col", default=None)
    col.add_argument("--gtin-col",     default=None)
    col.add_argument("--upc-col",      default=None)
    col.add_argument("--ean-col",      default=None)
    col.add_argument("--mpn-col",      default=None)
    col.add_argument("--sku-col",      default=None)

    tuning = parser.add_argument_group("tuning")
    tuning.add_argument("--mode", choices=["exact", "family"], default="exact")
    tuning.add_argument("--disable-semantic", action="store_true")
    tuning.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL)
    tuning.add_argument("--match-threshold", type=float, default=0.82)
    tuning.add_argument("--review-threshold", type=float, default=0.68)

    llm = parser.add_argument_group("LLM features")
    llm.add_argument("--llm-extract", action="store_true",
                     help="Use LLM to extract brand/model/variant/specs from product names.")
    llm.add_argument("--llm-resolve", action="store_true",
                     help="Send uncertain pairs to an LLM for automatic resolution.")
    llm.add_argument("--llm-api-key", default=None,
                     help="OpenAI API key (or set OPENAI_API_KEY env var).")
    llm.add_argument("--llm-model", default="gpt-4o-mini",
                     help="OpenAI model to use for resolution.")
    llm.add_argument("--llm-extract-model", default="gpt-4o-mini",
                     help="OpenAI model to use for feature extraction.")
    llm.add_argument("--llm-max-pairs", type=int, default=50,
                     help="Max number of uncertain pairs to send to the LLM resolver.")

    parser.add_argument("--demo", action="store_true",
                        help="Run on built-in sample data.")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    args = build_arg_parser().parse_args()

    # This code determines how the input DataFrame 'frame' is prepared for product matching:
    # - If the user specifies the --demo flag, it loads a built-in sample dataset into a DataFrame.
    # - Else, if the user provides an input file path via --input, it loads that file into a DataFrame using load_input.
    if args.demo:
        frame = pd.DataFrame(SAMPLE_DATA)
    elif args.input:
        frame = load_input(args.input)
 
    else:
        print("Error: provide --input <file> or use --demo.", file=sys.stderr)
        raise SystemExit(1)

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
        local_model_name=args.local_model,
        match_threshold=args.match_threshold,
        review_threshold=args.review_threshold,
        llm_extract=args.llm_extract,
        llm_extract_model=args.llm_extract_model,
        llm_resolve=args.llm_resolve,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_max_pairs=args.llm_max_pairs,
    )

    matcher = ProductMatcher(config)
    try:
        matched, reviews = matcher.run(frame)
        save_json(args.output, matched)
        save_json(args.reviews_output, reviews)
        print(f"Saved {len(matched)} clusters  -> {args.output}")
        print(f"Saved {len(reviews)} review pairs -> {args.reviews_output}")
        preview = json.dumps(matched[:3], ensure_ascii=False, indent=2)
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        print(preview)
    finally:
        matcher.close()
