"""
io_utils.py
-----------
File I/O helpers for loading input data and saving JSON results.

Contains:
- load_input()   – read CSV / Excel / JSON into a pandas DataFrame
- save_json()    – write any serialisable object to a JSON file (UTF-8)
- SAMPLE_DATA    – built-in demo dataset with mixed product types
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd

SAMPLE_DATA: List[Dict[str, Any]] = [
    # --- Phones (multilingual) ---
    {"name": "Samsung Galaxy S23",         "price": 3150, "seller": "A"},
    {"name": "סמסונג גלקסי 23",             "price": 3100, "seller": "B"},
    {"name": "سامسونج جالكسي S23",          "price": 3200, "seller": "C"},
    {"name": "iPhone 15 Pro 128GB",        "price": 4700, "seller": "D"},
    {"name": "אייפון 15 פרו 128GB",         "price": 4650, "seller": "E"},
    # --- Laptops ---
    {"name": "Dell XPS 15 16GB 512GB SSD", "price": 5200, "seller": "F"},
    {"name": "Dell XPS 15 16GB 512GB",     "price": 5100, "seller": "G"},
    {"name": "Dell XPS 15 32GB 1TB SSD",   "price": 6800, "seller": "H"},
    # --- Headphones ---
    {"name": "Sony WH-1000XM5 Black",      "price": 1400, "seller": "I"},
    {"name": "Sony WH1000XM5 black",       "price": 1350, "seller": "J"},
    {"name": "Sony WH-1000XM4 Black",      "price": 1100, "seller": "K"},
]


def load_input(path: str) -> pd.DataFrame:
    """Read a product list from CSV, Excel, or JSON."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_obj)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_obj)
    if suffix == ".json":
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported file format '{suffix}'. Use .csv, .xlsx, .xls, or .json")


def save_json(path: str, data: object) -> None:
    """Write *data* as pretty-printed, UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
