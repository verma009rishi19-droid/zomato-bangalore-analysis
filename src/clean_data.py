"""Cleaning pipeline: raw Zomato CSV -> analysis-ready tables.

Run:  python src/clean_data.py [path/to/raw.csv]
Outputs (data/processed/):
    zomato_clean.csv      one row per restaurant
    zomato_listings.csv   one row per restaurant x listing type
    cleaning_log.json     what was changed, for the README / notebook
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "zomato_data.csv"
OUT_DIR = ROOT / "data" / "processed"

MIN_VOTES = 50          # below this, a rating is treated as low-confidence
PRIOR_WEIGHT = 50       # Bayesian average: acts like 50 "average" votes added to each restaurant
PRICE_BINS = [0, 200, 400, 600, 800, np.inf]
PRICE_LABELS = ["100-200", "201-400", "401-600", "601-800", "801+"]
RENAME = {"approx_cost(for two people)": "cost_for_two", "listed_in(type)": "listed_type", "rate": "rate_raw"}
DEDUP_KEYS = ["name", "online_order", "book_table", "rating", "votes", "cost_for_two"]


def parse_rating(value) -> float:
    """'4.1/5' or '3.8 /5' -> float. 'NEW', '-', NaN -> NaN (unrated, never 0)."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", str(value))
    return float(match.group(1)) if match else np.nan


def clean_name(value: str) -> str:
    """Collapse whitespace; title-case only names shouted in ALL CAPS ("CAFE NOVA")."""
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value.title() if value.isupper() else value


def bayesian_average(rating: pd.Series, votes: pd.Series, prior_mean: float, weight: int = PRIOR_WEIGHT) -> pd.Series:
    """Shrink ratings with few votes toward the overall mean."""
    return (votes * rating + weight * prior_mean) / (votes + weight)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log: dict = {"raw_rows": int(len(raw)), "raw_columns": list(raw.columns)}
    df = raw.rename(columns=RENAME).copy()

    # 1. Types
    df["name"] = df["name"].map(clean_name)
    df["rating"] = df["rate_raw"].map(parse_rating)
    df["cost_for_two"] = pd.to_numeric(df["cost_for_two"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    for col in ("online_order", "book_table"):
        df[col] = df[col].astype("string").str.strip().str.title().map({"Yes": True, "No": False})
    df["listed_type"] = df["listed_type"].astype("string").str.strip().str.title()
    log["rating_format_fixed"] = int(df["rate_raw"].astype(str).str.contains(r"\s").sum())
    log["names_recased"] = int((df["name"] != raw["name"].astype(str).str.strip()).sum())

    # 2. Validate (fail loudly instead of silently analysing bad rows)
    bad = df[df[["rating", "cost_for_two", "votes", "online_order", "book_table"]].isna().any(axis=1)]
    log["rows_with_missing_core_fields"] = int(len(bad))
    df = df.drop(bad.index)
    assert df["rating"].between(0, 5).all() and (df["cost_for_two"] > 0).all() and (df["votes"] >= 0).all()

    # 3. Same restaurant listed under several types -> one restaurant row, many listing rows
    df["restaurant_id"] = df.groupby(DEDUP_KEYS, sort=False).ngroup() + 1
    listings = df[["restaurant_id", "listed_type"]].drop_duplicates().reset_index(drop=True)
    types = listings.groupby("restaurant_id")["listed_type"].agg("; ".join).rename("listed_types")
    rest = df.drop_duplicates("restaurant_id").drop(columns=["listed_type", "rate_raw"]).set_index("restaurant_id")
    rest = rest.join(types).reset_index()
    log["listing_rows"] = int(len(listings))
    log["restaurants"] = int(len(rest))
    log["multi_listed_restaurants"] = int(rest["listed_types"].str.contains(";").sum())

    # 4. Derived fields
    rest["votes"] = rest["votes"].astype(int)
    rest["cost_for_two"] = rest["cost_for_two"].astype(int)
    rest["price_band"] = pd.cut(rest["cost_for_two"], PRICE_BINS, labels=PRICE_LABELS).astype(str)
    prior = float(rest["rating"].mean())
    rest["bayes_rating"] = bayesian_average(rest["rating"], rest["votes"], prior).round(2)
    rest["reliable_rating"] = rest["votes"] >= MIN_VOTES
    log["low_confidence_ratings"] = int((~rest["reliable_rating"]).sum())
    log["zero_vote_restaurants"] = int((rest["votes"] == 0).sum())
    log["bayesian_prior_mean"] = round(prior, 3)

    cols = ["restaurant_id", "name", "online_order", "book_table", "rating", "votes", "cost_for_two",
            "price_band", "bayes_rating", "reliable_rating", "listed_types"]
    return rest[cols], listings, log


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW
    print(f"Reading {path}")
    rest, listings, log = clean(pd.read_csv(path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rest.to_csv(OUT_DIR / "zomato_clean.csv", index=False)
    listings.to_csv(OUT_DIR / "zomato_listings.csv", index=False)
    log["source_file"] = path.name
    (OUT_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
