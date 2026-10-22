"""Load the cleaned CSVs into PostgreSQL.

    export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/zomato
    python src/load_to_postgres.py
"""
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL first, e.g. postgresql+psycopg2://user:pass@localhost:5432/zomato")
    rest = pd.read_csv(ROOT / "data" / "processed" / "zomato_clean.csv")
    listings = pd.read_csv(ROOT / "data" / "processed" / "zomato_listings.csv")
    schema = (ROOT / "sql" / "schema.sql").read_text()
    with create_engine(url).begin() as conn:
        for stmt in filter(None, (s.strip() for s in schema.split(";"))):
            conn.execute(text(stmt))
        rest.to_sql("restaurants", conn, if_exists="append", index=False)
        listings.to_sql("listings", conn, if_exists="append", index=False)
    print(f"Loaded {len(rest)} restaurants and {len(listings)} listing rows.")


if __name__ == "__main__":
    main()
