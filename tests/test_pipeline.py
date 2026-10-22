import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clean_data import PRIOR_WEIGHT, bayesian_average, clean, clean_name, parse_rating  # noqa: E402


def test_parse_rating():
    assert parse_rating("4.1/5") == 4.1
    assert parse_rating("3.8 /5") == 3.8
    assert np.isnan(parse_rating("NEW")) and np.isnan(parse_rating("-")) and np.isnan(parse_rating(np.nan))


def test_clean_name_only_recases_all_caps():
    assert clean_name("CAFE NOVA") == "Cafe Nova"
    assert clean_name("Domino's  Pizza ") == "Domino's Pizza"


def test_bayesian_average_shrinks_low_vote_ratings():
    r, v = pd.Series([4.5, 4.5]), pd.Series([0, 5000])
    out = bayesian_average(r, v, prior_mean=3.5)
    assert out[0] == 3.5                       # no votes -> the prior
    assert 4.4 < out[1] < 4.5                  # many votes -> close to the raw rating
    assert PRIOR_WEIGHT > 0


def _raw():
    return pd.DataFrame({
        "name": ["Cafe Nova", "CAFE NOVA", "Grill"],
        "online_order": ["No", "No", "Yes"], "book_table": ["No", "No", "Yes"],
        "rate": ["3.2/5", "3.2/5", "4.0 /5"], "votes": [11, 11, 300],
        "approx_cost(for two people)": [600, 600, 900], "listed_in(type)": ["Cafes", "Dining", "Dining"],
    })


def test_clean_merges_multi_listed_restaurants():
    rest, listings, log = clean(_raw())
    assert len(rest) == 2 and len(listings) == 3
    assert log["multi_listed_restaurants"] == 1
    assert rest["rating"].tolist() == [3.2, 4.0]
    assert rest.loc[rest["name"] == "Grill", "price_band"].iloc[0] == "801+"


def test_sql_queries_run_on_real_pipeline_output():
    rest, listings, _ = clean(pd.read_csv(ROOT / "data" / "raw" / "zomato_data.csv"))
    con = sqlite3.connect(":memory:")
    rest.to_sql("restaurants", con, index=False)
    listings.to_sql("listings", con, index=False)
    text = "\n".join(l for l in (ROOT / "sql" / "queries.sql").read_text().splitlines() if not l.strip().startswith("--"))
    queries = [q for q in text.split(";") if q.strip()]
    assert len(queries) == 9
    for q in queries:
        assert len(pd.read_sql(q, con)) > 0
