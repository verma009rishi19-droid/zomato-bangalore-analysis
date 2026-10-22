# Zomato Bangalore Restaurants: Pricing & Rating Analysis

End-to-end analysis of restaurant pricing, ratings, votes and services: messy data cleaned in **Python**, questions answered in **SQL**, and findings backed by confidence intervals rather than raw averages.

## Business questions
1. Does a higher price mean a better rating?
2. Do online ordering and table booking go with higher ratings?
3. Does the listing type (Dining, Cafes, Buffet, Other) matter?
4. How trustworthy are the ratings, and which restaurants are genuinely good value?

## Dataset
`data/raw/zomato_data.csv`: 148 rows, 7 columns (`name`, `online_order`, `book_table`, `rate`, `votes`, `approx_cost(for two people)`, `listed_in(type)`). It has **no location or cuisine columns**, so the analysis covers price, rating, votes, services and listing type only.

## Pipeline
```
data/raw ──► src/clean_data.py ──► data/processed ──► src/eda.py ──► reports/ (7 charts + insights.md)
                                        │
                                        └──► src/load_to_postgres.py ──► sql/queries.sql (9 queries)
```

| Step | File | What it does |
|---|---|---|
| Clean | `src/clean_data.py` | Parses `4.1/5` and `3.8 /5`, fixes ALL-CAPS names, merges restaurants listed under several types, builds price bands and a **vote-adjusted (Bayesian) rating** |
| Analyse | `src/eda.py` | Charts, Spearman and Mann-Whitney tests, bootstrap 95% CIs, OLS regression, auto-written insights |
| SQL | `sql/schema.sql`, `sql/queries.sql` | PostgreSQL schema and 9 queries (CTEs, window functions, RANK, ROW_NUMBER) |
| Notebook | `notebooks/zomato_analysis.ipynb` | Narrated walkthrough, already executed |
| Tests | `tests/` | pytest: parsing, merging, Bayesian maths, and all 9 SQL queries |

## Cleaning decisions
- **Duplicates:** 3 restaurants appear under two listing types. They become one restaurant row plus separate listing rows, so restaurant-level stats are not double counted.
- **Ratings:** parsed to numbers; the one malformed value (`3.8 /5`) is repaired. No ratings were imputed or dropped.
- **Low-confidence ratings:** 75 of 145 restaurants have under 50 votes and 33 have a rating with **zero** votes. They are kept but flagged (`reliable_rating`), and key results are re-checked on the 50+ vote subset.
- **Vote-adjusted rating:** shrinks each rating toward the overall mean as if the restaurant had 50 extra average votes, so a 4.1 from 0 votes no longer outranks a 4.1 from 800.

## Key insights
1. The dataset has **145** restaurants. Median rating is **3.7**, median cost for two **₹400**, and median votes only **42**: votes are heavily right-skewed (max 4,884).
2. **75 of 145 restaurants (52%) have fewer than 50 votes**, and **33** carry a rating with zero votes. Those ratings are weak evidence, so conclusions below for price and online ordering are re-checked on the reliable subset.
3. Price and rating are positively related (Spearman ρ = **0.31**, p = 0.000178) across all restaurants, but ρ = **0.17** (p = 0.171, n=70) among those with 50+ votes, and the link is not statistically clear once low-vote restaurants are excluded. Average rating runs from 3.51 in the ₹100-200 band to 3.82 at 601-800; the top band has only 5 restaurants, so its interval is wide.
4. Online ordering: average rating **3.85** with (n=56) vs **3.49** without (n=89); difference +0.36, 95% CI [+0.24, +0.47], Mann-Whitney p = 2.16e-08; the interval excludes zero. Among restaurants with 50+ votes (n=70): +0.16, 95% CI [-0.04, +0.37]. Median cost differs too (₹500 with vs ₹300 without), so price may confound this.
5. Table booking: average rating **4.13** with (n=7) vs **3.60** without (n=138); difference +0.53, 95% CI [+0.31, +0.73], Mann-Whitney p = 0.00103; the interval excludes zero. Only a handful of restaurants take bookings, so treat this as a hint, not a finding.
6. Listing type is associated with rating: average rating by type is Other 3.91 (n=8), Buffet 3.84 (n=7), Cafes 3.77 (n=23), Dining 3.57 (n=110). Kruskal-Wallis p = **0.0113**. Dining holds most of the data, so the other groups are small.
7. Votes and rating move together (Spearman ρ = **0.64**, p = 5.84e-18); popularity and quality probably feed each other. Restaurants with under 50 votes average 3.45 vs 3.82 for the rest, so thinly reviewed places do not just look noisier, they rate lower.
8. A vote-adjusted rating (Bayesian average, prior worth 50 votes) keeps 9 of the raw top 10 in the adjusted top 10. Top 5 after adjustment: Onesta (4.58, 2,556 votes); Meghana Foods (4.39, 4,401 votes); Empire Restaurant (4.39, 4,884 votes); Corner House Ice Cream (4.21, 345 votes); Szechuan Dragon (4.18, 1,647 votes). Biggest downward corrections: Spicy Tandoor (4.1 to 3.63, 0 votes); Swathi Cool Point (4.1 to 3.63, 0 votes); Shree Cool Point (4.1 to 3.80, 28 votes).
9. **7** restaurants combine a 4.0+ rating, 50+ votes, and cost at or below the median ₹400: Corner House Ice Cream (4.3, ₹400); Kabab Magic (4.1, ₹400); The Biryani Cafe (4.1, ₹300); Sri Guru Kottureshwara Davangere Benne Dosa (4.1, ₹150); Frozen Bottle (4.2, ₹400). A further **4** score 4.0+ on thin evidence and are worth watching, not trusting.
10. A regression of rating on cost, online ordering, table booking and listing type explains **27%** of the variation. Effects whose intervals exclude zero: Online ordering +0.30 [+0.18, +0.43]; Table booking +0.37 [+0.10, +0.65]. Online ordering, adjusted for the other factors: +0.30 [+0.18, +0.43]. Associations only, not causes.

## Charts
![Rating by price band](reports/figures/02_rating_by_price_band.png)
![Online ordering and table booking](reports/figures/03_services_vs_rating.png)
![Adjusted ranking](reports/figures/06_adjusted_ranking.png)
![Regression coefficients](reports/figures/07_regression_coefficients.png)

## Run it
```bash
pip install -r requirements.txt
python src/clean_data.py
python src/eda.py        # run from the repo root
pytest
```

**Optional: SQL layer**
```bash
export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/zomato
python src/load_to_postgres.py
psql -d zomato -f sql/queries.sql
```

## Limitations
- Only 145 restaurants; the table-booking (n=7) and several listing-type groups are very small.
- Many ratings rest on few votes, and ratings are self-selected.
- All findings are associations. For example, online ordering may simply go with pricier, more established restaurants.
- No location or cuisine data, so area and cuisine questions cannot be answered from this file.

## Next steps
- Build a Power BI or Tableau dashboard on `data/processed/zomato_clean.csv`, filtered by price band, listing type and vote threshold.
- Repeat the analysis on a larger Zomato dataset that includes location and cuisine.

## Tech stack
Python (pandas, NumPy, SciPy, matplotlib) · PostgreSQL · Jupyter · pytest

## Author
Rishi Verma · [LinkedIn](https://linkedin.com/in/rishi-verma8973972b5) · [GitHub](https://github.com/verma009rishi19-droid)
