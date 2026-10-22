"""EDA + statistics. Produces charts in reports/figures/ and reports/insights.md.

Run:  python src/eda.py
Every number in insights.md is computed here. With only ~145 restaurants, group
comparisons carry bootstrap 95% confidence intervals and are described as
associations, not causes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from clean_data import MIN_VOTES, PRICE_LABELS, PRIOR_WEIGHT

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, REPORT = ROOT / "data" / "processed", ROOT / "reports" / "figures", ROOT / "reports" / "insights.md"
RED, GREY, LIGHT = "#E23744", "#444444", "#BBBBBB"
N_BOOT = 5000
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})


# ---------- helpers ----------
def save(name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG / name)
    plt.close()


def boot_mean_ci(x, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    means = np.random.default_rng(seed).choice(x, (N_BOOT, len(x))).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def boot_diff_ci(a, b, seed: int = 0) -> tuple[float, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (N_BOOT, len(a))).mean(axis=1) - rng.choice(b, (N_BOOT, len(b))).mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def higher(x: float) -> str:
    return "higher" if x > 0 else "lower"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    rest = pd.read_csv(DATA / "zomato_clean.csv")
    listings = pd.read_csv(DATA / "zomato_listings.csv")
    return rest, listings


def describe_group_gap(df: pd.DataFrame, col: str, label: str) -> tuple[str, tuple]:
    a, b = df.loc[df[col], "rating"], df.loc[~df[col], "rating"]
    diff, (lo, hi) = a.mean() - b.mean(), boot_diff_ci(a, b)
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    verdict = "the interval includes zero, so this is not clearly different" if lo <= 0 <= hi else "the interval excludes zero"
    text = (f"{label}: average rating **{a.mean():.2f}** with (n={len(a)}) vs **{b.mean():.2f}** without (n={len(b)}); "
            f"difference {diff:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], Mann-Whitney p = {p:.3g}; {verdict}.")
    return text, (a, b, diff, lo, hi)


# ---------- analysis ----------
def run(rest: pd.DataFrame, listings: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    ld = listings.merge(rest.drop(columns=["listed_types"]), on="restaurant_id")   # 148 listing rows
    median_cost = rest["cost_for_two"].median()

    # 1 overview ------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].hist(rest["rating"], bins=np.arange(2.5, 4.8, 0.1), color=RED, edgecolor="white")
    ax[0].set(title="Rating", xlabel="Rating (out of 5)", ylabel="Restaurants")
    ax[1].hist(rest["cost_for_two"], bins=np.arange(50, 1001, 50), color=GREY, edgecolor="white")
    ax[1].set(title="Cost for two", xlabel="INR")
    ax[2].hist(rest["votes"] + 1, bins=np.logspace(0, np.log10(rest["votes"].max() + 1), 20), color=LIGHT, edgecolor="white")
    ax[2].set(xscale="log", title="Votes (log scale, +1)", xlabel="Votes + 1")
    save("01_distributions.png")
    low = (~rest["reliable_rating"]).sum()
    notes.append(f"The dataset has **{len(rest)}** restaurants. Median rating is **{rest['rating'].median():.1f}**, median cost for two "
                 f"**₹{median_cost:,.0f}**, and median votes only **{rest['votes'].median():.0f}**: votes are heavily right-skewed "
                 f"(max {rest['votes'].max():,}).")
    notes.append(f"**{low} of {len(rest)} restaurants ({low / len(rest):.0%}) have fewer than {MIN_VOTES} votes**, and "
                 f"**{(rest['votes'] == 0).sum()}** carry a rating with zero votes. Those ratings are weak evidence, so conclusions "
                 "below for price and online ordering are re-checked on the reliable subset.")

    # 2 price vs rating -----------------------------------------------
    rho, p = stats.spearmanr(rest["cost_for_two"], rest["rating"])
    rel = rest[rest["reliable_rating"]]
    rho_rel, p_rel = stats.spearmanr(rel["cost_for_two"], rel["rating"])
    bands = rest.groupby("price_band")["rating"].agg(["mean", "count"]).reindex(PRICE_LABELS)
    cis = np.array([boot_mean_ci(rest.loc[rest["price_band"] == b, "rating"]) for b in PRICE_LABELS])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(PRICE_LABELS, bands["mean"], color=RED, yerr=[bands["mean"] - cis[:, 0], cis[:, 1] - bands["mean"]], capsize=4)
    for i, (m, n) in enumerate(zip(bands["mean"], bands["count"])):
        ax.text(i, 2.65, f"n={n}", ha="center", color="white", fontsize=9)
    ax.set(title="Average rating by price band (95% CI)", xlabel="Cost for two (INR)", ylabel="Avg rating", ylim=(2.5, 4.4))
    save("02_rating_by_price_band.png")
    robust = ("and the link is not statistically clear once low-vote restaurants are excluded" if p_rel > 0.05
              else "and it holds up among restaurants with enough votes")
    notes.append(f"Price and rating are positively related (Spearman ρ = **{rho:.2f}**, p = {p:.3g}) across all restaurants, but "
                 f"ρ = **{rho_rel:.2f}** (p = {p_rel:.3g}, n={len(rel)}) among those with {MIN_VOTES}+ votes, {robust}. "
                 f"Average rating runs from {bands['mean'].iloc[0]:.2f} in the ₹100-200 band to {bands['mean'].max():.2f} at "
                 f"{bands['mean'].idxmax()}; the top band has only {int(bands['count'].iloc[-1])} restaurants, so its interval is wide.")

    # 3 online ordering & table booking --------------------------------
    t_on, (a1, b1, *_) = describe_group_gap(rest, "online_order", "Online ordering")
    t_bk, (a2, b2, *_) = describe_group_gap(rest, "book_table", "Table booking")
    cost_on = (rest.loc[rest["online_order"], "cost_for_two"].median(), rest.loc[~rest["online_order"], "cost_for_two"].median())
    on_rel, off_rel = rel.loc[rel["online_order"], "rating"], rel.loc[~rel["online_order"], "rating"]
    d_rel, (lo_rel, hi_rel) = on_rel.mean() - off_rel.mean(), boot_diff_ci(on_rel, off_rel)
    t_on += f" Among restaurants with {MIN_VOTES}+ votes (n={len(rel)}): {d_rel:+.2f}, 95% CI [{lo_rel:+.2f}, {hi_rel:+.2f}]."
    notes += [t_on + f" Median cost differs too (₹{cost_on[0]:,.0f} with vs ₹{cost_on[1]:,.0f} without), so price may confound this.",
              t_bk + " Only a handful of restaurants take bookings, so treat this as a hint, not a finding."]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    rng = np.random.default_rng(1)
    for axis, (title, groups) in zip(ax, [("Online ordering", (a1, b1)), ("Table booking", (a2, b2))]):
        for i, (g, color, lab) in enumerate(zip(groups, [RED, GREY], ["Yes", "No"])):
            axis.scatter(np.full(len(g), i) + rng.uniform(-0.15, 0.15, len(g)), g, s=14, alpha=0.5, color=color)
            axis.hlines(g.mean(), i - 0.3, i + 0.3, color="k", lw=2)
            axis.text(i, 2.45, f"{lab}\nn={len(g)}", ha="center", fontsize=9)
        axis.set(title=title, xticks=[], xlim=(-0.6, 1.6), ylim=(2.3, 4.8))
    ax[0].set_ylabel("Rating (black bar = mean)")
    save("03_services_vs_rating.png")

    # 4 listing type ---------------------------------------------------
    order = ld.groupby("listed_type")["rating"].mean().sort_values(ascending=False).index.tolist()
    groups = [ld.loc[ld["listed_type"] == t, "rating"] for t in order]
    kw = stats.kruskal(*groups)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, g in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(g)), g, s=14, alpha=0.5, color=RED)
        ax.hlines(g.mean(), i - 0.3, i + 0.3, color="k", lw=2)
        ax.text(i, 2.4, f"n={len(g)}", ha="center", fontsize=9)
    ax.set(xticks=range(len(order)), xticklabels=order, ylim=(2.3, 4.8), ylabel="Rating (black bar = mean)", title="Rating by listing type")
    save("04_rating_by_listing_type.png")
    desc = ", ".join(f"{t} {g.mean():.2f} (n={len(g)})" for t, g in zip(order, groups))
    notes.append(f"Listing type is associated with rating: average rating by type is {desc}. Kruskal-Wallis p = **{kw.pvalue:.3g}**. Dining holds most of the data, so the other groups are small.")

    # 5 votes vs rating + reliability -----------------------------------
    rho_v, p_v = stats.spearmanr(rest["votes"], rest["rating"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for flag, color, lab in [(False, LIGHT, f"< {MIN_VOTES} votes"), (True, RED, f"{MIN_VOTES}+ votes")]:
        s = rest[rest["reliable_rating"] == flag]
        ax.scatter(s["votes"] + 1, s["rating"], s=18, alpha=0.7, color=color, label=lab)
    ax.axvline(MIN_VOTES, ls="--", color="k", lw=1)
    ax.set(xscale="log", xlabel="Votes + 1 (log scale)", ylabel="Rating", title="Do popular restaurants rate higher?")
    ax.legend(frameon=False)
    save("05_votes_vs_rating.png")
    notes.append(f"Votes and rating move together (Spearman ρ = **{rho_v:.2f}**, p = {p_v:.3g}); popularity and quality probably feed each other. "
                 f"Restaurants with under {MIN_VOTES} votes average {rest.loc[~rest['reliable_rating'], 'rating'].mean():.2f} "
                 f"vs {rel['rating'].mean():.2f} for the rest, so thinly reviewed places do not just look noisier, they rate lower.")

    # 6 Bayesian re-rank -----------------------------------------------
    top_raw = rest.sort_values(["rating", "votes"], ascending=False).head(10)
    top_bayes = rest.sort_values("bayes_rating", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    tb = top_bayes.iloc[::-1]
    ax.hlines(range(len(tb)), tb["bayes_rating"], tb["rating"], color=LIGHT)
    ax.scatter(tb["rating"], range(len(tb)), color=GREY, label="Raw rating", zorder=3)
    ax.scatter(tb["bayes_rating"], range(len(tb)), color=RED, label="Adjusted rating", zorder=3)
    ax.set(yticks=range(len(tb)), yticklabels=[f"{n[:28]} ({v:,} votes)" for n, v in zip(tb["name"], tb["votes"])],
           title="Top 10 by vote-adjusted rating", xlabel="Rating")
    ax.legend(frameon=False, loc="lower right")
    save("06_adjusted_ranking.png")
    overlap = len(set(top_raw["restaurant_id"]) & set(top_bayes["restaurant_id"]))
    drops = rest.assign(shift=rest["rating"] - rest["bayes_rating"]).sort_values("shift", ascending=False).head(3)
    notes.append(f"A vote-adjusted rating (Bayesian average, prior worth {PRIOR_WEIGHT} votes) keeps {overlap} of the raw top 10 in the adjusted top 10. "
                 "Top 5 after adjustment: " + "; ".join(f"{r.name} ({r.bayes_rating:.2f}, {r.votes:,} votes)" for r in top_bayes.head(5).itertuples())
                 + ". Biggest downward corrections: " + "; ".join(f"{r.name} ({r.rating} to {r.bayes_rating:.2f}, {r.votes} votes)" for r in drops.itertuples()) + ".")

    # 7 value picks + unproven ---------------------------------------------
    value = rest[(rest["reliable_rating"]) & (rest["rating"] >= 4.0) & (rest["cost_for_two"] <= median_cost)].sort_values("bayes_rating", ascending=False)
    unproven = rest[(~rest["reliable_rating"]) & (rest["rating"] >= 4.0)]
    notes.append(f"**{len(value)}** restaurants combine a 4.0+ rating, {MIN_VOTES}+ votes, and cost at or below the median ₹{median_cost:,.0f}"
                 + (": " + "; ".join(f"{r.name} ({r.rating}, ₹{r.cost_for_two})" for r in value.head(5).itertuples()) if len(value) else "")
                 + f". A further **{len(unproven)}** score 4.0+ on thin evidence and are worth watching, not trusting.")

    # 8 regression ---------------------------------------------------------
    types = [t for t in order if t != "Dining"]
    X = np.column_stack([np.ones(len(ld)), np.log2(ld["cost_for_two"]), ld["online_order"].astype(int), ld["book_table"].astype(int)]
                        + [(ld["listed_type"] == t).astype(int) for t in types])
    names = ["Intercept", "Cost (per doubling)", "Online ordering", "Table booking"] + [f"Type: {t} (vs Dining)" for t in types]
    y = ld["rating"].to_numpy(float)
    beta = ols(X, y)
    r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    idx = np.random.default_rng(2).integers(0, len(y), (2000, len(y)))
    boots = np.array([ols(X[i], y[i]) for i in idx])
    ci = np.percentile(boots, [2.5, 97.5], axis=0).T
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    k = range(1, len(beta))
    ax.errorbar(beta[1:], list(k), xerr=[beta[1:] - ci[1:, 0], ci[1:, 1] - beta[1:]], fmt="o", color=RED, capsize=4)
    ax.axvline(0, color="k", lw=1)
    ax.set(yticks=list(k), yticklabels=names[1:], title=f"Rating drivers: OLS coefficients, 95% bootstrap CI (R² = {r2:.2f})", xlabel="Change in rating")
    ax.invert_yaxis()
    save("07_regression_coefficients.png")
    clear = [f"{n} {b:+.2f} [{lo:+.2f}, {hi:+.2f}]" for n, b, (lo, hi) in zip(names[1:], beta[1:], ci[1:]) if lo > 0 or hi < 0]
    notes.append(f"A regression of rating on cost, online ordering, table booking and listing type explains **{r2:.0%}** of the variation. "
                 + ("Effects whose intervals exclude zero: " + "; ".join(clear) + "." if clear else "No single effect is clearly different from zero.")
                 + f" Online ordering, adjusted for the other factors: {beta[2]:+.2f} [{ci[2, 0]:+.2f}, {ci[2, 1]:+.2f}]. Associations only, not causes.")
    return notes


def write_report(notes: list[str], n: int) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("# Key insights\n\n" + f"_Generated by `src/eda.py` from {n} restaurants._\n\n"
                      + "\n".join(f"{i}. {t}" for i, t in enumerate(notes, 1)) + "\n")


if __name__ == "__main__":
    r, l = load()
    write_report(run(r, l), len(r))
    print(REPORT.read_text())
