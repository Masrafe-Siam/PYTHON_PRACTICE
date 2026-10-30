"""Reproducible statistical analysis for TU Dortmund M.Sc. Data Science, Summer 2027.

This script reads the OFFICIAL waterquality.csv; it never synthesizes observations.
Question 1: spring versus autumn concentrations for oxygen and nitrate.
Question 2: river-section differences, with identifiable pairwise groups.

Usage:
    python analyze_waterquality.py --data waterquality.csv --out results
    python analyze_waterquality.py --download --out results

Statistical caveat: Tests assume the observations are independent; the assignment
provides no sampling timestamps, pairing identifier, or clustering metadata.
This assumption cannot be proven from the concentration values alone.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import studentized_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.graphics.gofplots import qqplot

DATA_URL = ("https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/"
            "Studium/Studiengaenge-Infos/Data_Science/waterquality.csv")
SEASONS = ["spring", "summer", "autumn", "winter"]
OUTCOMES = ["oxygen", "nitrate"]
ALPHA = 0.05


def normalized(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).casefold())


def discover_column(columns, target: str) -> str:
    aliases = {
        "season": ["season", "seasons", "saison", "jahreszeit"],
        "section": ["section", "sections", "riversection", "riverpart",
                    "segment", "reach", "abschnitt", "location", "site"],
        "oxygen": ["oxygen", "oxygene", "dissolvedoxygen", "o2"],
        "nitrate": ["nitrate", "nitrates", "no3", "nitrat"],
    }[target]
    found = [(c, normalized(c)) for c in columns]
    for a in aliases:
        hits = [c for c, c0 in found if c0 == a]
        if len(hits) == 1:
            return hits[0]
    for a in aliases:
        hits = [c for c, c0 in found if a in c0]
        if len(hits) == 1:
            return hits[0]
    raise ValueError(f"Cannot identify '{target}' column. Available columns: {list(columns)}. "
                     "Edit discover_column() using your CSV header.")


def load_data(source: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load CSV, audit source records, then use complete cases for tests."""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Official dataset not found at {p}. Download it from:\n{DATA_URL}\n"
                                "Place it alongside this script (or use --download).")
    raw = pd.read_csv(p, sep=None, engine="python", decimal=",", encoding="utf-8-sig")
    rename = {discover_column(raw.columns, k): k for k in ["season", "section", *OUTCOMES]}
    if len(rename) != 4:
        raise ValueError(f"Column inference is ambiguous: {rename}")
    data = raw.rename(columns=rename)[["season", "section", *OUTCOMES]].copy()
    source_count = len(data)
    for var in OUTCOMES:
        if data[var].dtype == "object":
            data[var] = data[var].str.replace(",", ".", regex=False).str.strip()
        data[var] = pd.to_numeric(data[var], errors="coerce")
    seasons = data["season"].astype("string").str.strip().str.casefold()
    season_alias = {"fall": "autumn", "herbst": "autumn", "fruehling": "spring",
                    "fruhling": "spring", "frühling": "spring", "sommer": "summer",
                    "winter": "winter"}
    data["season"] = seasons.replace(season_alias)
    data["section"] = data["section"].astype("string").str.strip()
    data.loc[data["section"].isin(["", "<NA>"]), "section"] = pd.NA

    data_nan = {c: int(data[c].isna().sum()) for c in data.columns}
    complete = data.dropna(subset=["season", "section", *OUTCOMES]).copy()
    unrecognized = sorted(set(complete["season"].unique()) - set(SEASONS))
    if unrecognized:
        raise ValueError(f"Unrecognized season labels {unrecognized}; expected {SEASONS}. "
                         "Please inspect the data before continuing.")
    section_vals = sorted(complete["section"].unique(), key=sort_section)
    if len(section_vals) < 2:
        raise ValueError("Need at least two sections for comparison.")
    if not {"spring", "autumn"}.issubset(set(complete["season"])):
        raise ValueError("Both spring and autumn observations are required.")
    negatives = {v: int((complete[v] < 0).sum()) for v in OUTCOMES}
    if any(negatives.values()):
        raise ValueError(f"Unexpected negative concentrations; investigate source: {negatives}")
    complete["season"] = pd.Categorical(complete["season"], categories=SEASONS, ordered=True)
    complete["section"] = pd.Categorical(complete["section"], categories=section_vals, ordered=True)
    counts = complete.groupby(["section", "season"], observed=False).size()
    warnings = []
    if source_count != 900:
        warnings.append(f"Expected 900 original rows per assignment; found {source_count}.")
    if len(section_vals) != 5:
        warnings.append(f"Expected five sections; found {len(section_vals)}.")
    if any(counts != 45):
        warnings.append("Not all section x season combinations contain the expected 45 complete observations.")
    if source_count != len(complete):
        warnings.append(f"Removed {source_count - len(complete)} incomplete rows before statistical tests.")
    audit = {
        "source_rows": source_count,
        "complete_rows": len(complete),
        "columns_original": list(map(str, raw.columns)),
        "column_mapping": {k: str(v) for k, v in rename.items()},
        "missing_values_after_conversion": data_nan,
        "identical_rows_found_not_removed": int(complete.duplicated().sum()),
        "section_labels": section_vals,
        "cell_counts": counts.reset_index(name="n"),
        "warnings": warnings,
    }
    return complete, audit


def sort_section(s: str):
    try:
        return (0, float(str(s)), str(s))
    except (ValueError, TypeError):
        return (1, str(s))


def describe(data: pd.DataFrame, by: list[str], var: str) -> pd.DataFrame:
    result = []
    for key, g in data.groupby(by, observed=True, sort=False):
        keys = (key,) if not isinstance(key, tuple) else key
        x = g[var].to_numpy(dtype=float)
        q1, q3 = np.quantile(x, [.25, .75])
        r = dict(zip(by, [str(v) for v in keys]))
        r.update(variable=var, n=len(x), mean=np.mean(x), median=np.median(x),
                 sd=np.std(x, ddof=1) if len(x) > 1 else np.nan,
                 min=np.min(x), q1=q1, q3=q3, iqr=q3-q1, max=np.max(x))
        result.append(r)
    return pd.DataFrame(result)


def welch_compare(a: np.ndarray, b: np.ndarray, alpha=ALPHA):
    """Independent-sample Welch test, CI of mean(a)-mean(b), Hedges' g."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se2 = v1/n1 + v2/n2
    if se2 <= 0:
        raise ValueError("Zero standard error for Welch comparison")
    t = (m1-m2)/np.sqrt(se2)
    df = se2**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    p = 2*stats.t.sf(abs(t), df)
    half_width = stats.t.ppf(1-alpha/2, df)*np.sqrt(se2)
    sp = np.sqrt(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))
    g = (m1-m2)/sp * (1 - 3/(4*(n1+n2-2)-1)) if sp > 0 else np.nan
    return {"n_a": n1, "n_b": n2, "mean_a": m1, "mean_b": m2,
            "difference_a_minus_b": m1-m2, "t": t, "df": df, "p": p,
            "ci95_low": m1-m2-half_width, "ci95_high": m1-m2+half_width,
            "hedges_g": g}


def season_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in OUTCOMES:
        a = data.loc[data["season"] == "spring", var].to_numpy()
        b = data.loc[data["season"] == "autumn", var].to_numpy()
        rows.append({"outcome": var, "comparison": "spring - autumn", **welch_compare(a, b)})
    frame = pd.DataFrame(rows)
    frame["p_holm_two_outcomes"] = multipletests(frame["p"], method="holm")[1]
    frame["significant_holm_05"] = frame["p_holm_two_outcomes"] < ALPHA
    return frame


def pairwise_games_howell(data: pd.DataFrame, var: str,
                           section_col="section") -> pd.DataFrame:
    """Games-Howell simultaneous pairwise comparisons for unequal variances.

    Differences are mean(group_a)-mean(group_b). Critical values are from
    the studentized range with k groups and Welch-Satterthwaite degrees of freedom.
    """
    grouped = [(str(k), grp[var].to_numpy(dtype=float))
               for k, grp in data.groupby(section_col, observed=True)]
    k = len(grouped)
    rows = []
    for (ga, a), (gb, b) in itertools.combinations(grouped, 2):
        n1, n2 = len(a), len(b)
        v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
        se2 = v1/n1+v2/n2
        diff = np.mean(a)-np.mean(b)
        df = se2**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        q = np.sqrt(2)*abs(diff)/np.sqrt(se2)
        p_adj = studentized_range.sf(q, k, df)
        half = studentized_range.ppf(1-ALPHA, k, df)*np.sqrt(se2/2)
        rows.append({"group_a": ga, "group_b": gb, "difference_a_minus_b": diff,
                     "p_adjusted": p_adj, "ci95_low": diff-half,
                     "ci95_high": diff+half, "welch_df": df,
                     "significant_05": p_adj < ALPHA})
    return pd.DataFrame(rows)


def section_tests(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    omnibus, pairwise, posthoc_selection = [], [], []
    for var in OUTCOMES:
        grouped = [grp[var].to_numpy(dtype=float)
                   for _, grp in data.groupby("section", observed=True)]
        f_w, p_w = stats.f_oneway(*grouped, equal_var=False)
        lev, p_lev = stats.levene(*grouped, center="median")
        pooled = np.concatenate(grouped)
        grand = pooled.mean()
        ss_between = sum(len(g)*(g.mean()-grand)**2 for g in grouped)
        ss_total = ((pooled-grand)**2).sum()
        omnibus.append(dict(outcome=var, method="Welch one-way ANOVA (all seasons pooled)",
                            welch_F=f_w, p=p_w, levene_F=lev, levene_p=p_lev,
                            descriptive_eta_squared=ss_between/ss_total))
        # Welch omnibus pairs naturally with Games-Howell: no preliminary variance
        # pretest is needed, and its studentized-range p-values address multiplicity.
        method = "Games-Howell (Welch-compatible, variance-robust)"
        pairs = pairwise_games_howell(data, var)
        pairs.insert(0, "outcome", var)
        pairs.insert(1, "method", method)
        pairwise.append(pairs)
        posthoc_selection.append({"outcome": var, "selected": method})
    return pd.DataFrame(omnibus), pd.concat(pairwise, ignore_index=True), pd.DataFrame(posthoc_selection)


def factorial_tests(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Two-way fixed-effects season × section OLS; Type-II ANOVA per outcome."""
    rows, diag, residuals = [], [], {}
    for var in OUTCOMES:
        model = ols(f"{var} ~ C(season, Sum) * C(section, Sum)", data=data).fit()
        anova = anova_lm(model, typ=2)
        denom = float(anova.loc["Residual", "sum_sq"])
        for term in anova.index:
            if term == "Residual":
                continue
            a = anova.loc[term]
            rows.append({"outcome": var, "term": term, "F": a["F"],
                         "df_term": a["df"], "df_residual": anova.loc["Residual", "df"],
                         "p": a["PR(>F)"],
                         "partial_eta_squared": a["sum_sq"]/(a["sum_sq"]+denom)})
        resid = model.resid.to_numpy()
        shap_w, shap_p = stats.shapiro(resid)
        lev_stat, lev_p = stats.levene(*[
            g[var].to_numpy() for _, g in data.groupby(["season", "section"], observed=True)
        ], center="median")
        diag.append({"outcome": var, "residual_shapiro_W": shap_w,
                     "residual_shapiro_p": shap_p,
                     "levene_20_cells_F": lev_stat, "levene_20_cells_p": lev_p,
                     "R_squared": model.rsquared})
        residuals[var] = resid
    return pd.DataFrame(rows), pd.DataFrame(diag), residuals


def interaction_posthoc(data: pd.DataFrame, factorial: pd.DataFrame) -> pd.DataFrame:
    """Season-specific pairwise section Welch tests, Holm across all season/pair tests per outcome.

    Only reported for outcomes with a significant season × section interaction.
    """
    all_rows = []
    for var in OUTCOMES:
        interaction = factorial[(factorial.outcome == var) &
                                (factorial.term.str.contains(":", regex=False))]
        if interaction.empty or float(interaction["p"].iloc[0]) >= ALPHA:
            continue
        rows = []
        for season in SEASONS:
            sliced = data.loc[data["season"] == season]
            group_data = [(str(s), g[var].to_numpy()) for s, g in sliced.groupby("section", observed=True)]
            for (a_name, a), (b_name, b) in itertools.combinations(group_data, 2):
                r = welch_compare(a, b)
                rows.append({"outcome": var, "season": season, "group_a": a_name,
                             "group_b": b_name, **r})
        if rows:
            f = pd.DataFrame(rows)
            f["holm_adjusted_p_all_seasons"] = multipletests(f.p, method="holm")[1]
            f["significant_holm_05"] = f["holm_adjusted_p_all_seasons"] < ALPHA
            all_rows.append(f)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["outcome", "season", "group_a", "group_b", "p", "holm_adjusted_p_all_seasons"])


def make_plots(data: pd.DataFrame, residuals: dict, out: Path):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.18)
    palette = ["#2F6B83", "#A36B32", "#4D8064", "#99586F"]
    titles = {"oxygen": "Dissolved oxygen", "nitrate": "Nitrate"}
    for var in OUTCOMES:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        sns.boxplot(data=data, x="season", y=var, order=SEASONS, hue="season",
                    palette=dict(zip(SEASONS, palette)), dodge=False, legend=False,
                    showfliers=True, linewidth=1.05, ax=ax)
        ax.set(xlabel="Season", ylabel=f"{titles[var]} (mg/L)",
               title=f"{titles[var]} concentrations across seasons")
        fig.tight_layout(); fig.savefig(out / f"figure_{var}_season_boxplot.png", dpi=300);plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        sns.boxplot(data=data, x="section", y=var, order=list(data.section.cat.categories),
                    color="#7BA0B6", showfliers=True, linewidth=1.05, ax=ax)
        ax.set(xlabel="River section (headwaters to mouth)",
               ylabel=f"{titles[var]} (mg/L)",
               title=f"{titles[var]} concentrations across river sections")
        fig.tight_layout(); fig.savefig(out / f"figure_{var}_section_boxplot.png", dpi=300);plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        grouped = data.groupby(["season", "section"], observed=True)[var].agg(["mean", "std", "count"])
        for i, season in enumerate(SEASONS):
            if season not in set(data.season.astype(str)):
                continue
            vals = grouped.loc[season].reindex(data.section.cat.categories)
            # Descriptive 95% Student-t confidence intervals for cell means.
            errors = stats.t.ppf(.975, vals["count"]-1)*vals["std"]/np.sqrt(vals["count"])
            ax.errorbar(range(1, len(vals)+1), vals["mean"], yerr=errors,
                        marker="o", markersize=5, linewidth=1.5, capsize=3,
                        color=palette[i], label=season.capitalize())
        ax.set_xticks(range(1, len(data.section.cat.categories)+1),
                      list(data.section.cat.categories))
        ax.set(xlabel="River section (headwaters to mouth)",
               ylabel=f"Mean {titles[var].lower()} (mg/L)",
               title=f"{titles[var]}: section × season means (95% CI)")
        ax.legend(title="Season", frameon=True, ncol=2)
        fig.tight_layout(); fig.savefig(out / f"figure_{var}_interaction.png", dpi=300);plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.7, 4.7))
        qqplot(residuals[var], line="45", fit=True, ax=ax, markerfacecolor="#5A8195",
               markeredgecolor="#5A8195", alpha=0.45)
        ax.set_title(f"{titles[var]}: normal Q–Q plot of model residuals")
        fig.tight_layout(); fig.savefig(out / f"figure_{var}_residuals_qq.png", dpi=300);plt.close(fig)


def write_summary(out: Path, audit, season, sections, pairs, factorial, diagnostics, interactions):
    lines = ["# VERIFIED RUN OUTPUT — TU Dortmund Summer 2027 waterquality.csv",
             "", "Results in this file are computed from the local CSV; no fabricated data are used.",
             "", "## Data audit",
             f"- Original observations: {audit['source_rows']}",
             f"- Complete observations used: {audit['complete_rows']}",
             f"- Original columns: {', '.join(audit['columns_original'])}",
             f"- River sections: {', '.join(audit['section_labels'])}",
             f"- Identical rows retained: {audit['identical_rows_found_not_removed']}"]
    if audit["warnings"]:
        lines += ["- Warnings: " + "; ".join(audit["warnings"])]
    lines += ["", "## Research question 1: spring vs autumn", "",
              "Welch independent-samples t-tests; differences defined as spring minus autumn.",
              "Familywise Holm adjustment across the two chemical outcomes.", ""]
    for _, r in season.iterrows():
        lines += [f"- **{r.outcome.capitalize()}**: spring = {r.mean_a:.4g} mg/L; "
                  f"autumn = {r.mean_b:.4g} mg/L; difference = {r.difference_a_minus_b:+.4g} mg/L; "
                  f"t({r.df:.2f}) = {r.t:.4g}; unadjusted p = {r.p:.4g}; "
                  f"Holm-adjusted p = {r.p_holm_two_outcomes:.4g}; "
                  f"95% CI [{r.ci95_low:.4g}, {r.ci95_high:.4g}]; Hedges g = {r.hedges_g:.3g}."]
    lines += ["", "## Research question 2: differences between river sections", "",
              "Primary one-way omnibus test: Welch ANOVA, season-pooled marginal section means.",
              "Post-hoc method: Games–Howell (appropriate for Welch ANOVA and unequal variances).", ""]
    for _, r in sections.iterrows():
        df_pairs = pairs[(pairs.outcome == r.outcome) & (pairs.significant_05.astype(str).str.lower() == "true")]
        lines += [f"- **{r.outcome.capitalize()}**: Welch F = {r.welch_F:.4g}, p = {r.p:.4g}; "
                  f"Levene p = {r.levene_p:.4g}; descriptive eta squared = {r.descriptive_eta_squared:.4g}."]
        if len(df_pairs):
            for _, s in df_pairs.iterrows():
                lines += [f"  - {s.group_a} vs {s.group_b}: estimated difference (a−b) "
                          f"{s.difference_a_minus_b:+.4g} mg/L; adjusted p = {s.p_adjusted:.4g}; "
                          f"CI [{s.ci95_low:.4g}, {s.ci95_high:.4g}]."]
        else:
            lines += ["  - No statistically significant pairwise comparisons under the selected test."]
    lines += ["", "## Two-factor analysis: season, section, interaction", "",
              "Type-II ANOVA from two-way fixed-effects OLS, season × river section."]
    for _, r in factorial.iterrows():
        lines += [f"- {r.outcome}: {r.term}, F({r.df_term:.0f}, {r.df_residual:.0f}) "
                  f"= {r.F:.4g}, p = {r.p:.4g}, partial eta² = {r.partial_eta_squared:.4g}."]
    lines += ["", "## Diagnostics", ""]
    for _, r in diagnostics.iterrows():
        lines += [f"- {r.outcome}: Shapiro–Wilk residual p = {r.residual_shapiro_p:.4g}; "
                  f"Levene across 20 cells p = {r.levene_20_cells_p:.4g}; "
                  f"model R² = {r.R_squared:.4g}."]
    if len(interactions):
        lines += ["", "## Significant interaction: simple section contrasts", "",
                  "See `interaction_posthoc.csv`. Holm correction spans all seasonal section-pair tests per outcome."]
    lines += ["", "## Reporting caveats", "",
              "- Observational concentration differences are not necessarily causal or environmentally harmful.",
              "- No sampling dates, spatial autocorrelation metadata or measurement pairing are specified in the assignment.",
              "  Independence and the absence of temporal/spatial clustering are not verifiable from the supplied description.",
              "- Standard tests use a 0.05 significance threshold; nonsignificance is not evidence of equivalence.",
              "- An interaction that is statistically significant makes season-averaged section comparisons potentially incomplete.",
              "- Report numerical effect sizes and confidence intervals in addition to p-values.", "",
              f"## Computational environment", "",
              f"Python {sys.version.split()[0]}; pandas {pd.__version__}; numpy {np.__version__}; "
              f"scipy {scipy.__version__}; statsmodels {statsmodels.__version__}; "
              f"matplotlib {matplotlib.__version__}; seaborn {sns.__version__}.", ""]
    (out / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")


def download_official(path: Path):
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official CSV: {DATA_URL}")
    try:
        r = requests.get(DATA_URL, timeout=30)
        r.raise_for_status()
        if len(r.content) < 50 or b"<html" in r.content[:500].lower():
            raise ValueError("Response does not look like the requested CSV.")
        path.write_bytes(r.content)
    except Exception as exc:
        raise RuntimeError(f"Could not download the official CSV: {exc}\n"
                           "Download manually using your browser and use --data waterquality.csv") from exc


def run_analysis(data_path: str | Path, out_path: str | Path):
    out = Path(out_path)
    out.mkdir(parents=True, exist_ok=True)
    df, audit = load_data(data_path)
    print("DATA AUDIT:", audit["source_rows"], "rows,", audit["complete_rows"], "complete rows")
    print(audit["cell_counts"].to_string(index=False))
    if audit["warnings"]:
        print("WARNINGS:", " | ".join(audit["warnings"]))
    audit["cell_counts"].to_csv(out / "group_counts.csv", index=False)
    audit_records = [dict(check="input_rows", value=audit["source_rows"]),
                     dict(check="complete_rows", value=audit["complete_rows"]),
                     dict(check="identical_rows_retained", value=audit["identical_rows_found_not_removed"])]
    audit_records += [dict(check="missing_"+str(k), value=v)
                      for k, v in audit["missing_values_after_conversion"].items()]
    pd.DataFrame(audit_records).to_csv(out / "data_audit.csv", index=False)
    descs = []
    for var in OUTCOMES:
        for group in [["season"], ["section"], ["season", "section"]]:
            z = describe(df, group, var)
            z.insert(0, "grouping", "+".join(group))
            descs.append(z)
    desc = pd.concat(descs, ignore_index=True)
    desc.to_csv(out / "descriptive_statistics.csv", index=False)
    seasons = season_tests(df)
    seasons.to_csv(out / "spring_vs_autumn.csv", index=False)
    sections, pairs, selection = section_tests(df)
    sections.to_csv(out / "section_omnibus.csv", index=False)
    pairs.to_csv(out / "section_posthoc.csv", index=False)
    selection.to_csv(out / "posthoc_methods.csv", index=False)
    factorial, diag, residuals = factorial_tests(df)
    factorial.to_csv(out / "factorial_anova.csv", index=False)
    diag.to_csv(out / "diagnostics.csv", index=False)
    interactions = interaction_posthoc(df, factorial)
    interactions.to_csv(out / "interaction_posthoc.csv", index=False)
    make_plots(df, residuals, out)
    write_summary(out, audit, seasons, sections, pairs, factorial, diag, interactions)
    print("\nSPRING VS AUTUMN (Welch; Holm across two outcomes):")
    print(seasons.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("\nSECTIONS (Welch ANOVA):")
    print(sections.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("\nFACTORIAL ANOVA:")
    print(factorial.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\nResults written to: {out.resolve()}")
    return {"data": df, "audit": audit, "descriptives": desc,
            "spring_vs_autumn": seasons, "section_omnibus": sections,
            "section_posthoc": pairs, "factorial_anova": factorial,
            "diagnostics": diag, "interaction_posthoc": interactions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="waterquality.csv", help="Path to official CSV")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--download", action="store_true", help="Download official dataset before analysis")
    args = parser.parse_args()
    if args.download:
        download_official(Path(args.data))
    run_analysis(args.data, args.out)


if __name__ == "__main__":
    main()
