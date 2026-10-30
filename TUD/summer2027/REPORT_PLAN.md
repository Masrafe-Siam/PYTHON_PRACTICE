# TU Dortmund M.Sc. Data Science Summer 2027 — Step 9 report plan

## Purpose
Only a planning and results package, **not** a submitted report. All main results have been recomputed from the official `waterquality.csv`, read with separator `;` and decimal `.`. Full report must be **exactly 10 pages** (including title, TOC and bibliography). The university asks for mathematical definitions, appropriate tests and referenced methods.

## Research questions and confirmed conclusions
1. **Spring vs autumn:** For oxygen, equal-weight, section-adjusted spring-minus-autumn mean difference = +0.151938 mg/L, individual 95% CI [-0.002322, 0.306197], two-sided p=0.053541, Holm p=0.053541: no statistically significant average difference detected. For nitrate, difference = +0.870662 mg/L, individual 95% CI [0.607196, 1.134129], two-sided p=1.46945e-10, Holm p=2.93889e-10: significantly higher in spring. CIs are individual, not Holm-adjusted simultaneous CIs.
2. **River-section differences:** Full 4-season × 5-section factorial model (n=45/cell), Type III ANOVA with sum coding. Oxygen section effect: F(4,880)=9.5194, p=1.54835e-07, partial eta^2=0.04148; oxygen interaction: F(12,880)=3.8967, p=7.45712e-06. Nitrate section: F(4,880)=544.8342, p=2.47680e-236, partial eta^2=0.71236; nitrate interaction: F(12,880)=0.8555, p=0.592764. Four seasons and five river sections form a balanced design.
3. **Which section pairs differ:** Oxygen: six significant out of 40 within-season comparisons (Holm over all 40): summer A1-A4, A1-A5, A2-A4, A2-A5, A3-A5; autumn A4-A5. Nitrate: all 10 between-section comparisons averaged equally over seasons significant using Tukey familywise adjustment; means increase from A1 to A5.
4. **Robustness:** HC3 robust covariance and a secondary outlier sensitivity check do not change the major conclusions. Do not drop flagged observations from primary results. Independence cannot be verified because the CSV lacks sampling dates and detailed site/repeated-measurement identifiers.

## Recommended exactly-10-page allocation
1. **Title page** — applicant name, title, date, TU Dortmund application report.
2. **Table of contents** — including section and bibliography page numbers.
3. **Introduction / motivation** — real-world importance, exact research questions, very brief headline findings, paper map.
4. **Problem and data description** — university-provided dataset; 900 rows, 5 sections, 4 seasons, 45 each cell; variable measurement scales; integrity checks; limits of provenance/independence.
5. **Methods I** — descriptive statistics, boxplots, factorial model, core assumptions and diagnostics; mathematics.
6. **Methods II** — planned contrasts, F-test, effect sizes, Holm and Tukey post-hoc methods, confidence intervals, robustness; mathematics.
7. **Evaluation I** — descriptive table, selected Figure 1, diagnostic summary and spring/autumn inferential table.
8. **Evaluation II** — selected Figure 2, compact ANOVA table and pairwise comparisons; nitrate all 10 can be summarized in text/matrix if space is tight, while full tables remain reproducible in source code.
9. **Summary and discussion** — direct answers to both university questions, effect magnitude, uncertainty, noncausality and independence limitation, possible future data collection.
10. **Bibliography** — all method, source and software citations used. Avoid appendix to comply with ten-page total.

## Main evidence / visual assets
- **Figure 1**: `figures/Figure_1_Spring_Autumn_Boxplots.png`. Caption: "Observed spring and autumn distributions of dissolved oxygen and nitrate concentrations pooled across five river sections (n=225 per season for each variable). Boxes represent the interquartile range and median; whiskers follow the 1.5×IQR rule."
- **Figure 2**: `figures/Figure_2_Season_Section_Mean_Profiles.png`. Caption: "Mean dissolved oxygen and nitrate concentrations by river section and season (n=45 observations per plotted mean). Profiles show the observed seasonal and spatial patterns; they do not themselves establish significance."
- **Table 1**: `results/report_table_1_descriptive.csv`. Combine seasonal and section means ± SD into compact display with n; optional median and IQR shown in full supplementary CSV files.
- **Table 2**: `results/report_table_2_spring_autumn.csv`. Planned spring/autumn contrasts, point estimates, 95% individual confidence intervals, two-outcome Holm-adjusted p values.
- **Table 3**: `results/report_table_3_anova.csv`. F, dfs, p, partial eta squared for season, section and interaction (two outcomes).
- **Pairwise results**: `results/report_table_4_oxygen_significant_pairs.csv` (six significant oxygen differences) and `results/report_table_5_nitrate_all_pairs.csv` (all ten nitrate pairs with Tukey simultaneous CIs). The full 40 oxygen comparisons are retained in `results/full_oxygen_pairwise_40.csv`, including nonsignificant results.

## Statistical methods and textual scope
- State two-sided hypotheses and alpha=0.05; tests performed separately for oxygen and nitrate.
- Distinguish the *section-averaged* spring/autumn hypothesis from section-specific differences. In the full factorial model, average the five within-section seasonal differences equally.
- For oxygen, interaction significance motivates comparisons of sections separately within each season, with Holm control over all 40 tests.
- For nitrate, compare section marginal means averaged over four seasons, with Tukey studentized-range adjustment over 10 pairwise tests.
- Identify the specific multiplicity families, and state the 95% spring/autumn CIs are unadjusted individual intervals, whereas nitrate post-hoc CIs are Tukey-adjusted simultaneous intervals.
- Keep the null of independent errors as a conditional assumption, not a confirmed fact. Avoid causal claims about pollution sources, temperature or upstream/downstream mechanics.
- Not every exploratory output or robustness check needs a separate graphic or table in the 10-page report; keep the result package for reproducibility.

## Reproducibility
Run `python step9_finalize_results.py --csv waterquality.csv --out results/step_9`. This reproduces all main numerical tables. `software_and_input_provenance.txt` stores exact package versions of the runtime, and the SHA256 hash of the file for audit. Check it again when running locally (your software versions may differ).
