# TU Dortmund M.Sc. Data Science — Summer 2027

## Analysis (before writing the application report)

**Source:** official `waterquality.csv` from TU Dortmund:
https://statistik.tu-dortmund.de/storages/statistik/r/Downloads/Studium/Studiengaenge-Infos/Data_Science/waterquality.csv

The assignment expects **5 sections × 4 seasons × 45 measurements = 900 rows**.

### Run using VS Code or PowerShell

1. Download `waterquality.csv` from the official link above and save it in this folder. The source CSV is not included in this package because it could not be retrieved by the preparation environment.
2. Create and activate a virtual environment (optional):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python analyze_waterquality.py --data waterquality.csv --out results`

Alternatively, try downloading with the script itself:
`python analyze_waterquality.py --download --out results`

The `water_quality_analysis.ipynb` notebook executes the same reproducible pipeline and displays the actual results in VS Code/Jupyter.

### Files the analysis will produce

- `results_summary.md`: descriptive and inferential results, confidence intervals, interpretation cautions.
- `data_audit.csv`, `group_counts.csv`: missingness, dimensions, group sizes.
- `descriptive_statistics.csv`: mean, median, SD, quartiles, IQR, minima, maxima.
- `spring_vs_autumn.csv`: Welch tests, 95% confidence intervals, Hedges' g, Holm-adjusted p-values across the two outcomes.
- `section_omnibus.csv`: Welch ANOVA across the five sections (marginal differences across seasons).
- `section_posthoc.csv`: Games–Howell, variance-robust and compatible with Welch ANOVA; all 10 section pairs per outcome.
- `factorial_anova.csv`: season, section, and their interaction; partial eta squared.
- `interaction_posthoc.csv`: season-specific section contrasts, if interaction detected; Holm correction across each outcome's family of such contrasts.
- `diagnostics.csv`: residual normality (Shapiro–Wilk), Levene equal-variance tests, model R².
- Eight high-resolution PNG figures: boxplots by season/section, cell-mean interaction plots, residual Q–Q plots.

### Methodological decisions to discuss before finalizing

- These tests generally assume that the 45 readings within every section/season are independent samples. The assignment does not specify dates, temporal autocorrelation, paired units, or sampling procedure. Verify these assumptions if supporting metadata exist; if not, note this limitation.
- An omnibus difference in all four seasons is not the same as a spring-vs-autumn contrast: both are reported separately.
- Seasonal comparisons pool across river sections (balanced section sizes make the marginal comparison interpretable), while factorial ANOVA accounts explicitly for both season and section and tests an interaction.
- The section post-hoc tests target **marginal section differences across all seasons**. If the interaction is significant, season-specific pairwise analyses are added to avoid hiding effect variation across seasons.
- Levene testing is included only as a descriptive diagnostic: Games–Howell is used regardless of Levene p-value, avoiding an error-prone preliminary-test selection.
- Software is cited in the output for the future report; statistical books will need to be cited in the methods section later.

**No synthetic data, fabricated p-values or invented results are supplied.**
