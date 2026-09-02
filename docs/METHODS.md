# Methods

## Objective and design

This study asked whether baseline cancer cell-line expression carries a signal that transfers between pharmacogenomic platforms for trametinib, and whether Hallmark pathway aggregation improves transfer relative to selected individual genes.

PRISM response and DepMap expression formed the training domain. GDSC response and Sanger-derived expression formed the external domain. The GDSC outcomes were not used for model fitting, preprocessing, feature selection, or hyperparameter selection.

## Training cohort construction

1. The PRISM replicate-collapsed response matrix was reshaped from treatment-by-cell-line format to one treatment-cell-line record per row.
2. Treatment metadata mapped the treatment key to the drug name.
3. Records were filtered to the exact case-insensitive name `trametinib`.
4. Multiple retained records for the same DepMap model, if present, were averaged to one response per `depmap_id`.
5. DepMap protein-coding RNA expression was loaded with `ModelID` standardized to `depmap_id`.
6. PRISM response and DepMap expression were inner-joined on `depmap_id`.
7. The final training cohort contained 551 models with an observed response and expression.

The model input matrix `X` contained numeric baseline expression values. The outcome vector `y` contained PRISM replicate-collapsed log-fold change. Identifiers and tissue labels were retained for matching and auditing but were not predictors.

## Internal validation

Random cell-line validation used five folds with shuffling and `random_state=42`. Each model received the same folds. Every cell line was predicted once by a model that had not trained on that row.

The complete preprocessing and model pipeline was fitted separately within each training fold:

- Median imputation.
- Univariate `f_regression` feature scoring.
- Selection of at most 1,000 genes.
- Standardization for Elastic Net.
- Model fitting.

This placement prevents held-out responses from influencing feature selection or scaling.

A stricter grouped sensitivity analysis used five `GroupKFold` splits defined by `primary_tissue`. Every test-fold tissue was absent from that fold's training set.

## Models

### Mean baseline

`DummyRegressor` predicted the training-fold mean response for every test row. It quantified performance available without expression information.

### Elastic Net

Elastic Net fitted a linear combination of selected standardized genes while penalizing coefficient size. `ElasticNetCV` tested `l1_ratio` values `0.1, 0.5, 0.9, 1.0` and 40 alpha values from `10^-3` to `10^1`, using inner three-fold cross-validation. The maximum iteration count was 50,000 and tolerance was `10^-3`.

### Random Forest

The Random Forest used 300 regression trees, `max_features="sqrt"`, a minimum of three samples per terminal leaf, parallel execution, and `random_state=42`. It was preceded by the same fold-specific imputation and 1,000-gene selection.

## Pathway representation

The human MSigDB Hallmark gene-symbol collection defined the pathways. Gene labels in DepMap were reduced from formats such as `BRAF (673)` to uppercase symbols. Pathways required at least 10 matched genes.

Within each training fold, the pathway transformer:

1. Estimated per-gene medians from training rows.
2. Imputed training and test rows with stored training medians.
3. Estimated per-gene means and standard deviations from training rows.
4. Converted expression to z-scores using those stored parameters.
5. Averaged member-gene z-scores to one score per pathway.

The reported pathway representation contained 50 Hallmark scores built from 4,374 unique matched genes.

## External-cohort construction

GDSC1 and GDSC2 fitted-dose-response workbooks were filtered to exact case-insensitive `Trametinib`. The response-expression join used `SANGER_MODEL_ID`. The Cell Model Passports RNA-seq matrix was transposed so one row represented one Sanger model and one column represented one gene.

The DepMap `Model.csv` crosswalk mapped PRISM/DepMap `ModelID` values to `SangerModelID`. Directly mapped training identities were excluded. Training identities without a Sanger mapping were additionally compared using normalized cell-line names; the audit found no additional matches.

The primary strict cohort required:

- GDSC2 response.
- A valid expression profile.
- No mapped identity seen in PRISM training.
- Sanger-derived expression.

This produced 351 GDSC2 models. The parallel GDSC1 strict cohort contained 319 models and was used as a cross-screen replication. Analyses allowing all expression sources were retained as sensitivity cohorts.

## Cross-platform gene alignment

DepMap and GDSC feature names were reduced to gene symbols. Duplicate GDSC rows for an overlapping symbol were averaged. A fixed set of 19,204 common gene symbols was placed in identical order in the training and external matrices.

For the gene analysis, the full pipeline was fitted on all 551 PRISM rows and selected 1,000 features using PRISM outcomes only. It then predicted GDSC expression rows.

For the pathway analysis, pathway medians, means, standard deviations, and model parameters were learned from the same 551 PRISM rows and applied unchanged to GDSC.

## Evaluation

Internal validation used Pearson correlation, Spearman rank correlation, root mean squared error, and mean absolute error because predictions and outcomes shared the PRISM scale.

External validation used Pearson and Spearman correlations. Cross-platform RMSE and MAE were not interpreted because PRISM log-fold change, GDSC `LN_IC50`, and GDSC `AUC` are different response quantities with different scales.

Nonparametric bootstrap sampling with replacement over external cell lines generated 95% intervals for model correlations. Paired bootstrap intervals compared pathway and gene correlations on identical rows. One thousand successful bootstrap samples were targeted for the external analyses.

## Reproducibility controls

- Identical row identifiers were excluded from strict external cohorts.
- GDSC outcomes used during training were audited as zero.
- Preprocessing was fitted only on training rows.
- Models and representations were evaluated on identical cohort rows.
- Split manifests and predictions were saved locally; aggregate feature audits and metric tables were retained for public reporting.
- Random seeds were fixed where supported.
