# Limitations and responsible interpretation

## Scientific scope

- This is a one-drug proof of concept. The gene-versus-pathway result cannot be generalized to all anticancer drugs.
- Experiments involve immortalized or established cancer cell lines, not patients.
- Bulk RNA expression averages over cell populations and cannot describe intratumoral single-cell heterogeneity.
- Cell-line systems omit important pharmacokinetic, immune, stromal, microenvironmental, and treatment-history effects.
- Association does not establish that selected genes cause drug sensitivity.

## Cross-dataset comparability

- PRISM log-fold change and GDSC `LN_IC50`/`AUC` are different measurements. External RMSE and MAE would mix incompatible scales; correlations are therefore emphasized.
- DepMap and Sanger expression measurements can differ because of laboratory, library-preparation, sequencing, normalization, and processing effects.
- GDSC1 and GDSC2 share many cell-line identities. GDSC1 supports cross-screen replication but is not a completely independent set of biological models relative to GDSC2.
- Of 551 PRISM training models, 106 lacked a direct Sanger identifier. Normalized-name auditing found no additional GDSC matches, but unresolved mapping uncertainty remains.

## Modeling limitations

- The sample size is small relative to the number of genes.
- Univariate feature selection can miss multivariable patterns and does not identify causal biomarkers.
- Elastic Net represents additive linear effects after preprocessing.
- Random Forest can model nonlinearities but can overfit high-dimensional, modest-sample data and shrink extreme predictions through averaging.
- Mean-z Hallmark scoring assumes that averaging member-gene expression is a useful representation. It can cancel opposing signals, ignore gene directionality and interaction structure, and discard drug-specific effects.
- Hyperparameter exploration was limited to the documented grids/settings.

## Interpretation rules

Appropriate claim:

> Baseline expression carried a reproducible signal associated with trametinib response across the tested cell-line datasets, but simple Hallmark pathway aggregation did not improve external generalization over selected individual genes.

Inappropriate claims include:

- “The model predicts whether a patient will respond.”
- “Pearson r=0.590 means 59% accuracy.”
- “The selected genes are trametinib biomarkers” without further validation.
- “Pathways are worse than genes for cancer drug response” based on one drug.
- “The model is clinically validated.”

## Needed next steps

1. Freeze the trametinib analysis and prespecify a small, mechanistically diverse drug panel.
2. Apply the same pipeline without drug-specific tuning after results are viewed.
3. Evaluate stability of selected genes and biological enrichment.
4. Add batch-aware or domain-adaptation sensitivity analyses.
5. Validate findings in additional experiments and, only if justified, patient-relevant datasets.
6. Obtain expert biological and statistical review before publication claims.

