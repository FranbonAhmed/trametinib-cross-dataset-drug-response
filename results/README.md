# Derived results

These files are analysis outputs, not raw source datasets.

## Figures

| File | Contents |
|---|---|
| `trametinib_gdsc2_strict_gene_external_validation.png` | Primary strict GDSC2 gene-model validation |
| `trametinib_gdsc_gene_vs_pathway_external_comparison.png` | External gene-versus-pathway comparisons |
| `trametinib_gene_vs_pathway_comparison.png` | Internal representation comparison |

## Central tables

| File | Contents |
|---|---|
| `trametinib_validation_comparison.csv` | Random-fold and held-out-tissue internal metrics |
| `trametinib_gdsc_gene_external_metrics.csv` | Gene-model external metrics and bootstrap intervals |
| `trametinib_gdsc_pathway_external_metrics.csv` | Pathway-model external metrics and bootstrap intervals |
| `trametinib_gdsc_gene_vs_pathway_external_comparison.csv` | Paired representation differences and conclusions |
| `trametinib_gdsc_gene_external_feature_audit.csv` | Gene alignment, training rows, selected features, and leakage audit |
| `trametinib_gdsc_pathway_external_feature_audit.csv` | Pathway coverage, training rows, and leakage audit |

The numbered scripts generate row-level prediction and split-manifest files during a local full-data run. Those files are intentionally ignored by Git because they contain provider-derived row-level outcomes. The public aggregate audits preserve the reported cohort counts and inclusion checks. None of these outputs should be interpreted as patient data or clinical predictions.

Provider-controlled raw files are excluded. Users reproducing the analysis must obtain them from the sources listed in [data/README.md](../data/README.md).
