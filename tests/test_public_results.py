from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def _one_row(frame, **conditions):
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    assert len(selected) == 1, (conditions, len(selected))
    return selected.iloc[0]


def test_primary_external_gene_result():
    metrics = pd.read_csv(TABLES / "trametinib_gdsc_gene_external_metrics.csv")
    row = _one_row(
        metrics,
        cohort="GDSC2_strict_sanger_heldout",
        outcome="LN_IC50",
        model="Elastic Net",
    )

    assert int(row["n"]) == 351
    assert np.isclose(row["pearson_r"], 0.5904729370630469)
    assert np.isclose(row["spearman_rho"], 0.6226481851481852)
    assert row["pearson_ci_2_5"] > 0
    assert row["spearman_ci_2_5"] > 0


def test_feature_and_leakage_audits():
    gene = pd.read_csv(TABLES / "trametinib_gdsc_gene_external_feature_audit.csv").iloc[0]
    pathway = pd.read_csv(
        TABLES / "trametinib_gdsc_pathway_external_feature_audit.csv"
    ).iloc[0]

    assert int(gene["prism_training_rows_with_expression"]) == 551
    assert int(gene["common_gene_symbols_used"]) == 19204
    assert int(gene["features_selected_inside_prism_training"]) == 1000
    assert int(gene["primary_strict_gdsc2_models"]) == 351
    assert int(gene["gdsc_outcomes_used_during_training"]) == 0

    assert int(pathway["pathway_member_gene_symbols_used"]) == 4374
    assert int(pathway["hallmark_pathways_used"]) == 50
    assert int(pathway["gdsc_outcomes_used_during_training"]) == 0


def test_primary_pathway_difference_and_overall_counts():
    comparison = pd.read_csv(
        TABLES / "trametinib_gdsc_gene_vs_pathway_external_comparison.csv"
    )
    row = _one_row(
        comparison,
        cohort="GDSC2_strict_sanger_heldout",
        outcome="LN_IC50",
        model="Elastic Net",
    )

    assert np.isclose(row["pathway_pearson_r"], 0.5124385069534468)
    assert np.isclose(row["pearson_difference_pathway_minus_gene"], -0.07803443247589026)
    assert row["pearson_difference_ci_97_5"] < 0
    assert row["pearson_conclusion"] == "individual genes favored"

    conclusions = pd.concat(
        [comparison["pearson_conclusion"], comparison["spearman_conclusion"]],
        ignore_index=True,
    )
    assert len(conclusions) == 32
    assert int((conclusions == "individual genes favored").sum()) == 18
    assert int((conclusions == "difference uncertain").sum()) == 14
    assert int(conclusions.str.contains("pathway", case=False).sum()) == 0


def test_gdsc1_replication_cohort_size():
    metrics = pd.read_csv(TABLES / "trametinib_gdsc_gene_external_metrics.csv")
    row = _one_row(
        metrics,
        cohort="GDSC1_strict_sanger_heldout",
        outcome="LN_IC50",
        model="Elastic Net",
    )
    assert int(row["n"]) == 319
    assert np.isclose(row["pearson_r"], 0.529918921112305)
