"""Externally validate Hallmark pathway models and compare them with genes.

This script is the external-validation counterpart to scripts 07 and 11. It
uses the same 50 MSigDB Hallmark definitions and the same pathway-scoring
method used in the internal comparison. All pathway preprocessing and model
fitting are learned from the PRISM/DepMap training cohort. GDSC outcomes are
used only after predictions have been generated.

Because PRISM log-fold-change and GDSC LN_IC50/AUC use different measurement
scales, performance is compared with Pearson and Spearman correlations. Paired
bootstrap intervals quantify the pathway-minus-gene performance difference on
the exact same external cell lines.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


MODEL_ORDER = ["Mean baseline", "Elastic Net", "Random forest"]
LEARNED_MODELS = ["Elastic Net", "Random forest"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gene and Hallmark-pathway models in external GDSC data."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root; defaults to the parent of the scripts folder.",
    )
    parser.add_argument("--drug", default="trametinib")
    parser.add_argument(
        "--gene-sets",
        default="data/raw/msigdb/h.all.v2026.1.Hs.symbols.gmt",
        help="Hallmark Gene Symbols GMT file, relative to the project root.",
    )
    parser.add_argument("--minimum-matched-genes", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--pathway-support-script",
        type=Path,
        default=None,
        help="Advanced/testing option; defaults to sibling script 07.",
    )
    parser.add_argument(
        "--external-support-script",
        type=Path,
        default=None,
        help="Advanced/testing option; defaults to sibling script 11.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_support_module(path: Path, module_name: str):
    require_file(path)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load support script: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def cohort_definitions() -> list[dict]:
    return [
        {
            "cohort": "GDSC2_strict_sanger_heldout",
            "dataset": "GDSC2",
            "roles": {"strict_external_sanger_expression"},
        },
        {
            "cohort": "GDSC2_heldout_all_expression_sources",
            "dataset": "GDSC2",
            "roles": {
                "strict_external_sanger_expression",
                "heldout_cell_line_broad_expression",
            },
        },
        {
            "cohort": "GDSC1_strict_sanger_heldout",
            "dataset": "GDSC1",
            "roles": {"strict_external_sanger_expression"},
        },
        {
            "cohort": "GDSC1_heldout_all_expression_sources",
            "dataset": "GDSC1",
            "roles": {
                "strict_external_sanger_expression",
                "heldout_cell_line_broad_expression",
            },
        },
    ]


def prepare_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, low_memory=False)
    required = {
        "dataset",
        "SANGER_MODEL_ID",
        "CELL_LINE_NAME",
        "HAS_EXPRESSION",
        "EXPRESSION_DATA_SOURCE",
        "ANALYSIS_ROLE",
        "LN_IC50",
        "AUC",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{path.name} is missing {sorted(missing)}")
    manifest["SANGER_MODEL_ID"] = (
        manifest["SANGER_MODEL_ID"].astype(str).str.strip()
    )
    manifest["HAS_EXPRESSION"] = (
        manifest["HAS_EXPRESSION"].astype(str).str.lower().eq("true")
    )
    manifest["LN_IC50"] = pd.to_numeric(manifest["LN_IC50"], errors="coerce")
    manifest["AUC"] = pd.to_numeric(manifest["AUC"], errors="coerce")
    return manifest


def selected_external_ids(manifest: pd.DataFrame, definitions: list[dict]) -> set[str]:
    selected = set()
    for definition in definitions:
        mask = (
            manifest["dataset"].eq(definition["dataset"])
            & manifest["HAS_EXPRESSION"]
            & manifest["ANALYSIS_ROLE"].isin(definition["roles"])
        )
        selected.update(manifest.loc[mask, "SANGER_MODEL_ID"])
    return selected


def make_cohort_predictions(
    manifest: pd.DataFrame,
    definitions: list[dict],
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for definition in definitions:
        mask = (
            manifest["dataset"].eq(definition["dataset"])
            & manifest["HAS_EXPRESSION"]
            & manifest["ANALYSIS_ROLE"].isin(definition["roles"])
        )
        cohort = manifest.loc[mask].copy()
        cohort = cohort[cohort["SANGER_MODEL_ID"].isin(all_predictions.index)]
        cohort = cohort.drop_duplicates("SANGER_MODEL_ID", keep="first")
        cohort = cohort.set_index("SANGER_MODEL_ID")
        predictions = all_predictions.reindex(cohort.index)
        output = cohort.reset_index()[
            [
                "SANGER_MODEL_ID",
                "CELL_LINE_NAME",
                "EXPRESSION_DATA_SOURCE",
                "LN_IC50",
                "AUC",
            ]
        ].copy()
        output.insert(0, "dataset", definition["dataset"])
        output.insert(0, "cohort", definition["cohort"])
        for model_name in MODEL_ORDER:
            output[model_name] = predictions[model_name].to_numpy()
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def calculate_external_metrics(
    predictions: pd.DataFrame,
    external_support,
    bootstrap_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []
    seed_counter = 0
    for (cohort, dataset), group in predictions.groupby(
        ["cohort", "dataset"], sort=False
    ):
        for outcome in ("LN_IC50", "AUC"):
            for model_name in MODEL_ORDER:
                estimates = external_support.correlation_metrics(
                    group[outcome], group[model_name]
                )
                intervals = external_support.bootstrap_correlation_intervals(
                    group[outcome],
                    group[model_name],
                    bootstrap_repeats,
                    random_state + seed_counter,
                )
                seed_counter += 1
                rows.append(
                    {
                        "cohort": cohort,
                        "dataset": dataset,
                        "outcome": outcome,
                        "representation": "Hallmark pathways",
                        "model": model_name,
                        **estimates,
                        **intervals,
                        "metric_note": (
                            "Correlations only; PRISM and GDSC outcomes use "
                            "different measurement scales."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_bootstrap_differences(
    observed,
    gene_predicted,
    pathway_predicted,
    repeats: int,
    random_state: int,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    gene_predicted = np.asarray(gene_predicted, dtype=float)
    pathway_predicted = np.asarray(pathway_predicted, dtype=float)
    valid = (
        np.isfinite(observed)
        & np.isfinite(gene_predicted)
        & np.isfinite(pathway_predicted)
    )
    observed = observed[valid]
    gene_predicted = gene_predicted[valid]
    pathway_predicted = pathway_predicted[valid]
    rng = np.random.default_rng(random_state)
    pearson_differences = []
    spearman_differences = []
    for _ in range(repeats):
        index = rng.integers(0, len(observed), size=len(observed))
        sample_observed = observed[index]
        sample_gene = gene_predicted[index]
        sample_pathway = pathway_predicted[index]
        if (
            np.unique(sample_observed).size < 2
            or np.unique(sample_gene).size < 2
            or np.unique(sample_pathway).size < 2
        ):
            continue
        pearson_differences.append(
            float(pearsonr(sample_observed, sample_pathway).statistic)
            - float(pearsonr(sample_observed, sample_gene).statistic)
        )
        spearman_differences.append(
            float(spearmanr(sample_observed, sample_pathway).statistic)
            - float(spearmanr(sample_observed, sample_gene).statistic)
        )
    if not pearson_differences:
        return {
            "pearson_difference_ci_2_5": np.nan,
            "pearson_difference_ci_97_5": np.nan,
            "spearman_difference_ci_2_5": np.nan,
            "spearman_difference_ci_97_5": np.nan,
            "successful_paired_bootstraps": 0,
        }
    return {
        "pearson_difference_ci_2_5": float(
            np.quantile(pearson_differences, 0.025)
        ),
        "pearson_difference_ci_97_5": float(
            np.quantile(pearson_differences, 0.975)
        ),
        "spearman_difference_ci_2_5": float(
            np.quantile(spearman_differences, 0.025)
        ),
        "spearman_difference_ci_97_5": float(
            np.quantile(spearman_differences, 0.975)
        ),
        "successful_paired_bootstraps": len(pearson_differences),
    }


def interval_conclusion(lower: float, upper: float) -> str:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return "not estimable"
    if lower > 0:
        return "pathway favored"
    if upper < 0:
        return "individual genes favored"
    return "difference uncertain"


def compare_with_gene_predictions(
    pathway_predictions: pd.DataFrame,
    gene_predictions_path: Path,
    external_support,
    bootstrap_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    gene = pd.read_csv(gene_predictions_path, low_memory=False)
    keys = ["cohort", "dataset", "SANGER_MODEL_ID"]
    required = {*keys, "LN_IC50", "AUC", *LEARNED_MODELS}
    missing = required.difference(gene.columns)
    if missing:
        raise ValueError(f"{gene_predictions_path.name} is missing {sorted(missing)}")
    if gene.duplicated(keys).any():
        raise ValueError("Gene external predictions contain duplicate cohort/model IDs.")

    gene_keep = gene[keys + LEARNED_MODELS].rename(
        columns={model: f"gene_{model}" for model in LEARNED_MODELS}
    )
    pathway = pathway_predictions.rename(
        columns={model: f"pathway_{model}" for model in LEARNED_MODELS}
    )
    combined = pathway.merge(gene_keep, on=keys, how="inner", validate="one_to_one")
    if len(combined) != len(pathway_predictions):
        raise ValueError(
            "Gene and pathway external predictions did not align on every cohort row."
        )

    rows = []
    seed_counter = 0
    for (cohort, dataset), group in combined.groupby(keys[:2], sort=False):
        for outcome in ("LN_IC50", "AUC"):
            for model in LEARNED_MODELS:
                gene_metrics = external_support.correlation_metrics(
                    group[outcome], group[f"gene_{model}"]
                )
                pathway_metrics = external_support.correlation_metrics(
                    group[outcome], group[f"pathway_{model}"]
                )
                paired = paired_bootstrap_differences(
                    group[outcome],
                    group[f"gene_{model}"],
                    group[f"pathway_{model}"],
                    bootstrap_repeats,
                    random_state + seed_counter,
                )
                seed_counter += 1
                pearson_difference = (
                    pathway_metrics["pearson_r"] - gene_metrics["pearson_r"]
                )
                spearman_difference = (
                    pathway_metrics["spearman_rho"] - gene_metrics["spearman_rho"]
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "dataset": dataset,
                        "outcome": outcome,
                        "model": model,
                        "n": gene_metrics["n"],
                        "gene_pearson_r": gene_metrics["pearson_r"],
                        "pathway_pearson_r": pathway_metrics["pearson_r"],
                        "pearson_difference_pathway_minus_gene": pearson_difference,
                        "pearson_difference_ci_2_5": paired[
                            "pearson_difference_ci_2_5"
                        ],
                        "pearson_difference_ci_97_5": paired[
                            "pearson_difference_ci_97_5"
                        ],
                        "pearson_conclusion": interval_conclusion(
                            paired["pearson_difference_ci_2_5"],
                            paired["pearson_difference_ci_97_5"],
                        ),
                        "gene_spearman_rho": gene_metrics["spearman_rho"],
                        "pathway_spearman_rho": pathway_metrics["spearman_rho"],
                        "spearman_difference_pathway_minus_gene": spearman_difference,
                        "spearman_difference_ci_2_5": paired[
                            "spearman_difference_ci_2_5"
                        ],
                        "spearman_difference_ci_97_5": paired[
                            "spearman_difference_ci_97_5"
                        ],
                        "spearman_conclusion": interval_conclusion(
                            paired["spearman_difference_ci_2_5"],
                            paired["spearman_difference_ci_97_5"],
                        ),
                        "successful_paired_bootstraps": paired[
                            "successful_paired_bootstraps"
                        ],
                    }
                )
    return pd.DataFrame(rows)


def make_primary_comparison_figure(comparison: pd.DataFrame, path: Path) -> None:
    primary = comparison[
        comparison["cohort"].eq("GDSC2_strict_sanger_heldout")
        & comparison["outcome"].eq("LN_IC50")
    ].set_index("model")
    x = np.arange(len(LEARNED_MODELS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for axis, metric, title in [
        (axes[0], "pearson", "Pearson correlation"),
        (axes[1], "spearman", "Spearman correlation"),
    ]:
        gene_values = [
            primary.loc[model, f"gene_{metric}_r" if metric == "pearson" else "gene_spearman_rho"]
            for model in LEARNED_MODELS
        ]
        pathway_values = [
            primary.loc[
                model,
                f"pathway_{metric}_r" if metric == "pearson" else "pathway_spearman_rho",
            ]
            for model in LEARNED_MODELS
        ]
        axis.bar(x - width / 2, gene_values, width, label="Individual genes")
        axis.bar(x + width / 2, pathway_values, width, label="Hallmark pathways")
        axis.set_xticks(x)
        axis.set_xticklabels(LEARNED_MODELS)
        axis.set_ylim(0, max(gene_values + pathway_values) * 1.25)
        axis.set_ylabel("Correlation (higher is better)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("GDSC2 strict external validation: genes versus pathways")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arguments = parse_arguments()
    project = arguments.project_root.resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    script_folder = Path(__file__).resolve().parent
    pathway_script = arguments.pathway_support_script or (
        script_folder / "07_run_pathway_comparison.py"
    )
    external_script = arguments.external_support_script or (
        script_folder / "11_run_gdsc_gene_external_validation.py"
    )
    pathway_support = load_support_module(
        pathway_script.resolve(), "pathway_comparison_support"
    )
    external_support = load_support_module(
        external_script.resolve(), "external_validation_support"
    )

    prefix = external_support.safe_name(arguments.drug)
    table_folder = project / "results" / "tables"
    figure_folder = project / "results" / "figures"
    table_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)

    depmap_expression_path = (
        project
        / "data"
        / "raw"
        / "depmap"
        / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    gdsc_expression_path = (
        project
        / "data"
        / "raw"
        / "gdsc"
        / "rnaseq_merged_rsem_tpm_20260323.csv"
    )
    gene_set_path = Path(arguments.gene_sets)
    if not gene_set_path.is_absolute():
        gene_set_path = project / gene_set_path
    training_predictions_path = table_folder / f"{prefix}_predictions.csv"
    split_manifest_path = table_folder / "gdsc_external_validation_split_manifest.csv"
    gene_external_predictions_path = (
        table_folder / f"{prefix}_gdsc_gene_external_predictions.csv"
    )
    for path in (
        depmap_expression_path,
        gdsc_expression_path,
        gene_set_path,
        training_predictions_path,
        split_manifest_path,
        gene_external_predictions_path,
    ):
        require_file(path)

    training_predictions = pd.read_csv(training_predictions_path, low_memory=False)
    required_training = {"depmap_id", "observed"}
    missing_training = required_training.difference(training_predictions.columns)
    if missing_training:
        raise ValueError(
            f"{training_predictions_path.name} is missing {sorted(missing_training)}"
        )
    training_predictions["depmap_id"] = (
        training_predictions["depmap_id"].astype(str).str.strip()
    )
    manifest = prepare_manifest(split_manifest_path)
    definitions = cohort_definitions()
    selected_ids = selected_external_ids(manifest, definitions)

    print(f"Reading Hallmark gene sets from {gene_set_path.name}...")
    gene_sets = pathway_support.read_gmt(gene_set_path)
    print(f"Hallmark gene sets found: {len(gene_sets)}")
    x_train, y_train, training_audit = external_support.load_training_expression(
        depmap_expression_path, training_predictions
    )
    x_gdsc, gdsc_audit = external_support.load_gdsc_expression(
        gdsc_expression_path, selected_ids, set(x_train.columns)
    )
    common_genes = sorted(set(x_train.columns).intersection(x_gdsc.columns))
    matching = pathway_support.match_gene_sets(
        common_genes,
        gene_sets,
        minimum_matched_genes=arguments.minimum_matched_genes,
    )
    pathway_gene_names = [common_genes[index] for index in matching["union_columns"]]
    x_train = x_train[pathway_gene_names]
    x_gdsc = x_gdsc[pathway_gene_names]
    print(
        f"Training on {len(y_train):,} PRISM models, "
        f"{len(pathway_gene_names):,} pathway-member genes, and "
        f"{len(matching['pathway_names'])} pathways."
    )

    models = pathway_support.make_models(
        matching["pathway_indices"], random_state=arguments.random_state
    )
    all_predictions = pd.DataFrame(index=x_gdsc.index)
    for model_name in MODEL_ORDER:
        print(f"Fitting pathway {model_name}...")
        models[model_name].fit(x_train, y_train)
        all_predictions[model_name] = models[model_name].predict(x_gdsc)

    pathway_predictions = make_cohort_predictions(
        manifest, definitions, all_predictions
    )
    pathway_metrics = calculate_external_metrics(
        pathway_predictions,
        external_support,
        arguments.bootstrap_repeats,
        arguments.random_state,
    )
    comparison = compare_with_gene_predictions(
        pathway_predictions,
        gene_external_predictions_path,
        external_support,
        arguments.bootstrap_repeats,
        arguments.random_state + 1000,
    )

    feature_audit = pd.DataFrame(
        [
            {
                **training_audit,
                **gdsc_audit,
                "common_gene_symbols_available": len(common_genes),
                "pathway_member_gene_symbols_used": len(pathway_gene_names),
                "hallmark_pathways_used": len(matching["pathway_names"]),
                "minimum_matched_genes": arguments.minimum_matched_genes,
                "gene_set_file": gene_set_path.name,
                "gdsc_outcomes_used_during_training": 0,
            }
        ]
    )

    metrics_path = table_folder / f"{prefix}_gdsc_pathway_external_metrics.csv"
    predictions_path = (
        table_folder / f"{prefix}_gdsc_pathway_external_predictions.csv"
    )
    coverage_path = (
        table_folder / f"{prefix}_gdsc_pathway_gene_coverage.csv"
    )
    audit_path = (
        table_folder / f"{prefix}_gdsc_pathway_external_feature_audit.csv"
    )
    comparison_path = (
        table_folder / f"{prefix}_gdsc_gene_vs_pathway_external_comparison.csv"
    )
    figure_path = (
        figure_folder / f"{prefix}_gdsc_gene_vs_pathway_external_comparison.png"
    )
    pathway_metrics.to_csv(metrics_path, index=False)
    pathway_predictions.to_csv(predictions_path, index=False)
    matching["coverage"].to_csv(coverage_path, index=False)
    feature_audit.to_csv(audit_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    make_primary_comparison_figure(comparison, figure_path)

    primary = comparison[
        comparison["cohort"].eq("GDSC2_strict_sanger_heldout")
        & comparison["outcome"].eq("LN_IC50")
    ]
    display_columns = [
        "model",
        "n",
        "gene_pearson_r",
        "pathway_pearson_r",
        "pearson_difference_pathway_minus_gene",
        "pearson_difference_ci_2_5",
        "pearson_difference_ci_97_5",
        "pearson_conclusion",
        "gene_spearman_rho",
        "pathway_spearman_rho",
        "spearman_difference_pathway_minus_gene",
        "spearman_difference_ci_2_5",
        "spearman_difference_ci_97_5",
        "spearman_conclusion",
    ]
    print("\nPRIMARY GDSC2 STRICT GENE-VERSUS-PATHWAY COMPARISON")
    print(primary[display_columns].to_string(index=False))
    print("\nCreated:")
    for path in (
        metrics_path,
        predictions_path,
        coverage_path,
        audit_path,
        comparison_path,
        figure_path,
    ):
        print(path)
    print("\nGDSC GENE-VERSUS-PATHWAY EXTERNAL COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
