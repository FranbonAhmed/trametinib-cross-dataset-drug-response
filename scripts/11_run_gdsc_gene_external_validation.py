"""Train PRISM gene-level models and externally validate them in GDSC.

Training uses the original DepMap expression and PRISM trametinib response.
Feature selection, imputation, scaling, and model fitting are learned only from
the PRISM training cohort. GDSC outcomes are used only for final evaluation.

Because PRISM log-fold-change and GDSC LN_IC50/AUC are different measurement
scales, external performance is evaluated with Pearson and Spearman
correlations rather than raw RMSE or MAE.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEP_ID_CANDIDATES = [
    "ModelID",
    "model_id",
    "DepMap_ID",
    "depmap_id",
    "row_name",
    "Unnamed: 0",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe gene-level PRISM-to-GDSC validation."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root; defaults to the parent of the scripts folder.",
    )
    parser.add_argument("--drug", default="trametinib")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def safe_name(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return value or "drug"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def find_depmap_id_column(columns) -> str:
    for candidate in DEP_ID_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not identify a DepMap model-ID column in {list(columns)}")


def depmap_gene_symbol(column: str) -> str:
    text = str(column).strip()
    match = re.fullmatch(r"(.+?)\s+\(\d+\)", text)
    return match.group(1).strip() if match else text


def load_training_expression(
    expression_path: Path, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    header = pd.read_csv(expression_path, nrows=0)
    id_column = find_depmap_id_column(header.columns)
    training_ids = set(predictions["depmap_id"].astype(str).str.strip())

    pieces = []
    print("Loading only PRISM training rows from DepMap expression...")
    for chunk in pd.read_csv(expression_path, chunksize=64, low_memory=False):
        ids = chunk[id_column].astype(str).str.strip()
        selected = chunk.loc[ids.isin(training_ids)].copy()
        if not selected.empty:
            selected[id_column] = ids.loc[selected.index]
            pieces.append(selected)
    if not pieces:
        raise ValueError("No PRISM training IDs matched the DepMap expression file.")

    expression = pd.concat(pieces, ignore_index=True)
    expression = expression.rename(columns={id_column: "depmap_id"})
    expression = expression.drop_duplicates("depmap_id", keep="first")
    merged = predictions[["depmap_id", "observed"]].merge(
        expression,
        on="depmap_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) < 40:
        raise ValueError(f"Only {len(merged)} PRISM training models matched expression.")

    raw_gene_columns = [
        column for column in expression.columns if column != "depmap_id"
    ]
    symbol_to_columns: dict[str, list[str]] = defaultdict(list)
    for column in raw_gene_columns:
        symbol = depmap_gene_symbol(column)
        if symbol:
            symbol_to_columns[symbol].append(column)

    gene_values = {}
    duplicate_symbols = 0
    for symbol, columns in symbol_to_columns.items():
        numeric = merged[columns].apply(pd.to_numeric, errors="coerce")
        if len(columns) == 1:
            gene_values[symbol] = numeric.iloc[:, 0].to_numpy(dtype="float32")
        else:
            duplicate_symbols += 1
            gene_values[symbol] = numeric.mean(axis=1).to_numpy(dtype="float32")

    x_train = pd.DataFrame(gene_values, index=merged["depmap_id"].astype(str))
    y_train = pd.Series(
        pd.to_numeric(merged["observed"], errors="coerce").to_numpy(),
        index=x_train.index,
        name="prism_logfold_change",
    )
    valid = y_train.notna()
    x_train = x_train.loc[valid]
    y_train = y_train.loc[valid]

    audit = {
        "prism_prediction_rows": len(predictions),
        "prism_training_rows_with_expression": len(x_train),
        "depmap_raw_gene_columns": len(raw_gene_columns),
        "depmap_unique_gene_symbols": len(symbol_to_columns),
        "depmap_duplicate_gene_symbols_averaged": duplicate_symbols,
    }
    return x_train, y_train, audit


def read_gdsc_model_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        model_id_row = next(reader)
        model_name_row = next(reader)
        source_row = next(reader)
        gene_header = next(reader)
    labels = [
        (model_id_row, "model_id"),
        (model_name_row, "model_name"),
        (source_row, "data_source"),
        (gene_header, "gene_symbol"),
    ]
    for row, expected in labels:
        if not row or row[0].strip() != expected:
            raise ValueError(
                f"Expected GDSC expression header label {expected}, found "
                f"{row[0] if row else '<empty>'}."
            )
    return [value.strip() for value in model_id_row[3:]]


def load_gdsc_expression(
    path: Path, selected_model_ids: set[str], training_genes: set[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    all_model_ids = read_gdsc_model_ids(path)
    available_selected_ids = [
        model_id for model_id in all_model_ids if model_id in selected_model_ids
    ]
    if not available_selected_ids:
        raise ValueError("None of the selected GDSC models were found in expression.")

    position_by_model = {
        model_id: position + 3 for position, model_id in enumerate(all_model_ids)
    }
    selected_positions = [0, 1, 2] + [
        position_by_model[model_id] for model_id in available_selected_ids
    ]

    print(
        f"Loading {len(available_selected_ids):,} selected GDSC expression columns..."
    )
    expression = pd.read_csv(
        path,
        skiprows=4,
        header=None,
        usecols=selected_positions,
        low_memory=False,
    )
    expression.columns = [
        "gene_symbol",
        "ensembl_gene_id",
        "sanger_gene_id",
        *available_selected_ids,
    ]
    expression["gene_symbol"] = expression["gene_symbol"].astype(str).str.strip()
    expression = expression[expression["gene_symbol"].isin(training_genes)].copy()
    raw_matching_rows = len(expression)
    duplicate_gene_rows = int(expression["gene_symbol"].duplicated(keep=False).sum())

    numeric = expression[available_selected_ids].apply(
        pd.to_numeric, errors="coerce"
    ).astype("float32")
    numeric.insert(0, "gene_symbol", expression["gene_symbol"].to_numpy())
    collapsed = numeric.groupby("gene_symbol", sort=False).mean()
    x_gdsc = collapsed.T
    x_gdsc.index.name = "SANGER_MODEL_ID"

    audit = {
        "gdsc_expression_models_loaded": len(available_selected_ids),
        "gdsc_rows_matching_depmap_gene_symbols_before_collapse": raw_matching_rows,
        "gdsc_duplicate_gene_rows_in_overlap": duplicate_gene_rows,
        "gdsc_unique_gene_symbols_in_overlap": len(collapsed),
    }
    return x_gdsc, audit


def correlation_metrics(observed, predicted) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 3 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return {
            "n": int(len(observed)),
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "prediction_sd": float(np.std(predicted)) if len(predicted) else np.nan,
        }
    return {
        "n": int(len(observed)),
        "pearson_r": float(pearsonr(observed, predicted).statistic),
        "spearman_rho": float(spearmanr(observed, predicted).statistic),
        "prediction_sd": float(np.std(predicted)),
    }


def bootstrap_correlation_intervals(
    observed, predicted, repeats: int, random_state: int
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    if len(observed) < 3 or np.unique(predicted).size < 2:
        return {
            "pearson_ci_2_5": np.nan,
            "pearson_ci_97_5": np.nan,
            "spearman_ci_2_5": np.nan,
            "spearman_ci_97_5": np.nan,
            "successful_bootstraps": 0,
        }

    rng = np.random.default_rng(random_state)
    pearson_values = []
    spearman_values = []
    for _ in range(repeats):
        index = rng.integers(0, len(observed), size=len(observed))
        sampled_observed = observed[index]
        sampled_predicted = predicted[index]
        if (
            np.unique(sampled_observed).size < 2
            or np.unique(sampled_predicted).size < 2
        ):
            continue
        pearson_values.append(
            float(pearsonr(sampled_observed, sampled_predicted).statistic)
        )
        spearman_values.append(
            float(spearmanr(sampled_observed, sampled_predicted).statistic)
        )

    if not pearson_values:
        return {
            "pearson_ci_2_5": np.nan,
            "pearson_ci_97_5": np.nan,
            "spearman_ci_2_5": np.nan,
            "spearman_ci_97_5": np.nan,
            "successful_bootstraps": 0,
        }
    return {
        "pearson_ci_2_5": float(np.quantile(pearson_values, 0.025)),
        "pearson_ci_97_5": float(np.quantile(pearson_values, 0.975)),
        "spearman_ci_2_5": float(np.quantile(spearman_values, 0.025)),
        "spearman_ci_97_5": float(np.quantile(spearman_values, 0.975)),
        "successful_bootstraps": len(pearson_values),
    }


def make_primary_figure(predictions: pd.DataFrame, path: Path) -> None:
    primary = predictions[
        predictions["cohort"] == "GDSC2_strict_sanger_heldout"
    ].copy()
    models = ["Elastic Net", "Random forest"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for axis, model in zip(axes, models):
        axis.scatter(
            primary[model],
            primary["LN_IC50"],
            alpha=0.7,
            s=27,
        )
        if len(primary) >= 2 and primary[model].nunique() > 1:
            slope, intercept = np.polyfit(primary[model], primary["LN_IC50"], 1)
            x_values = np.linspace(primary[model].min(), primary[model].max(), 100)
            axis.plot(x_values, slope * x_values + intercept, "--", color="gray")
        rho = spearmanr(primary[model], primary["LN_IC50"]).statistic
        axis.set_title(f"{model}\nSpearman rho = {rho:.3f}")
        axis.set_xlabel("Predicted PRISM log-fold-change")
        axis.set_ylabel("Observed GDSC2 LN_IC50")
    fig.suptitle("Strict external validation: held-out Sanger-expression models")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    arguments = parse_arguments()
    project = arguments.project_root.resolve()
    prefix = safe_name(arguments.drug)
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
    training_predictions_path = table_folder / f"{prefix}_predictions.csv"
    split_manifest_path = table_folder / "gdsc_external_validation_split_manifest.csv"

    for path in (
        depmap_expression_path,
        gdsc_expression_path,
        training_predictions_path,
        split_manifest_path,
    ):
        require_file(path)

    predictions = pd.read_csv(training_predictions_path, low_memory=False)
    required_prediction_columns = {"depmap_id", "observed"}
    missing_prediction_columns = required_prediction_columns.difference(
        predictions.columns
    )
    if missing_prediction_columns:
        raise ValueError(
            f"{training_predictions_path.name} is missing "
            f"{sorted(missing_prediction_columns)}"
        )
    predictions["depmap_id"] = predictions["depmap_id"].astype(str).str.strip()
    if predictions["depmap_id"].duplicated().any():
        raise ValueError("Training prediction file contains duplicate DepMap IDs.")

    manifest = pd.read_csv(split_manifest_path, low_memory=False)
    required_manifest_columns = {
        "dataset",
        "SANGER_MODEL_ID",
        "HAS_EXPRESSION",
        "ANALYSIS_ROLE",
        "LN_IC50",
        "AUC",
    }
    missing_manifest_columns = required_manifest_columns.difference(manifest.columns)
    if missing_manifest_columns:
        raise ValueError(
            f"{split_manifest_path.name} is missing {sorted(missing_manifest_columns)}"
        )
    manifest["SANGER_MODEL_ID"] = (
        manifest["SANGER_MODEL_ID"].astype(str).str.strip()
    )
    manifest["HAS_EXPRESSION"] = (
        manifest["HAS_EXPRESSION"].astype(str).str.lower().eq("true")
    )
    manifest["LN_IC50"] = pd.to_numeric(manifest["LN_IC50"], errors="coerce")
    manifest["AUC"] = pd.to_numeric(manifest["AUC"], errors="coerce")

    cohort_definitions = [
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

    selected_ids = set()
    for definition in cohort_definitions:
        mask = (
            manifest["dataset"].eq(definition["dataset"])
            & manifest["HAS_EXPRESSION"]
            & manifest["ANALYSIS_ROLE"].isin(definition["roles"])
        )
        selected_ids.update(manifest.loc[mask, "SANGER_MODEL_ID"])

    x_train, y_train, training_audit = load_training_expression(
        depmap_expression_path, predictions
    )
    x_gdsc, gdsc_audit = load_gdsc_expression(
        gdsc_expression_path, selected_ids, set(x_train.columns)
    )
    common_genes = sorted(set(x_train.columns).intersection(x_gdsc.columns))
    if len(common_genes) < 1000:
        raise ValueError(
            f"Only {len(common_genes)} common genes were found; expected at least 1000."
        )
    x_train = x_train[common_genes]
    x_gdsc = x_gdsc[common_genes]
    k_features = min(1000, len(common_genes))

    models = {
        "Mean baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "Elastic Net": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_regression, k=k_features)),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9, 1.0],
                        alphas=np.logspace(-3, 1, 40),
                        cv=3,
                        max_iter=50000,
                        tol=1e-3,
                        random_state=arguments.random_state,
                    ),
                ),
            ]
        ),
        "Random forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_regression, k=k_features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=arguments.random_state,
                    ),
                ),
            ]
        ),
    }

    all_predictions = pd.DataFrame(index=x_gdsc.index)
    print(
        f"Training on {len(y_train):,} PRISM models and {len(common_genes):,} "
        "common genes."
    )
    for model_name, model in models.items():
        print(f"Fitting {model_name}...")
        model.fit(x_train, y_train)
        all_predictions[model_name] = model.predict(x_gdsc)

    metric_rows = []
    prediction_rows = []
    seed_counter = 0
    for definition in cohort_definitions:
        mask = (
            manifest["dataset"].eq(definition["dataset"])
            & manifest["HAS_EXPRESSION"]
            & manifest["ANALYSIS_ROLE"].isin(definition["roles"])
        )
        cohort = manifest.loc[mask].copy()
        cohort = cohort[cohort["SANGER_MODEL_ID"].isin(x_gdsc.index)]
        cohort = cohort.drop_duplicates("SANGER_MODEL_ID", keep="first")
        cohort = cohort.set_index("SANGER_MODEL_ID")
        cohort_predictions = all_predictions.reindex(cohort.index)

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
        for model_name in models:
            output[model_name] = cohort_predictions[model_name].to_numpy()
        prediction_rows.append(output)

        for outcome in ("LN_IC50", "AUC"):
            observed = cohort[outcome]
            for model_name in models:
                predicted = cohort_predictions[model_name]
                estimates = correlation_metrics(observed, predicted)
                intervals = bootstrap_correlation_intervals(
                    observed,
                    predicted,
                    arguments.bootstrap_repeats,
                    arguments.random_state + seed_counter,
                )
                seed_counter += 1
                metric_rows.append(
                    {
                        "cohort": definition["cohort"],
                        "dataset": definition["dataset"],
                        "outcome": outcome,
                        "model": model_name,
                        **estimates,
                        **intervals,
                        "metric_note": (
                            "Correlations only; PRISM and GDSC outcomes use "
                            "different measurement scales."
                        ),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    external_predictions = pd.concat(prediction_rows, ignore_index=True)
    feature_audit = pd.DataFrame(
        [
            {
                **training_audit,
                **gdsc_audit,
                "common_gene_symbols_used": len(common_genes),
                "features_selected_inside_prism_training": k_features,
                "gdsc_outcomes_used_during_training": 0,
                "primary_strict_gdsc2_models": int(
                    (
                        manifest["dataset"].eq("GDSC2")
                        & manifest["ANALYSIS_ROLE"].eq(
                            "strict_external_sanger_expression"
                        )
                    ).sum()
                ),
            }
        ]
    )

    metrics_path = table_folder / f"{prefix}_gdsc_gene_external_metrics.csv"
    predictions_path = (
        table_folder / f"{prefix}_gdsc_gene_external_predictions.csv"
    )
    audit_path = table_folder / f"{prefix}_gdsc_gene_external_feature_audit.csv"
    figure_path = (
        figure_folder / f"{prefix}_gdsc2_strict_gene_external_validation.png"
    )

    metrics.to_csv(metrics_path, index=False)
    external_predictions.to_csv(predictions_path, index=False)
    feature_audit.to_csv(audit_path, index=False)
    make_primary_figure(external_predictions, figure_path)

    display_columns = [
        "cohort",
        "outcome",
        "model",
        "n",
        "pearson_r",
        "spearman_rho",
        "pearson_ci_2_5",
        "pearson_ci_97_5",
    ]
    print("\nGENE-LEVEL EXTERNAL VALIDATION METRICS")
    print(metrics[display_columns].to_string(index=False))
    print("\nCreated:")
    print(metrics_path)
    print(predictions_path)
    print(audit_path)
    print(figure_path)
    print("\nGDSC GENE-LEVEL EXTERNAL VALIDATION COMPLETED")


if __name__ == "__main__":
    main()
