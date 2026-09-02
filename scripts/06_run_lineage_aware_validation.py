"""Compare random-fold and tissue-grouped validation for one PRISM drug.

Beginner summary
----------------
The original baseline randomly divided cell lines into five folds. This script
performs a stricter sensitivity analysis: all cell lines from the same
``primary_tissue`` are kept together, so a tissue in the test fold never appears
in that fold's training data. This asks whether a model transfers to cancer
tissue groups it did not see while training.

The script does not overwrite the original baseline. It reuses the saved
random-fold out-of-fold predictions for a fair comparison on the same
tissue-annotated samples, then fits new tissue-grouped models.
"""

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_expression, load_prism_primary
from src.modeling import merge_one_drug, safe_name


MODEL_ORDER = ["Mean baseline", "Elastic Net", "Random forest"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hold out complete primary-tissue groups and compare the result "
            "with the saved random-fold baseline."
        )
    )
    parser.add_argument(
        "--drug",
        required=True,
        help="Exact drug name previously used with 04_run_one_drug_baseline.py",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of grouped outer folds (default: 5)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used inside the models (default: 42)",
    )
    return parser.parse_args()


def make_models(n_features, random_state=42):
    """Build the same three model pipelines used in the original baseline."""
    k_features = min(1000, n_features)
    return {
        "Mean baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor()),
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
                        random_state=random_state,
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
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def safe_correlation(function, y_true, y_pred):
    """Return NaN when a correlation is mathematically undefined."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(function(y_true, y_pred).statistic)


def calculate_metrics(y_true, y_pred):
    """Calculate the four evaluation measures used throughout the project."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": len(y_true),
        "pearson_r": safe_correlation(pearsonr, y_true, y_pred),
        "spearman_rho": safe_correlation(spearmanr, y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def calculate_prediction_table_metrics(predictions):
    """Recalculate random-fold metrics after matching to annotated samples."""
    rows = []
    for model_name in MODEL_ORDER:
        if model_name not in predictions.columns:
            raise ValueError(
                f"Saved baseline predictions are missing the '{model_name}' column."
            )
        rows.append(
            {
                "model": model_name,
                **calculate_metrics(predictions["observed"], predictions[model_name]),
            }
        )
    return pd.DataFrame(rows)


def make_tissue_lookup(response):
    """Create one checked primary-tissue label for each DepMap cell line."""
    required = {"depmap_id", "primary_tissue"}
    missing = required - set(response.columns)
    if missing:
        raise ValueError(
            "PRISM cell-line metadata is missing primary_tissue. "
            "Confirm that primary-screen-cell-line-info.csv is the official file."
        )

    lookup = response[["depmap_id", "primary_tissue"]].copy()
    lookup["depmap_id"] = lookup["depmap_id"].astype(str).str.strip()
    lookup["primary_tissue"] = lookup["primary_tissue"].astype("string").str.strip()
    lookup.loc[
        lookup["primary_tissue"].str.casefold().isin(["", "nan", "none", "unknown"]),
        "primary_tissue",
    ] = pd.NA

    conflicts = (
        lookup.dropna(subset=["primary_tissue"])
        .groupby("depmap_id")["primary_tissue"]
        .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(
            f"Found {len(conflicts)} cell lines with conflicting primary-tissue labels. "
            "Resolve the metadata before grouped validation."
        )

    # Prefer an available label when duplicate rows include both a missing and
    # a non-missing tissue value.
    lookup["_tissue_missing"] = lookup["primary_tissue"].isna()
    lookup = (
        lookup.sort_values(["depmap_id", "_tissue_missing"])
        .drop_duplicates("depmap_id", keep="first")
        .drop(columns="_tissue_missing")
    )
    return lookup


def make_group_splits(x, y, groups, n_splits):
    """Build folds and verify that no tissue appears in both train and test."""
    unique_groups = pd.Series(groups).nunique()
    if n_splits < 2:
        raise ValueError("--n-splits must be at least 2.")
    if unique_groups < n_splits:
        raise ValueError(
            f"Only {unique_groups} tissue groups are available; "
            f"--n-splits cannot be {n_splits}."
        )

    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(x, y, groups=groups))
    fold_numbers = np.empty(len(y), dtype=int)
    summary_rows = []

    groups_array = np.asarray(groups, dtype=str)
    for fold_number, (train_index, test_index) in enumerate(splits, start=1):
        train_tissues = set(groups_array[train_index])
        test_tissues = set(groups_array[test_index])
        overlap = train_tissues.intersection(test_tissues)
        if overlap:
            raise RuntimeError(
                f"Tissue leakage detected in fold {fold_number}: {sorted(overlap)}"
            )
        fold_numbers[test_index] = fold_number
        summary_rows.append(
            {
                "lineage_fold": fold_number,
                "n_train_cell_lines": len(train_index),
                "n_test_cell_lines": len(test_index),
                "n_train_tissues": len(train_tissues),
                "n_held_out_tissues": len(test_tissues),
                "held_out_tissues": " | ".join(sorted(test_tissues)),
            }
        )

    return splits, fold_numbers, pd.DataFrame(summary_rows)


def evaluate_grouped_models(x, y, splits, fold_numbers, random_state=42):
    """Generate leakage-safe out-of-fold predictions for all three models."""
    models = make_models(x.shape[1], random_state=random_state)
    predictions = pd.DataFrame(
        {"lineage_fold": fold_numbers, "observed": np.asarray(y, dtype=float)}
    )
    metric_rows = []

    for model_name in MODEL_ORDER:
        model = models[model_name]
        out_of_fold = np.full(len(y), np.nan, dtype=float)
        print(f"  Fitting {model_name} across {len(splits)} grouped folds...")
        for train_index, test_index in splits:
            fitted = clone(model)
            fitted.fit(x.iloc[train_index], y.iloc[train_index])
            out_of_fold[test_index] = fitted.predict(x.iloc[test_index])
        if np.isnan(out_of_fold).any():
            raise RuntimeError(f"{model_name} did not predict every held-out row.")
        predictions[model_name] = out_of_fold
        metric_rows.append(
            {"model": model_name, **calculate_metrics(y, out_of_fold)}
        )

    return pd.DataFrame(metric_rows), predictions


def make_comparison(random_metrics, grouped_metrics, n_tissues, n_excluded):
    """Combine both validation schemes and calculate transparent differences."""
    random_table = random_metrics.copy()
    random_table.insert(0, "validation_scheme", "Random cell-line folds")
    grouped_table = grouped_metrics.copy()
    grouped_table.insert(0, "validation_scheme", "Held-out tissue folds")
    comparison = pd.concat([random_table, grouped_table], ignore_index=True)
    comparison["n_primary_tissues"] = n_tissues
    comparison["n_excluded_missing_tissue"] = n_excluded

    reference = random_metrics.set_index("model")
    comparison["pearson_difference_from_random"] = comparison.apply(
        lambda row: row["pearson_r"] - reference.loc[row["model"], "pearson_r"],
        axis=1,
    )
    comparison["rmse_difference_from_random"] = comparison.apply(
        lambda row: row["rmse"] - reference.loc[row["model"], "rmse"], axis=1
    )
    return comparison


def save_comparison_figure(comparison, path, drug_name):
    """Plot correlation and error for random versus held-out-tissue validation."""
    model_order = ["Elastic Net", "Random forest", "Mean baseline"]
    scheme_order = ["Random cell-line folds", "Held-out tissue folds"]
    colors = ["#4C78A8", "#F58518"]
    x_positions = np.arange(len(model_order))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for scheme_index, (scheme, color) in enumerate(zip(scheme_order, colors)):
        subset = comparison[comparison["validation_scheme"] == scheme].set_index(
            "model"
        )
        offset = (scheme_index - 0.5) * width
        pearson_values = [subset.loc[model, "pearson_r"] for model in model_order]
        rmse_values = [subset.loc[model, "rmse"] for model in model_order]
        axes[0].bar(
            x_positions + offset,
            pearson_values,
            width,
            label=scheme,
            color=color,
        )
        axes[1].bar(
            x_positions + offset,
            rmse_values,
            width,
            label=scheme,
            color=color,
        )

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Pearson correlation (higher is better)")
    axes[0].set_ylabel("Pearson r")
    axes[1].set_title("Prediction error (lower is better)")
    axes[1].set_ylabel("RMSE")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(model_order, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{drug_name}: random versus held-out-tissue validation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    prefix = safe_name(args.drug)
    expression_path = (
        ROOT / "data/raw/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    response_path = (
        ROOT
        / "data/raw/prism/primary-screen-replicate-collapsed-logfold-change.csv"
    )
    treatment_path = (
        ROOT
        / "data/raw/prism/primary-screen-replicate-collapsed-treatment-info.csv"
    )
    cells_path = ROOT / "data/raw/prism/primary-screen-cell-line-info.csv"
    baseline_predictions_path = ROOT / f"results/tables/{prefix}_predictions.csv"

    required_paths = [
        expression_path,
        response_path,
        treatment_path,
        cells_path,
        baseline_predictions_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.relative_to(ROOT)}. "
                "Complete the real-data baseline before this validation step."
            )

    print("Loading PRISM response and tissue metadata...")
    response = load_prism_primary(response_path, treatment_path, cells_path)
    tissue_lookup = make_tissue_lookup(response)

    print("Loading DepMap expression. This may take several minutes...")
    expression = load_expression(expression_path)
    identifiers, x, y = merge_one_drug(expression, response, args.drug)
    identifiers = identifiers.reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    sample_metadata = identifiers.merge(
        tissue_lookup, on="depmap_id", how="left", validate="one_to_one"
    )
    has_tissue = sample_metadata["primary_tissue"].notna()
    n_excluded = int((~has_tissue).sum())
    sample_metadata = sample_metadata.loc[has_tissue].reset_index(drop=True)
    x = x.loc[has_tissue.to_numpy()].reset_index(drop=True)
    y = y.loc[has_tissue.to_numpy()].reset_index(drop=True)

    if len(y) < 40:
        raise ValueError(
            f"Only {len(y)} matched cell lines have primary-tissue labels; "
            "at least 40 are required."
        )

    groups = sample_metadata["primary_tissue"].astype(str)
    n_tissues = groups.nunique()
    print(
        f"Matched tissue-annotated cell lines: {len(y)}; "
        f"primary tissues: {n_tissues}; excluded missing tissue: {n_excluded}"
    )

    print("Loading the previously saved random-fold predictions...")
    saved_random = pd.read_csv(baseline_predictions_path)
    required_prediction_columns = {
        "depmap_id",
        "observed",
        "Mean baseline",
        "Elastic Net",
        "Random forest",
    }
    missing_columns = required_prediction_columns - set(saved_random.columns)
    if missing_columns:
        raise ValueError(
            "The saved baseline prediction file is missing columns: "
            f"{sorted(missing_columns)}. Rerun 04_run_one_drug_baseline.py."
        )
    saved_random["depmap_id"] = saved_random["depmap_id"].astype(str).str.strip()
    if saved_random["depmap_id"].duplicated().any():
        raise ValueError("Saved baseline predictions contain duplicate depmap_id rows.")

    random_matched = sample_metadata[["depmap_id", "primary_tissue"]].merge(
        saved_random,
        on="depmap_id",
        how="left",
        validate="one_to_one",
    )
    if random_matched["observed"].isna().any():
        raise ValueError(
            "Some tissue-annotated cell lines are absent from the saved baseline "
            "predictions. Rerun 04_run_one_drug_baseline.py with the same drug."
        )
    if not np.allclose(
        random_matched["observed"].to_numpy(dtype=float),
        y.to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            "Observed responses do not match the saved baseline. The source data "
            "or drug may have changed; rerun 04_run_one_drug_baseline.py first."
        )
    random_metrics = calculate_prediction_table_metrics(random_matched)

    splits, fold_numbers, fold_summary = make_group_splits(
        x, y, groups, args.n_splits
    )
    print("Running held-out-tissue validation...")
    grouped_metrics, grouped_predictions = evaluate_grouped_models(
        x,
        y,
        splits,
        fold_numbers,
        random_state=args.random_state,
    )
    grouped_predictions.insert(0, "primary_tissue", groups.to_numpy())
    grouped_predictions.insert(0, "depmap_id", sample_metadata["depmap_id"].to_numpy())

    comparison = make_comparison(
        random_metrics,
        grouped_metrics,
        n_tissues=n_tissues,
        n_excluded=n_excluded,
    )
    lineage_metrics = comparison[
        comparison["validation_scheme"] == "Held-out tissue folds"
    ].copy()

    table_dir = ROOT / "results/tables"
    figure_dir = ROOT / "results/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = table_dir / f"{prefix}_lineage_metrics.csv"
    predictions_path = table_dir / f"{prefix}_lineage_predictions.csv"
    comparison_path = table_dir / f"{prefix}_validation_comparison.csv"
    folds_path = table_dir / f"{prefix}_lineage_fold_assignments.csv"
    figure_path = figure_dir / f"{prefix}_validation_comparison.png"

    lineage_metrics.to_csv(metrics_path, index=False)
    grouped_predictions.to_csv(predictions_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    fold_summary.to_csv(folds_path, index=False)
    save_comparison_figure(comparison, figure_path, args.drug)

    print("\nVALIDATION COMPARISON")
    display_columns = [
        "validation_scheme",
        "model",
        "n",
        "pearson_r",
        "spearman_rho",
        "rmse",
        "mae",
    ]
    print(comparison[display_columns].to_string(index=False))
    print("\nHELD-OUT TISSUES BY FOLD")
    print(
        fold_summary[
            [
                "lineage_fold",
                "n_test_cell_lines",
                "n_held_out_tissues",
                "held_out_tissues",
            ]
        ].to_string(index=False)
    )
    print("\nLINEAGE-AWARE VALIDATION PASSED")
    print("Created:")
    for path in [
        metrics_path,
        predictions_path,
        comparison_path,
        folds_path,
        figure_path,
    ]:
        print(path)
    print(
        "\nInterpretation reminder: weaker held-out-tissue performance is not a "
        "software failure. It suggests that tissue composition contributed to "
        "the random-fold result. This is an internal sensitivity analysis; an "
        "independent dataset such as GDSC is still needed for external validation."
    )


if __name__ == "__main__":
    main()
