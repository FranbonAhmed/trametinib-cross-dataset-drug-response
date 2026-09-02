"""Audit out-of-fold predictions from the one-drug baseline.

This script does not refit a model. It checks prediction-file integrity and
summarizes fold stability, bootstrap uncertainty, residuals, response tails,
and tissue-specific performance using the already saved out-of-fold results.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import safe_name


MODEL_NAMES = ["Mean baseline", "Elastic Net", "Random forest"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drug",
        required=True,
        help="Exact drug name used for the baseline analysis.",
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=2000,
        help="Number of paired nonparametric bootstrap samples (default: 2000).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _correlation(function, observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(observed) < 3 or np.unique(observed).size < 2 or np.unique(predicted).size < 2:
        return np.nan
    return float(function(observed, predicted).statistic)


def calculate_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "n": int(len(observed)),
        "pearson_r": _correlation(pearsonr, observed, predicted),
        "spearman_rho": _correlation(spearmanr, observed, predicted),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "mean_error_pred_minus_obs": float(np.mean(predicted - observed)),
    }


def bootstrap_intervals(observed, predicted, repeats, random_state):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rng = np.random.default_rng(random_state)
    metric_names = ["pearson_r", "spearman_rho", "rmse", "mae"]
    estimates = calculate_metrics(observed, predicted)
    sampled = {name: [] for name in metric_names}

    for _ in range(repeats):
        index = rng.integers(0, len(observed), size=len(observed))
        values = calculate_metrics(observed[index], predicted[index])
        for name in metric_names:
            if np.isfinite(values[name]):
                sampled[name].append(values[name])

    rows = []
    for name in metric_names:
        values = np.asarray(sampled[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "estimate": estimates[name],
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "successful_bootstraps": int(len(values)),
            }
        )
    return rows


def find_tissue_metadata():
    path = ROOT / "data/raw/prism/primary-screen-cell-line-info.csv"
    if not path.exists():
        return None
    metadata = pd.read_csv(path, low_memory=False)
    id_candidates = ["depmap_id", "ModelID", "model_id", "DepMap_ID", "row_name"]
    id_column = next((column for column in id_candidates if column in metadata.columns), None)
    tissue_candidates = [
        "primary_tissue",
        "OncotreeLineage",
        "oncotree_lineage",
        "lineage",
    ]
    tissue_column = next(
        (column for column in tissue_candidates if column in metadata.columns), None
    )
    if id_column is None or tissue_column is None:
        return None
    metadata = metadata[[id_column, tissue_column]].rename(
        columns={id_column: "depmap_id", tissue_column: "primary_tissue"}
    )
    metadata["depmap_id"] = metadata["depmap_id"].astype(str).str.strip()
    metadata["primary_tissue"] = metadata["primary_tissue"].fillna("Unknown").astype(str)
    return metadata.drop_duplicates("depmap_id", keep="first")


def make_residual_figure(predictions, path):
    models = ["Elastic Net", "Random forest"]
    fig, axes = plt.subplots(len(models), 3, figsize=(14, 8))
    low = float(predictions[["observed"] + models].min().min())
    high = float(predictions[["observed"] + models].max().max())

    for row, model in enumerate(models):
        residual = predictions["observed"] - predictions[model]
        axes[row, 0].scatter(
            predictions["observed"],
            predictions[model],
            c=predictions["outer_fold"],
            cmap="viridis",
            alpha=0.7,
            s=25,
        )
        axes[row, 0].plot([low, high], [low, high], "--", color="gray", linewidth=1)
        axes[row, 0].set_title(f"{model}: observed vs predicted")
        axes[row, 0].set_xlabel("Observed response")
        axes[row, 0].set_ylabel("Out-of-fold prediction")

        axes[row, 1].scatter(predictions[model], residual, alpha=0.65, s=25)
        axes[row, 1].axhline(0, linestyle="--", color="gray", linewidth=1)
        axes[row, 1].set_title(f"{model}: residuals")
        axes[row, 1].set_xlabel("Predicted response")
        axes[row, 1].set_ylabel("Observed - predicted")

        axes[row, 2].hist(residual, bins=30, edgecolor="white")
        axes[row, 2].axvline(0, linestyle="--", color="gray", linewidth=1)
        axes[row, 2].set_title(f"{model}: residual distribution")
        axes[row, 2].set_xlabel("Observed - predicted")
        axes[row, 2].set_ylabel("Cell lines")

    fig.suptitle("One-drug baseline residual audit", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_fold_figure(fold_metrics, path):
    models = ["Elastic Net", "Random forest"]
    colors = {"Elastic Net": "tab:blue", "Random forest": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for model in models:
        selected = fold_metrics[fold_metrics["model"] == model].sort_values("outer_fold")
        axes[0].plot(
            selected["outer_fold"],
            selected["pearson_r"],
            marker="o",
            label=model,
            color=colors[model],
        )
        axes[1].plot(
            selected["outer_fold"],
            selected["rmse"],
            marker="o",
            label=model,
            color=colors[model],
        )
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].set_title("Pearson correlation by outer fold")
    axes[0].set_ylabel("Pearson r")
    axes[1].set_title("RMSE by outer fold")
    axes[1].set_ylabel("RMSE")
    for axis in axes:
        axis.set_xlabel("Outer fold")
        axis.set_xticks(sorted(fold_metrics["outer_fold"].unique()))
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    prefix = safe_name(args.drug)
    table_dir = ROOT / "results/tables"
    figure_dir = ROOT / "results/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = table_dir / f"{prefix}_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing {prediction_path.relative_to(ROOT)}")

    predictions = pd.read_csv(prediction_path)
    required = {"depmap_id", "outer_fold", "observed", *MODEL_NAMES}
    missing_columns = required - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            "Prediction file is missing columns "
            f"{sorted(missing_columns)}. Install the updated src/modeling.py and "
            "rerun the one-drug baseline before running this audit."
        )
    if predictions[list(required)].isna().any().any():
        raise ValueError("Prediction file contains missing required values.")
    if predictions["depmap_id"].duplicated().any():
        duplicates = int(predictions["depmap_id"].duplicated().sum())
        raise ValueError(f"Prediction file contains {duplicates} duplicate DepMap IDs.")
    folds = sorted(predictions["outer_fold"].astype(int).unique().tolist())
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected outer folds [1, 2, 3, 4, 5], found {folds}.")

    fold_rows = []
    for fold, group in predictions.groupby("outer_fold", sort=True):
        for model in MODEL_NAMES:
            fold_rows.append(
                {
                    "outer_fold": int(fold),
                    "model": model,
                    **calculate_metrics(group["observed"], group[model]),
                }
            )
    fold_metrics = pd.DataFrame(fold_rows)

    bootstrap_rows = []
    for model_number, model in enumerate(MODEL_NAMES):
        rows = bootstrap_intervals(
            predictions["observed"],
            predictions[model],
            args.bootstrap_repeats,
            args.random_state + model_number,
        )
        for row in rows:
            bootstrap_rows.append({"model": model, **row})
    bootstrap = pd.DataFrame(bootstrap_rows)

    lower_cutoff = float(predictions["observed"].quantile(0.10))
    upper_cutoff = float(predictions["observed"].quantile(0.90))
    tail_groups = {
        "lowest_response_decile": predictions["observed"] <= lower_cutoff,
        "middle_80_percent": predictions["observed"].between(
            lower_cutoff, upper_cutoff, inclusive="neither"
        ),
        "highest_response_decile": predictions["observed"] >= upper_cutoff,
    }
    tail_rows = []
    for group_name, mask in tail_groups.items():
        group = predictions.loc[mask]
        for model in MODEL_NAMES:
            tail_rows.append(
                {
                    "response_group": group_name,
                    "lower_cutoff": lower_cutoff,
                    "upper_cutoff": upper_cutoff,
                    "model": model,
                    **calculate_metrics(group["observed"], group[model]),
                }
            )
    tail_metrics = pd.DataFrame(tail_rows)

    audit_predictions = predictions.copy()
    for model in MODEL_NAMES:
        audit_predictions[f"residual_{safe_name(model)}"] = (
            audit_predictions["observed"] - audit_predictions[model]
        )
        audit_predictions[f"absolute_error_{safe_name(model)}"] = (
            audit_predictions[f"residual_{safe_name(model)}"].abs()
        )

    tissue_metrics = pd.DataFrame()
    metadata = find_tissue_metadata()
    if metadata is not None:
        audit_predictions = audit_predictions.merge(metadata, on="depmap_id", how="left")
        audit_predictions["primary_tissue"] = audit_predictions["primary_tissue"].fillna(
            "Unknown"
        )
        tissue_rows = []
        for tissue, group in audit_predictions.groupby("primary_tissue"):
            if len(group) < 15:
                continue
            for model in MODEL_NAMES:
                tissue_rows.append(
                    {
                        "primary_tissue": tissue,
                        "model": model,
                        **calculate_metrics(group["observed"], group[model]),
                    }
                )
        tissue_metrics = pd.DataFrame(tissue_rows)

    output_paths = {
        "fold metrics": table_dir / f"{prefix}_fold_metrics.csv",
        "bootstrap intervals": table_dir / f"{prefix}_bootstrap_intervals.csv",
        "response-tail metrics": table_dir / f"{prefix}_response_tail_metrics.csv",
        "audited predictions": table_dir / f"{prefix}_audited_predictions.csv",
        "tissue metrics": table_dir / f"{prefix}_tissue_metrics.csv",
        "residual figure": figure_dir / f"{prefix}_residual_diagnostics.png",
        "fold figure": figure_dir / f"{prefix}_fold_stability.png",
    }
    fold_metrics.to_csv(output_paths["fold metrics"], index=False)
    bootstrap.to_csv(output_paths["bootstrap intervals"], index=False)
    tail_metrics.to_csv(output_paths["response-tail metrics"], index=False)
    audit_predictions.to_csv(output_paths["audited predictions"], index=False)
    if not tissue_metrics.empty:
        tissue_metrics.to_csv(output_paths["tissue metrics"], index=False)
    make_residual_figure(predictions, output_paths["residual figure"])
    make_fold_figure(fold_metrics, output_paths["fold figure"])

    print("AUDIT PASSED")
    print(f"Rows: {len(predictions)}; unique DepMap IDs: {predictions['depmap_id'].nunique()}")
    print("Fold sizes:")
    print(predictions["outer_fold"].value_counts().sort_index().to_string())
    print("\nELASTIC NET FOLD METRICS")
    print(
        fold_metrics[fold_metrics["model"] == "Elastic Net"][
            ["outer_fold", "n", "pearson_r", "spearman_rho", "rmse", "mae"]
        ].to_string(index=False)
    )
    print("\nELASTIC NET 95% PAIRED BOOTSTRAP INTERVALS")
    print(bootstrap[bootstrap["model"] == "Elastic Net"].to_string(index=False))
    print("\nCreated:")
    for label, path in output_paths.items():
        if path.exists():
            print(f"{label}: {path}")
    if metadata is None:
        print("\nWARNING: Tissue metadata could not be identified; tissue metrics were skipped.")


if __name__ == "__main__":
    main()
