"""Leakage-safe baseline modeling for one drug."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def safe_name(text):
    value = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return value or "drug"


def merge_one_drug(expression, response, drug_name):
    names = response["drug_name"].astype(str)
    exact = names.str.casefold() == str(drug_name).casefold()
    if exact.any():
        selected = response.loc[exact].copy()
    else:
        contains = names.str.contains(str(drug_name), case=False, regex=False, na=False)
        matches = sorted(names[contains].dropna().unique().tolist())
        if len(matches) != 1:
            preview = matches[:20]
            raise ValueError(
                f"Drug name was not an exact unique match. Matches: {preview}. "
                "Use the exact name from candidate_drugs.csv."
            )
        selected = response.loc[names == matches[0]].copy()

    outcome = (
        selected.groupby("depmap_id", as_index=False)
        .agg(response=("response", "mean"))
        .dropna(subset=["response"])
    )
    merged = outcome.merge(expression, on="depmap_id", how="inner")
    if len(merged) < 40:
        raise ValueError(
            f"Only {len(merged)} matched cell lines. Choose a better-covered drug."
        )

    gene_cols = [c for c in expression.columns if c != "depmap_id"]
    x = merged[gene_cols].copy()
    y = merged["response"].astype(float).copy()
    return merged[["depmap_id", "response"]], x, y


def _metrics(y_true, y_pred):
    pearson = pearsonr(y_true, y_pred).statistic
    spearman = spearmanr(y_true, y_pred).statistic
    return {
        "n": len(y_true),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def evaluate_baselines(x, y, random_state=42):
    if len(y) < 40:
        raise ValueError("At least 40 matched samples are required for this starter.")
    k_features = min(1000, x.shape[1])
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    models = {
        "Mean baseline": Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor())]
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

    metrics_rows = []
    predictions = pd.DataFrame({"observed": np.asarray(y)})
    for name, model in models.items():
        predicted = cross_val_predict(model, x, y, cv=outer_cv, n_jobs=1)
        row = {"model": name, **_metrics(y, predicted)}
        metrics_rows.append(row)
        predictions[name] = predicted

    metrics = pd.DataFrame(metrics_rows).sort_values("rmse")
    return metrics, predictions


def save_results(metrics, predictions, output_root, prefix):
    output_root = Path(output_root)
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = table_dir / f"{prefix}_metrics.csv"
    predictions_path = table_dir / f"{prefix}_predictions.csv"
    figure_path = figure_dir / f"{prefix}_observed_vs_predicted.png"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    model_names = [
        c for c in predictions.columns if c not in {"observed", "depmap_id"}
    ]
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4.2))
    if len(model_names) == 1:
        axes = [axes]
    numeric_predictions = predictions[["observed"] + model_names]
    low = float(numeric_predictions.min().min())
    high = float(numeric_predictions.max().max())
    for ax, name in zip(axes, model_names):
        ax.scatter(predictions["observed"], predictions[name], alpha=0.7, s=26)
        ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Observed response")
        ax.set_ylabel("Out-of-fold predicted response")
    fig.suptitle("Leakage-safe five-fold cross-validation", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return metrics_path, predictions_path, figure_path
