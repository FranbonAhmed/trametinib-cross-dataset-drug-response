"""Compare individual-gene and Hallmark-pathway models for one PRISM drug.

This script is the next stage after scripts 04 and 06. It:

1. Reads the official MSigDB Hallmark gene sets in GMT format.
2. Matches HGNC gene symbols to the DepMap expression columns.
3. Converts thousands of genes into pathway scores inside each training fold.
4. Evaluates the same three models with the exact saved random and
   held-out-tissue folds.
5. Saves a direct gene-versus-pathway comparison without overwriting earlier
   results.

The pathway transformer learns medians, means, and standard deviations using
training rows only. Test rows therefore cannot influence feature construction.
"""

import argparse
from pathlib import Path
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_expression, load_prism_primary
from src.modeling import merge_one_drug, safe_name


MODEL_ORDER = ["Mean baseline", "Elastic Net", "Random forest"]
SCHEME_ORDER = ["Random cell-line folds", "Held-out tissue folds"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare individual genes with MSigDB Hallmark pathways."
    )
    parser.add_argument(
        "--drug",
        required=True,
        help="Exact drug name used in the previous baseline analyses",
    )
    parser.add_argument(
        "--gene-sets",
        default="data/raw/msigdb/h.all.v2026.1.Hs.symbols.gmt",
        help=(
            "Path to the MSigDB Hallmark gene-symbol GMT file, relative to the "
            "project root or as an absolute path"
        ),
    )
    parser.add_argument(
        "--minimum-matched-genes",
        type=int,
        default=10,
        help="Minimum matched genes required to retain a pathway (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Model seed (default: 42)",
    )
    return parser.parse_args()


def read_gmt(path):
    """Read a GMT file as {gene_set_name: [gene symbols]}."""
    path = Path(path)
    gene_sets = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n\r").split("\t")
            if not line.strip():
                continue
            if len(fields) < 3:
                raise ValueError(
                    f"Invalid GMT line {line_number}: expected a name, "
                    "description, and at least one gene."
                )
            name = fields[0].strip()
            if name in gene_sets:
                raise ValueError(f"Duplicate gene-set name in GMT file: {name}")
            genes = []
            seen = set()
            for value in fields[2:]:
                symbol = value.strip().upper()
                if symbol and symbol not in seen:
                    genes.append(symbol)
                    seen.add(symbol)
            gene_sets[name] = genes
    if not gene_sets:
        raise ValueError("The GMT file did not contain any gene sets.")
    return gene_sets


def expression_gene_symbol(column_name):
    """Convert a DepMap label such as 'BRAF (673)' to the symbol 'BRAF'."""
    value = re.sub(r"\s+\(\d+\)$", "", str(column_name).strip())
    return value.upper()


def match_gene_sets(expression_columns, gene_sets, minimum_matched_genes=10):
    """Match GMT symbols to expression columns and report pathway coverage."""
    if minimum_matched_genes < 1:
        raise ValueError("--minimum-matched-genes must be at least 1.")

    symbol_to_column = {}
    duplicate_symbols = set()
    for column_index, column_name in enumerate(expression_columns):
        symbol = expression_gene_symbol(column_name)
        if symbol in symbol_to_column:
            duplicate_symbols.add(symbol)
        else:
            symbol_to_column[symbol] = column_index

    original_indices = {}
    coverage_rows = []
    for pathway_name, genes in gene_sets.items():
        matched_symbols = [gene for gene in genes if gene in symbol_to_column]
        matched_indices = [symbol_to_column[gene] for gene in matched_symbols]
        included = len(matched_indices) >= minimum_matched_genes
        coverage_rows.append(
            {
                "pathway": pathway_name,
                "genes_in_gmt": len(genes),
                "matched_expression_genes": len(matched_indices),
                "coverage_fraction": (
                    len(matched_indices) / len(genes) if genes else np.nan
                ),
                "included_in_model": included,
            }
        )
        if included:
            original_indices[pathway_name] = matched_indices

    if len(original_indices) < 10:
        raise ValueError(
            f"Only {len(original_indices)} pathways passed the matching threshold. "
            "Confirm that you downloaded the human Gene Symbols GMT file, not "
            "the NCBI Gene IDs file."
        )

    union_columns = sorted(
        {index for indices in original_indices.values() for index in indices}
    )
    old_to_new = {old: new for new, old in enumerate(union_columns)}
    pathway_names = list(original_indices)
    pathway_indices = [
        np.asarray([old_to_new[index] for index in original_indices[name]], dtype=int)
        for name in pathway_names
    ]
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["included_in_model", "coverage_fraction", "pathway"],
        ascending=[False, False, True],
    )
    return {
        "union_columns": union_columns,
        "pathway_names": pathway_names,
        "pathway_indices": pathway_indices,
        "coverage": coverage,
        "duplicate_expression_symbols": sorted(duplicate_symbols),
    }


class PathwayScoreTransformer(BaseEstimator, TransformerMixin):
    """Create mean standardized-expression scores for fixed gene sets.

    Gene medians, means, and standard deviations are estimated in ``fit`` from
    the training samples only. Each pathway score is the mean standardized
    expression of its matched member genes.
    """

    def __init__(self, pathway_indices):
        self.pathway_indices = pathway_indices

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        if values.ndim != 2:
            raise ValueError("Expression input must be a two-dimensional matrix.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isnan(values), medians, values)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0, ddof=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)

        membership = np.zeros(
            (values.shape[1], len(self.pathway_indices)), dtype=np.float32
        )
        for pathway_column, indices in enumerate(self.pathway_indices):
            indices = np.asarray(indices, dtype=int)
            if len(indices) == 0:
                raise ValueError("A retained pathway has no matched genes.")
            if indices.min() < 0 or indices.max() >= values.shape[1]:
                raise ValueError("A pathway gene index is outside the expression matrix.")
            membership[indices, pathway_column] = 1.0 / len(indices)

        self.n_features_in_ = values.shape[1]
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.membership_ = membership
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Expression columns changed between fit and transform.")
        filled = np.where(np.isnan(values), self.medians_, values)
        standardized = (filled - self.means_) / self.scales_
        return standardized @ self.membership_


def make_models(pathway_indices, random_state=42):
    """Build models that perform pathway construction inside every fold."""
    def pathway_step():
        return PathwayScoreTransformer(pathway_indices=pathway_indices)

    return {
        "Mean baseline": Pipeline(
            [
                ("pathways", pathway_step()),
                ("model", DummyRegressor()),
            ]
        ),
        "Elastic Net": Pipeline(
            [
                ("pathways", pathway_step()),
                ("scale_pathways", StandardScaler()),
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
                ("pathways", pathway_step()),
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
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(function(y_true, y_pred).statistic)


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": len(y_true),
        "pearson_r": safe_correlation(pearsonr, y_true, y_pred),
        "spearman_rho": safe_correlation(spearmanr, y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def make_splits_from_saved_folds(fold_ids):
    """Reconstruct exact outer splits from previously saved fold numbers."""
    fold_series = pd.Series(fold_ids).reset_index(drop=True)
    if fold_series.isna().any():
        raise ValueError("Saved fold assignments contain missing values.")
    unique_folds = sorted(fold_series.unique().tolist())
    if len(unique_folds) < 2:
        raise ValueError("At least two saved folds are required.")
    splits = []
    fold_array = fold_series.to_numpy()
    for fold in unique_folds:
        test_index = np.flatnonzero(fold_array == fold)
        train_index = np.flatnonzero(fold_array != fold)
        if len(test_index) == 0 or len(train_index) == 0:
            raise ValueError(f"Saved fold {fold} has an empty train or test set.")
        splits.append((train_index, test_index))
    return splits


def align_saved_folds(identifiers, x, y, saved_path, fold_column):
    """Align saved fold assignments with freshly loaded expression rows."""
    saved = pd.read_csv(saved_path)
    required = {"depmap_id", "observed", fold_column}
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(f"{saved_path.name} is missing columns: {sorted(missing)}")
    if saved["depmap_id"].duplicated().any():
        raise ValueError(f"{saved_path.name} contains duplicate depmap_id rows.")

    source = identifiers[["depmap_id"]].reset_index(drop=True).copy()
    source["_expression_row"] = np.arange(len(source))
    source["depmap_id"] = source["depmap_id"].astype(str).str.strip()
    saved = saved[["depmap_id", "observed", fold_column]].copy()
    saved["depmap_id"] = saved["depmap_id"].astype(str).str.strip()
    aligned = source.merge(saved, on="depmap_id", how="inner", validate="one_to_one")
    if len(aligned) != len(saved):
        raise ValueError(
            f"Only {len(aligned)} of {len(saved)} saved prediction rows matched "
            "the current expression/response data. Do not continue after changing "
            "source releases without rerunning the earlier scripts."
        )

    row_positions = aligned["_expression_row"].to_numpy(dtype=int)
    aligned_x = x.iloc[row_positions].reset_index(drop=True)
    aligned_y = y.iloc[row_positions].reset_index(drop=True)
    if not np.allclose(
        aligned["observed"].to_numpy(dtype=float),
        aligned_y.to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-9,
    ):
        raise ValueError(
            f"Observed responses in {saved_path.name} do not match the current data."
        )
    return aligned, aligned_x, aligned_y


def evaluate_pathway_models(
    x,
    y,
    fold_ids,
    pathway_indices,
    validation_scheme,
    random_state=42,
):
    """Generate pathway-model out-of-fold predictions for saved splits."""
    splits = make_splits_from_saved_folds(fold_ids)
    models = make_models(pathway_indices, random_state=random_state)
    predictions = pd.DataFrame(
        {"outer_fold": np.asarray(fold_ids), "observed": np.asarray(y, dtype=float)}
    )
    metric_rows = []

    print(f"Running pathway models: {validation_scheme}")
    for model_name in MODEL_ORDER:
        print(f"  Fitting {model_name} across {len(splits)} folds...")
        out_of_fold = np.full(len(y), np.nan, dtype=float)
        for train_index, test_index in splits:
            model = clone(models[model_name])
            model.fit(x.iloc[train_index], y.iloc[train_index])
            out_of_fold[test_index] = model.predict(x.iloc[test_index])
        if np.isnan(out_of_fold).any():
            raise RuntimeError(f"{model_name} did not predict every held-out row.")
        predictions[model_name] = out_of_fold
        metric_rows.append(
            {
                "validation_scheme": validation_scheme,
                "representation": "Hallmark pathways",
                "model": model_name,
                **calculate_metrics(y, out_of_fold),
            }
        )
    return pd.DataFrame(metric_rows), predictions


def load_individual_gene_metrics(random_path, lineage_path):
    """Load previous gene-level metrics for the direct comparison table."""
    required = {"model", "n", "pearson_r", "spearman_rho", "rmse", "mae"}
    random_metrics = pd.read_csv(random_path)
    lineage_metrics = pd.read_csv(lineage_path)
    for path, table in [(random_path, random_metrics), (lineage_path, lineage_metrics)]:
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    random_metrics = random_metrics[list(required)].copy()
    random_metrics.insert(0, "representation", "Individual genes")
    random_metrics.insert(0, "validation_scheme", "Random cell-line folds")
    lineage_metrics = lineage_metrics[list(required)].copy()
    lineage_metrics.insert(0, "representation", "Individual genes")
    lineage_metrics.insert(0, "validation_scheme", "Held-out tissue folds")
    return pd.concat([random_metrics, lineage_metrics], ignore_index=True)


def add_differences_from_gene_models(comparison):
    """Add pathway-minus-gene differences for each model and validation scheme."""
    comparison = comparison.copy()
    gene_reference = (
        comparison[comparison["representation"] == "Individual genes"]
        .set_index(["validation_scheme", "model"])
    )

    def difference(row, metric):
        reference = gene_reference.loc[
            (row["validation_scheme"], row["model"]), metric
        ]
        return row[metric] - reference

    comparison["pearson_difference_pathway_minus_gene"] = comparison.apply(
        lambda row: difference(row, "pearson_r"), axis=1
    )
    comparison["rmse_difference_pathway_minus_gene"] = comparison.apply(
        lambda row: difference(row, "rmse"), axis=1
    )
    return comparison


def save_comparison_figure(comparison, path, drug_name):
    """Show gene-versus-pathway performance for both learned models."""
    model_order = ["Elastic Net", "Random forest"]
    representation_order = ["Individual genes", "Hallmark pathways"]
    colors = ["#4C78A8", "#54A24B"]
    x_positions = np.arange(len(SCHEME_ORDER))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex="col")
    for model_column, model_name in enumerate(model_order):
        model_table = comparison[comparison["model"] == model_name]
        for representation_index, (representation, color) in enumerate(
            zip(representation_order, colors)
        ):
            subset = model_table[
                model_table["representation"] == representation
            ].set_index("validation_scheme")
            offset = (representation_index - 0.5) * width
            pearson_values = [
                subset.loc[scheme, "pearson_r"] for scheme in SCHEME_ORDER
            ]
            rmse_values = [subset.loc[scheme, "rmse"] for scheme in SCHEME_ORDER]
            axes[0, model_column].bar(
                x_positions + offset,
                pearson_values,
                width,
                color=color,
                label=representation,
            )
            axes[1, model_column].bar(
                x_positions + offset,
                rmse_values,
                width,
                color=color,
                label=representation,
            )

        axes[0, model_column].axhline(0, color="black", linewidth=0.8)
        axes[0, model_column].set_title(model_name)
        axes[0, model_column].set_ylabel("Pearson r (higher is better)")
        axes[1, model_column].set_ylabel("RMSE (lower is better)")
        axes[1, model_column].set_xticks(x_positions)
        axes[1, model_column].set_xticklabels(
            ["Random folds", "Held-out tissues"], rotation=10, ha="right"
        )
        for row in range(2):
            axes[row, model_column].grid(axis="y", alpha=0.2)

    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{drug_name}: individual genes versus Hallmark pathways")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    prefix = safe_name(args.drug)
    gene_set_path = Path(args.gene_sets)
    if not gene_set_path.is_absolute():
        gene_set_path = ROOT / gene_set_path

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
    random_predictions_path = ROOT / f"results/tables/{prefix}_predictions.csv"
    lineage_predictions_path = (
        ROOT / f"results/tables/{prefix}_lineage_predictions.csv"
    )
    random_metrics_path = ROOT / f"results/tables/{prefix}_metrics.csv"
    lineage_metrics_path = ROOT / f"results/tables/{prefix}_lineage_metrics.csv"

    required_paths = [
        gene_set_path,
        expression_path,
        response_path,
        treatment_path,
        cells_path,
        random_predictions_path,
        lineage_predictions_path,
        random_metrics_path,
        lineage_metrics_path,
    ]
    for path in required_paths:
        if not path.exists():
            if path == gene_set_path:
                raise FileNotFoundError(
                    f"Missing {path}. Download the human MSigDB Hallmark "
                    "Gene Symbols GMT file and place it at this exact location."
                )
            raise FileNotFoundError(
                f"Missing {path.relative_to(ROOT)}. Complete scripts 04 and 06 first."
            )

    print(f"Reading Hallmark gene sets from {gene_set_path.name}...")
    gene_sets = read_gmt(gene_set_path)
    print(f"Gene sets found in GMT file: {len(gene_sets)}")

    print("Loading PRISM response metadata...")
    response = load_prism_primary(response_path, treatment_path, cells_path)
    print("Loading DepMap expression. This may take several minutes...")
    expression = load_expression(expression_path)
    identifiers, x, y = merge_one_drug(expression, response, args.drug)
    identifiers = identifiers.reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    matching = match_gene_sets(
        x.columns,
        gene_sets,
        minimum_matched_genes=args.minimum_matched_genes,
    )
    x = x.iloc[:, matching["union_columns"]].copy()
    n_included = len(matching["pathway_names"])
    print(
        f"Pathways retained: {n_included}; unique matched expression genes: "
        f"{x.shape[1]:,}"
    )
    if matching["duplicate_expression_symbols"]:
        print(
            "Warning: duplicate expression symbols detected; the first column was "
            f"used for {len(matching['duplicate_expression_symbols'])} symbols."
        )

    random_alignment, random_x, random_y = align_saved_folds(
        identifiers,
        x,
        y,
        random_predictions_path,
        "outer_fold",
    )
    lineage_alignment, lineage_x, lineage_y = align_saved_folds(
        identifiers,
        x,
        y,
        lineage_predictions_path,
        "lineage_fold",
    )

    random_pathway_metrics, random_pathway_predictions = evaluate_pathway_models(
        random_x,
        random_y,
        random_alignment["outer_fold"],
        matching["pathway_indices"],
        "Random cell-line folds",
        random_state=args.random_state,
    )
    lineage_pathway_metrics, lineage_pathway_predictions = evaluate_pathway_models(
        lineage_x,
        lineage_y,
        lineage_alignment["lineage_fold"],
        matching["pathway_indices"],
        "Held-out tissue folds",
        random_state=args.random_state,
    )

    random_pathway_predictions = random_pathway_predictions.rename(
        columns={"outer_fold": "random_outer_fold"}
    )
    random_pathway_predictions.insert(
        0, "depmap_id", random_alignment["depmap_id"].to_numpy()
    )
    lineage_pathway_predictions = lineage_pathway_predictions.rename(
        columns={"outer_fold": "lineage_fold"}
    )
    lineage_pathway_predictions.insert(
        0, "depmap_id", lineage_alignment["depmap_id"].to_numpy()
    )

    pathway_metrics = pd.concat(
        [random_pathway_metrics, lineage_pathway_metrics], ignore_index=True
    )
    pathway_metrics["gene_set_file"] = gene_set_path.name
    pathway_metrics["n_pathways"] = n_included

    gene_metrics = load_individual_gene_metrics(
        random_metrics_path, lineage_metrics_path
    )
    comparison_columns = [
        "validation_scheme",
        "representation",
        "model",
        "n",
        "pearson_r",
        "spearman_rho",
        "rmse",
        "mae",
    ]
    comparison = pd.concat(
        [gene_metrics[comparison_columns], pathway_metrics[comparison_columns]],
        ignore_index=True,
    )
    comparison = add_differences_from_gene_models(comparison)
    comparison["gene_set_file"] = np.where(
        comparison["representation"] == "Hallmark pathways",
        gene_set_path.name,
        "not applicable",
    )
    comparison["n_pathways"] = np.where(
        comparison["representation"] == "Hallmark pathways",
        n_included,
        np.nan,
    )

    table_dir = ROOT / "results/tables"
    figure_dir = ROOT / "results/figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pathway_metrics_path = table_dir / f"{prefix}_pathway_metrics.csv"
    random_pathway_predictions_path = (
        table_dir / f"{prefix}_pathway_random_predictions.csv"
    )
    lineage_pathway_predictions_path = (
        table_dir / f"{prefix}_pathway_lineage_predictions.csv"
    )
    coverage_path = table_dir / f"{prefix}_pathway_gene_coverage.csv"
    comparison_path = table_dir / f"{prefix}_gene_vs_pathway_comparison.csv"
    figure_path = figure_dir / f"{prefix}_gene_vs_pathway_comparison.png"

    pathway_metrics.to_csv(pathway_metrics_path, index=False)
    random_pathway_predictions.to_csv(random_pathway_predictions_path, index=False)
    lineage_pathway_predictions.to_csv(lineage_pathway_predictions_path, index=False)
    matching["coverage"].to_csv(coverage_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    save_comparison_figure(comparison, figure_path, args.drug)

    print("\nGENE-VERSUS-PATHWAY COMPARISON")
    display = comparison[
        comparison["model"].isin(["Elastic Net", "Random forest"])
    ][
        [
            "validation_scheme",
            "representation",
            "model",
            "n",
            "pearson_r",
            "spearman_rho",
            "rmse",
            "mae",
            "pearson_difference_pathway_minus_gene",
            "rmse_difference_pathway_minus_gene",
        ]
    ]
    print(display.to_string(index=False))
    print("\nPATHWAY COMPARISON PASSED")
    print("Created:")
    for path in [
        pathway_metrics_path,
        random_pathway_predictions_path,
        lineage_pathway_predictions_path,
        coverage_path,
        comparison_path,
        figure_path,
    ]:
        print(path)
    print(
        "\nInterpretation reminder: positive Pearson pathway-minus-gene values "
        "favor pathways; negative RMSE pathway-minus-gene values favor pathways. "
        "Either result is scientifically valid."
    )


if __name__ == "__main__":
    main()
