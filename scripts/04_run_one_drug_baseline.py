"""Run the first leakage-safe real-data baseline for one PRISM drug."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_expression, load_prism_primary
from src.modeling import evaluate_baselines, merge_one_drug, safe_name, save_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drug",
        required=True,
        help="Exact drug name from results/tables/candidate_drugs.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    expression_path = ROOT / "data/raw/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    response_path = ROOT / "data/raw/prism/primary-screen-replicate-collapsed-logfold-change.csv"
    treatment_path = ROOT / "data/raw/prism/primary-screen-replicate-collapsed-treatment-info.csv"
    cells_path = ROOT / "data/raw/prism/primary-screen-cell-line-info.csv"

    for path in [expression_path, response_path, treatment_path, cells_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.relative_to(ROOT)}. Complete the data inventory first."
            )

    print("Loading PRISM and harmonizing identifiers...")
    response = load_prism_primary(response_path, treatment_path, cells_path)
    print("Loading DepMap expression. This may take several minutes...")
    expression = load_expression(expression_path)
    identifiers, x, y = merge_one_drug(expression, response, args.drug)
    print(f"Matched cell lines: {len(y)}; expression features: {x.shape[1]:,}")

    metrics, predictions = evaluate_baselines(x, y)
    predictions.insert(0, "depmap_id", identifiers["depmap_id"].to_numpy())
    prefix = safe_name(args.drug)
    paths = save_results(metrics, predictions, ROOT / "results", prefix)

    print("\nREAL-DATA BASELINE METRICS")
    print(metrics.to_string(index=False))
    print("\nCreated:")
    for path in paths:
        print(path)
    print("\nNext: record the release, sample count, metrics, and limitations in the research log.")


if __name__ == "__main__":
    main()

