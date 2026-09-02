"""Run the verified baseline workflow on synthetic demonstration data."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_expression, load_tidy_response
from src.modeling import evaluate_baselines, merge_one_drug, save_results


def main():
    expression_path = ROOT / "data" / "demo" / "expression_demo.csv"
    response_path = ROOT / "data" / "demo" / "response_demo.csv"
    if not expression_path.exists() or not response_path.exists():
        raise FileNotFoundError(
            "Demo data are missing. Run: python scripts/01_make_demo_data.py"
        )

    expression = load_expression(expression_path)
    response = load_tidy_response(response_path)
    identifiers, x, y = merge_one_drug(expression, response, "DemoDrug")
    metrics, predictions = evaluate_baselines(x, y)
    predictions.insert(0, "depmap_id", identifiers["depmap_id"].to_numpy())
    paths = save_results(metrics, predictions, ROOT / "results", "demo")

    print("\nDEMO METRICS")
    print(metrics.to_string(index=False))
    print("\nCreated:")
    for path in paths:
        print(path)
    print("\nDemo baseline complete. These are synthetic, not scientific, results.")


if __name__ == "__main__":
    main()

