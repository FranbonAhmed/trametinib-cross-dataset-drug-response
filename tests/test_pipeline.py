from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import load_expression, load_tidy_response, rank_candidate_drugs
from src.modeling import evaluate_baselines, merge_one_drug


def test_demo_style_pipeline(tmp_path):
    rng = np.random.default_rng(7)
    n = 60
    ids = [f"ACH-{i:06d}" for i in range(n)]
    x = rng.normal(size=(n, 12))
    y = 0.8 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(scale=0.6, size=n)

    expression = pd.DataFrame(x, columns=[f"G{i} ({i})" for i in range(12)])
    expression.insert(0, "ModelID", ids)
    response = pd.DataFrame(
        {"depmap_id": ids, "drug_name": "TestDrug", "response": y}
    )

    exp_path = tmp_path / "expression.csv"
    res_path = tmp_path / "response.csv"
    expression.to_csv(exp_path, index=False)
    response.to_csv(res_path, index=False)

    loaded_expression = load_expression(exp_path)
    loaded_response = load_tidy_response(res_path)
    identifiers, features, outcome = merge_one_drug(
        loaded_expression, loaded_response, "TestDrug"
    )
    metrics, predictions = evaluate_baselines(features, outcome)

    assert len(identifiers) == n
    assert set(metrics["model"]) == {"Mean baseline", "Elastic Net", "Random forest"}
    assert len(predictions) == n
    assert metrics["rmse"].notna().all()


def test_candidate_ranking():
    response = pd.DataFrame(
        {
            "depmap_id": ["A", "B", "C", "A", "B"],
            "drug_name": ["Drug1", "Drug1", "Drug1", "Drug2", "Drug2"],
            "response": [0.0, 1.0, 2.0, 0.0, 0.2],
        }
    )
    ranked = rank_candidate_drugs(response)
    assert ranked.iloc[0]["drug_name"] == "Drug1"
    assert ranked.iloc[0]["n_cells"] == 3

