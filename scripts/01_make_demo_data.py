"""Generate a small synthetic dataset to verify the modeling workflow."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    rng = np.random.default_rng(42)
    n_samples = 160
    n_genes = 80

    depmap_ids = [f"ACH-{i:06d}" for i in range(1, n_samples + 1)]
    gene_names = [f"GENE{i:04d} ({1000 + i})" for i in range(1, n_genes + 1)]
    x = rng.normal(size=(n_samples, n_genes))

    # Simulated response has a learnable signal plus noise.
    y = (
        -0.75 * x[:, 0]
        + 0.55 * x[:, 1]
        - 0.35 * x[:, 2]
        + 0.20 * x[:, 3]
        + rng.normal(scale=0.8, size=n_samples)
    )

    tissues = rng.choice(
        ["lung", "breast", "colorectal", "skin"],
        size=n_samples,
        p=[0.30, 0.25, 0.25, 0.20],
    )

    expression = pd.DataFrame(x, columns=gene_names)
    expression.insert(0, "ModelID", depmap_ids)

    response = pd.DataFrame(
        {
            "depmap_id": depmap_ids,
            "drug_name": "DemoDrug",
            "response": y,
            "primary_tissue": tissues,
        }
    )

    out = ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    expression.to_csv(out / "expression_demo.csv", index=False)
    response.to_csv(out / "response_demo.csv", index=False)

    print(f"Created: {out / 'expression_demo.csv'}")
    print(f"Created: {out / 'response_demo.csv'}")
    print(f"Samples: {n_samples}; genes: {n_genes}; drug: DemoDrug")
    print("Reminder: these data are synthetic and are not a scientific result.")


if __name__ == "__main__":
    main()

