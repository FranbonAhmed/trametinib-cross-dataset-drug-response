"""Extract one drug from the GDSC1 and GDSC2 fitted-response workbooks.

The source workbooks are retained unchanged. This script creates a combined
analysis table, a dataset-level inventory, and a cross-screen agreement audit
under ``results/tables``. The detailed extracted record table is ignored by
Git because it is derived directly from provider data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
from scipy.stats import spearmanr


REQUIRED_COLUMNS = {
    "CELL_LINE_NAME",
    "SANGER_MODEL_ID",
    "DRUG_NAME",
    "LN_IC50",
    "AUC",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and audit one drug from GDSC1 and GDSC2."
    )
    parser.add_argument("--drug", default="trametinib")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root; defaults to the parent of the scripts folder.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).casefold()).strip("_")
    return value or "drug"


def require_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")


def load_one_dataset(path: Path, dataset: str, drug: str) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required GDSC workbook not found: {path}")

    frame = pd.read_excel(path, engine="openpyxl")
    require_columns(frame, path)

    drug_names = frame["DRUG_NAME"].astype("string").str.strip()
    selected = frame.loc[drug_names.str.casefold() == drug.casefold()].copy()
    if selected.empty:
        raise ValueError(f"No exact case-insensitive match for {drug!r} in {path.name}")

    selected.insert(0, "source_dataset", dataset)
    selected["SANGER_MODEL_ID"] = (
        selected["SANGER_MODEL_ID"].astype("string").str.strip()
    )
    selected["CELL_LINE_NAME"] = selected["CELL_LINE_NAME"].astype("string").str.strip()
    selected["LN_IC50"] = pd.to_numeric(selected["LN_IC50"], errors="coerce")
    selected["AUC"] = pd.to_numeric(selected["AUC"], errors="coerce")

    duplicate_keys = selected.duplicated(["source_dataset", "SANGER_MODEL_ID"], keep=False)
    duplicate_key_rows = int(duplicate_keys.sum())

    inventory = {
        "source_dataset": dataset,
        "source_file": path.name,
        "source_rows": int(len(frame)),
        "drug_name_requested": drug,
        "matching_drug_rows": int(len(selected)),
        "unique_sanger_models": int(selected["SANGER_MODEL_ID"].nunique(dropna=True)),
        "missing_sanger_model_id": int(selected["SANGER_MODEL_ID"].isna().sum()),
        "duplicate_key_rows": duplicate_key_rows,
        "missing_ln_ic50": int(selected["LN_IC50"].isna().sum()),
        "missing_auc": int(selected["AUC"].isna().sum()),
        "ln_ic50_min": float(selected["LN_IC50"].min()),
        "ln_ic50_median": float(selected["LN_IC50"].median()),
        "ln_ic50_max": float(selected["LN_IC50"].max()),
        "auc_min": float(selected["AUC"].min()),
        "auc_median": float(selected["AUC"].median()),
        "auc_max": float(selected["AUC"].max()),
    }
    return selected, inventory


def cross_screen_agreement(combined: pd.DataFrame) -> pd.DataFrame:
    columns = ["SANGER_MODEL_ID", "LN_IC50", "AUC"]
    first = combined.loc[combined["source_dataset"] == "GDSC1", columns].copy()
    second = combined.loc[combined["source_dataset"] == "GDSC2", columns].copy()
    shared = first.merge(second, on="SANGER_MODEL_ID", suffixes=("_GDSC1", "_GDSC2"))

    rows = []
    for outcome in ["LN_IC50", "AUC"]:
        values = shared[[f"{outcome}_GDSC1", f"{outcome}_GDSC2"]].dropna()
        rho = float(spearmanr(values.iloc[:, 0], values.iloc[:, 1]).statistic)
        rows.append(
            {
                "outcome": outcome,
                "shared_models": int(len(values)),
                "spearman_rho": rho,
                "note": "Cross-screen agreement; shared models are not independent cohorts.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    project = args.project_root.resolve()
    source_folder = project / "data" / "raw" / "gdsc"
    output_folder = project / "results" / "tables"
    output_folder.mkdir(parents=True, exist_ok=True)

    sources = {
        "GDSC1": source_folder / "GDSC1_fitted_dose_response_27Oct23.xlsx",
        "GDSC2": source_folder / "GDSC2_fitted_dose_response_27Oct23.xlsx",
    }

    frames = []
    inventory_rows = []
    for dataset, path in sources.items():
        selected, inventory = load_one_dataset(path, dataset, args.drug)
        frames.append(selected)
        inventory_rows.append(inventory)

    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["source_dataset", "SANGER_MODEL_ID"]).any():
        raise ValueError(
            "Duplicate dataset/model keys remain after drug filtering. "
            "Inspect technical replicates before external validation."
        )

    slug = safe_name(args.drug)
    records_path = output_folder / f"gdsc_{slug}_records.csv"
    inventory_path = output_folder / f"gdsc_{slug}_inventory.csv"
    agreement_path = output_folder / f"gdsc_{slug}_cross_screen_agreement.csv"

    combined.to_csv(records_path, index=False)
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)
    cross_screen_agreement(combined).to_csv(agreement_path, index=False)

    print(f"Saved {len(combined):,} filtered records to {records_path}")
    print(f"Saved dataset inventory to {inventory_path}")
    print(f"Saved cross-screen audit to {agreement_path}")


if __name__ == "__main__":
    main()

