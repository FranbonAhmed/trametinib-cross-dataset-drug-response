"""Audit official raw files and create a candidate-drug summary."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_io import inspect_csv, load_prism_primary, rank_candidate_drugs


EXPRESSION = ROOT / "data/raw/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
PRISM_RESPONSE = ROOT / "data/raw/prism/primary-screen-replicate-collapsed-logfold-change.csv"
PRISM_TREATMENT = ROOT / "data/raw/prism/primary-screen-replicate-collapsed-treatment-info.csv"
PRISM_CELLS = ROOT / "data/raw/prism/primary-screen-cell-line-info.csv"


def main():
    required = [EXPRESSION, PRISM_RESPONSE, PRISM_TREATMENT, PRISM_CELLS]
    missing = [path for path in required if not path.exists()]
    if missing:
        print("MISSING REQUIRED FILES")
        for path in missing:
            print(f"- {path.relative_to(ROOT)}")
        print("\nFollow Part 4 in BEGINNER_START_HERE.md, then run this again.")
        raise SystemExit(1)

    print("FILE INVENTORY")
    for path in required:
        info = inspect_csv(path)
        shown = info["columns"][:12]
        suffix = " ..." if len(info["columns"]) > 12 else ""
        print(f"\n{path.relative_to(ROOT)}")
        print(f"  size_mb: {info['size_mb']:.2f}")
        print(f"  first columns: {shown}{suffix}")

    response = load_prism_primary(PRISM_RESPONSE, PRISM_TREATMENT, PRISM_CELLS)
    print(f"\nTidy PRISM observations: {len(response):,}")
    print(f"Unique cell lines: {response['depmap_id'].nunique():,}")
    print(f"Unique drug names: {response['drug_name'].nunique():,}")

    ranked = rank_candidate_drugs(response)
    out = ROOT / "results/tables/candidate_drugs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)

    preferred = [
        "trametinib",
        "selumetinib",
        "dabrafenib",
        "gefitinib",
        "erlotinib",
        "olaparib",
        "palbociclib",
        "lapatinib",
    ]
    mask = ranked["drug_name"].astype(str).str.casefold().isin(preferred)
    print("\nPREFERRED CANDIDATES PRESENT")
    if mask.any():
        print(ranked.loc[mask].to_string(index=False))
    else:
        print("No exact preferred names found. Search candidate_drugs.csv manually.")

    print(f"\nCreated: {out}")
    print("Do not choose a drug until you review sample size and response SD.")


if __name__ == "__main__":
    main()

