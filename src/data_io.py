"""Input and harmonization utilities for DepMap expression and PRISM data."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ID_CANDIDATES = [
    "ModelID",
    "model_id",
    "DepMap_ID",
    "depmap_id",
    "row_name",
    "Unnamed: 0",
]


def _first_matching_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def inspect_csv(path, nrows=3):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sample = pd.read_csv(path, nrows=nrows)
    return {
        "path": str(path),
        "columns": sample.columns.tolist(),
        "sample": sample,
        "size_mb": path.stat().st_size / (1024**2),
    }


def load_expression(path):
    """Load a wide DepMap expression matrix and standardize its model ID."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    id_col = _first_matching_column(header.columns, ID_CANDIDATES)
    if id_col is None:
        id_col = header.columns[0]

    expression = pd.read_csv(path, low_memory=False)
    expression = expression.rename(columns={id_col: "depmap_id"})
    expression["depmap_id"] = expression["depmap_id"].astype(str).str.strip()
    expression = expression[expression["depmap_id"].str.match(r"^ACH-\d+", na=False)]
    expression = expression.drop_duplicates("depmap_id", keep="first")

    gene_cols = [c for c in expression.columns if c != "depmap_id"]
    expression[gene_cols] = expression[gene_cols].apply(
        pd.to_numeric, errors="coerce"
    ).astype("float32")
    return expression


def load_tidy_response(path):
    """Load an already tidy response table used by the synthetic demo."""
    response = pd.read_csv(path)
    required = {"depmap_id", "drug_name", "response"}
    missing = required - set(response.columns)
    if missing:
        raise ValueError(f"Tidy response is missing columns: {sorted(missing)}")
    response["response"] = pd.to_numeric(response["response"], errors="coerce")
    return response.dropna(subset=["depmap_id", "drug_name", "response"])


def _normalize_cell_columns(response, cell_info):
    """Map response matrix column labels to DepMap IDs when possible."""
    value_columns = response.columns[1:].tolist()
    if sum(bool(re.match(r"^ACH-\d+", str(c))) for c in value_columns) >= 10:
        return {c: str(c) for c in value_columns}

    mappings = {}
    for source in ["row_name", "ccle_name", "depmap_id"]:
        if source in cell_info.columns and "depmap_id" in cell_info.columns:
            pairs = cell_info[[source, "depmap_id"]].dropna().drop_duplicates(source)
            mappings.update(dict(zip(pairs[source].astype(str), pairs["depmap_id"])))
    return {c: mappings.get(str(c)) for c in value_columns}


def load_prism_primary(response_path, treatment_path, cell_info_path):
    """Convert the PRISM primary replicate-collapsed matrix to tidy rows.

    The official release is normally treatment-by-cell-line. The function also
    detects a transposed matrix and fails clearly if identifier mapping cannot
    be established.
    """
    response = pd.read_csv(response_path, low_memory=False)
    treatment = pd.read_csv(treatment_path, low_memory=False)
    cell_info = pd.read_csv(cell_info_path, low_memory=False)

    if "depmap_id" not in cell_info.columns:
        source = _first_matching_column(cell_info.columns, ID_CANDIDATES)
        if source is None:
            raise ValueError("Could not identify depmap_id in PRISM cell-line info.")
        cell_info = cell_info.rename(columns={source: "depmap_id"})

    first_col = response.columns[0]
    first_values = response[first_col].astype(str)
    rows_are_cells = first_values.str.match(r"^ACH-\d+").mean() > 0.5

    if rows_are_cells:
        tidy = response.melt(
            id_vars=first_col,
            var_name="treatment_key",
            value_name="response",
        ).rename(columns={first_col: "depmap_id"})
    else:
        mapping = _normalize_cell_columns(response, cell_info)
        usable = [c for c, model_id in mapping.items() if model_id]
        if len(usable) < 10:
            raise ValueError(
                "Could not map enough PRISM matrix columns to DepMap IDs. "
                "Review the README and the printed inventory before editing code."
            )
        tidy = response[[first_col] + usable].melt(
            id_vars=first_col,
            var_name="cell_key",
            value_name="response",
        )
        tidy = tidy.rename(columns={first_col: "treatment_key"})
        tidy["depmap_id"] = tidy["cell_key"].map(mapping)
        tidy = tidy.drop(columns="cell_key")

    # Join treatment metadata using an explicit shared key when possible.
    join_candidates = [
        "column_name",
        "row_name",
        "broad_id",
        "Broad_ID",
        "treatment_key",
    ]
    treatment_join = None
    for candidate in join_candidates:
        if candidate in treatment.columns:
            overlap = set(tidy["treatment_key"].astype(str)).intersection(
                set(treatment[candidate].astype(str))
            )
            if overlap:
                treatment_join = candidate
                break

    if treatment_join is not None:
        treatment = treatment.copy()
        treatment[treatment_join] = treatment[treatment_join].astype(str)
        tidy["treatment_key"] = tidy["treatment_key"].astype(str)
        tidy = tidy.merge(
            treatment,
            left_on="treatment_key",
            right_on=treatment_join,
            how="left",
            suffixes=("", "_treatment"),
        )
    elif len(treatment) == response.shape[0] and not rows_are_cells:
        treatment = treatment.copy()
        treatment["treatment_key"] = response[first_col].astype(str).to_numpy()
        tidy["treatment_key"] = tidy["treatment_key"].astype(str)
        tidy = tidy.merge(treatment, on="treatment_key", how="left")
    else:
        raise ValueError(
            "Treatment metadata could not be joined safely. Do not join by row "
            "position unless row counts and the release README confirm alignment."
        )

    name_col = _first_matching_column(
        tidy.columns,
        ["name", "drug_name", "compound_name", "pert_iname", "compound"],
    )
    if name_col is None:
        tidy["drug_name"] = tidy["treatment_key"]
    else:
        tidy["drug_name"] = tidy[name_col].astype(str)

    tidy["response"] = pd.to_numeric(tidy["response"], errors="coerce")
    tidy["depmap_id"] = tidy["depmap_id"].astype(str).str.strip()
    tidy = tidy.dropna(subset=["depmap_id", "drug_name", "response"])

    tissue_cols = [
        c
        for c in ["depmap_id", "ccle_name", "primary_tissue", "secondary_tissue"]
        if c in cell_info.columns
    ]
    if "depmap_id" in tissue_cols:
        tidy = tidy.merge(
            cell_info[tissue_cols].drop_duplicates("depmap_id"),
            on="depmap_id",
            how="left",
        )
    return tidy


def rank_candidate_drugs(tidy_response):
    """Summarize sample size and variation for each named compound."""
    ranked = (
        tidy_response.groupby("drug_name", dropna=False)
        .agg(
            n_cells=("depmap_id", "nunique"),
            n_observations=("response", "size"),
            response_mean=("response", "mean"),
            response_sd=("response", "std"),
            response_min=("response", "min"),
            response_max=("response", "max"),
        )
        .reset_index()
    )
    ranked = ranked.sort_values(
        ["n_cells", "response_sd"], ascending=[False, False]
    )
    return ranked

