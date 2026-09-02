"""Audit cell-line independence before PRISM-to-GDSC external validation.

The script does not fit a model. It maps the ACH identifiers used in the
PRISM training predictions to Sanger model identifiers, then separates GDSC
models into previously seen and genuinely held-out cell-line groups.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit leakage-safe PRISM-to-GDSC validation cohorts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root; defaults to the parent of the scripts folder.",
    )
    parser.add_argument(
        "--drug",
        default="trametinib",
        help="Safe filename prefix used for the PRISM prediction file.",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return value or "drug"


def normalized_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(
    columns: list[str], exact_normalized_names: set[str], contains: tuple[str, ...] = ()
) -> str | None:
    for column in columns:
        if normalized_column_name(column) in exact_normalized_names:
            return column
    if contains:
        for column in columns:
            normalized = normalized_column_name(column)
            if all(part in normalized for part in contains):
                return column
    return None


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames=None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_mapping(path: Path):
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path.name} contains no data rows.")
    columns = list(rows[0].keys())
    ach_column = find_column(
        columns,
        {"modelid", "depmapid", "achmodelid", "broadmodelid"},
    )
    sanger_column = find_column(
        columns,
        {"sangermodelid", "sangerid", "sidm"},
        contains=("sanger", "model"),
    )
    if ach_column is None or sanger_column is None:
        raise ValueError(
            "Could not identify both ACH/DepMap and Sanger model-ID columns in "
            f"{path.name}. Available columns: {columns}"
        )

    ach_to_sanger: dict[str, set[str]] = defaultdict(set)
    sanger_to_ach: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ach_id = str(row.get(ach_column, "")).strip()
        sanger_id = str(row.get(sanger_column, "")).strip()
        if not re.fullmatch(r"ACH-\d+", ach_id):
            continue
        if not re.fullmatch(r"SIDM\d+", sanger_id):
            continue
        ach_to_sanger[ach_id].add(sanger_id)
        sanger_to_ach[sanger_id].add(ach_id)
    if not ach_to_sanger:
        raise ValueError(
            f"No valid ACH-to-SIDM mappings were found using {ach_column} and "
            f"{sanger_column}."
        )
    return ach_column, sanger_column, ach_to_sanger, sanger_to_ach


def load_training_ids(path: Path) -> set[str]:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path.name} contains no data rows.")
    columns = list(rows[0].keys())
    id_column = find_column(columns, {"depmapid", "modelid", "achmodelid"})
    if id_column is None:
        raise ValueError(
            f"Could not find the DepMap ID column in {path.name}. "
            f"Available columns: {columns}"
        )
    return {
        str(row.get(id_column, "")).strip()
        for row in rows
        if re.fullmatch(r"ACH-\d+", str(row.get(id_column, "")).strip())
    }


def main() -> None:
    arguments = parse_arguments()
    project = arguments.project_root.resolve()
    table_folder = project / "results" / "tables"
    table_folder.mkdir(parents=True, exist_ok=True)

    model_path = project / "data" / "raw" / "depmap" / "Model.csv"
    prediction_path = table_folder / f"{safe_name(arguments.drug)}_predictions.csv"
    response_path = table_folder / "gdsc_trametinib_records.csv"
    expression_manifest_path = table_folder / "gdsc_expression_model_manifest.csv"

    ach_column, sanger_column, ach_to_sanger, sanger_to_ach = load_model_mapping(
        model_path
    )
    training_ach_ids = load_training_ids(prediction_path)
    training_sidm_ids = {
        sanger_id
        for ach_id in training_ach_ids
        for sanger_id in ach_to_sanger.get(ach_id, set())
    }
    unmapped_training_ach_ids = sorted(
        ach_id for ach_id in training_ach_ids if ach_id not in ach_to_sanger
    )

    expression_manifest = read_rows(expression_manifest_path)
    expression_source_by_sidm = {
        row["SANGER_MODEL_ID"].strip(): row["EXPRESSION_DATA_SOURCE"].strip()
        for row in expression_manifest
        if row.get("SANGER_MODEL_ID", "").strip()
    }

    response_records = read_rows(response_path)
    required_response_columns = {
        "source_dataset",
        "SANGER_MODEL_ID",
        "CELL_LINE_NAME",
    }
    if response_records:
        missing = required_response_columns.difference(response_records[0].keys())
        if missing:
            raise ValueError(
                f"{response_path.name} is missing columns: {sorted(missing)}"
            )

    summary_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for dataset in ("GDSC1", "GDSC2"):
        dataset_rows = [
            row for row in response_records if row["source_dataset"].strip() == dataset
        ]
        record_by_sidm = {
            row["SANGER_MODEL_ID"].strip(): row
            for row in dataset_rows
            if row["SANGER_MODEL_ID"].strip()
        }
        response_ids = set(record_by_sidm)
        expression_ids = response_ids.intersection(expression_source_by_sidm)
        seen_ids = expression_ids.intersection(training_sidm_ids)
        held_out_ids = expression_ids.difference(training_sidm_ids)
        held_out_sanger_ids = {
            model_id
            for model_id in held_out_ids
            if expression_source_by_sidm[model_id] == "Sanger"
        }
        held_out_broad_ids = {
            model_id
            for model_id in held_out_ids
            if expression_source_by_sidm[model_id] == "Broad"
        }
        seen_sanger_ids = {
            model_id
            for model_id in seen_ids
            if expression_source_by_sidm[model_id] == "Sanger"
        }
        seen_broad_ids = {
            model_id
            for model_id in seen_ids
            if expression_source_by_sidm[model_id] == "Broad"
        }

        summary_rows.append(
            {
                "dataset": dataset,
                "prism_training_ach_models": len(training_ach_ids),
                "prism_training_models_mapped_to_sidm": len(training_sidm_ids),
                "prism_training_ach_models_without_sidm_mapping": len(
                    unmapped_training_ach_ids
                ),
                "gdsc_trametinib_response_models": len(response_ids),
                "gdsc_models_with_expression": len(expression_ids),
                "gdsc_models_without_expression": len(response_ids - expression_ids),
                "previously_seen_prism_training_models": len(seen_ids),
                "held_out_models_any_expression_source": len(held_out_ids),
                "held_out_sanger_expression_models": len(held_out_sanger_ids),
                "held_out_broad_expression_models": len(held_out_broad_ids),
                "seen_sanger_expression_models": len(seen_sanger_ids),
                "seen_broad_expression_models": len(seen_broad_ids),
            }
        )

        for model_id in sorted(response_ids):
            has_expression = model_id in expression_ids
            seen_in_training = model_id in training_sidm_ids
            source = expression_source_by_sidm.get(model_id, "")
            if not has_expression:
                role = "exclude_no_expression"
            elif not seen_in_training and source == "Sanger":
                role = "strict_external_sanger_expression"
            elif not seen_in_training and source == "Broad":
                role = "heldout_cell_line_broad_expression"
            elif seen_in_training and source == "Sanger":
                role = "same_cell_cross_assay_sanger_expression"
            else:
                role = "same_cell_cross_assay_broad_expression"

            source_row = record_by_sidm[model_id]
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "SANGER_MODEL_ID": model_id,
                    "CELL_LINE_NAME": source_row.get("CELL_LINE_NAME", ""),
                    "ACH_MODEL_IDS": ";".join(sorted(sanger_to_ach.get(model_id, set()))),
                    "HAS_EXPRESSION": has_expression,
                    "EXPRESSION_DATA_SOURCE": source,
                    "SEEN_IN_PRISM_TRAINING": seen_in_training,
                    "ANALYSIS_ROLE": role,
                    "LN_IC50": source_row.get("LN_IC50", ""),
                    "AUC": source_row.get("AUC", ""),
                }
            )

    summary_path = table_folder / "gdsc_external_validation_split_audit.csv"
    manifest_path = table_folder / "gdsc_external_validation_split_manifest.csv"
    unmapped_path = table_folder / "prism_training_ids_without_sanger_mapping.csv"

    write_rows(summary_path, summary_rows)
    write_rows(manifest_path, manifest_rows)
    write_rows(
        unmapped_path,
        [{"ACH_MODEL_ID": model_id} for model_id in unmapped_training_ach_ids],
        fieldnames=["ACH_MODEL_ID"],
    )

    print("EXTERNAL VALIDATION SPLIT AUDIT")
    print(f"Model.csv ACH column: {ach_column}")
    print(f"Model.csv Sanger column: {sanger_column}")
    print(f"PRISM training ACH models: {len(training_ach_ids)}")
    print(f"Training models mapped to SIDM: {len(training_sidm_ids)}")
    print(f"Training ACH models without SIDM mapping: {len(unmapped_training_ach_ids)}")

    for row in summary_rows:
        print(f"\n{row['dataset']}")
        print(
            "  Response models with expression: "
            f"{row['gdsc_models_with_expression']}"
        )
        print(
            "  Previously seen in PRISM training: "
            f"{row['previously_seen_prism_training_models']}"
        )
        print(
            "  Held-out models, any expression source: "
            f"{row['held_out_models_any_expression_source']}"
        )
        print(
            "  Strict held-out Sanger-expression models: "
            f"{row['held_out_sanger_expression_models']}"
        )

    print("\nSaved:")
    print(summary_path)
    print(manifest_path)
    print(unmapped_path)
    print("\nEXTERNAL VALIDATION SPLIT AUDIT COMPLETED")


if __name__ == "__main__":
    main()
