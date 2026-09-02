"""Audit the GDSC RNA-seq TPM matrix and trametinib model overlap.

This script is read-only with respect to the raw data. It writes small audit
tables to results/tables so the matching decisions can be reproduced.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


EXPRESSION_FILENAME = "rnaseq_merged_rsem_tpm_20260323.csv"
RESPONSE_FILENAME = "gdsc_trametinib_records.csv"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit GDSC TPM expression coverage for trametinib models."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root. By default, this is the parent of the scripts folder.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def read_response_records(path: Path) -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = {"GDSC1": [], "GDSC2": []}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_dataset", "SANGER_MODEL_ID", "CELL_LINE_NAME"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            dataset = row["source_dataset"].strip()
            if dataset in records:
                records[dataset].append(row)
    return records


def count_seen(value: str, seen: set[str], duplicates: set[str]) -> None:
    if not value:
        return
    if value in seen:
        duplicates.add(value)
    else:
        seen.add(value)


def audit_expression(path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    print(f"Reading expression matrix: {path}")
    print("This may take several minutes. The original file will not be changed.\n")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            model_id_row = next(reader)
            model_name_row = next(reader)
            source_row = next(reader)
            gene_header = next(reader)
        except StopIteration as exc:
            raise ValueError("Expression file ended before its four header rows.") from exc

        expected_labels = [
            (model_id_row, "model_id"),
            (model_name_row, "model_name"),
            (source_row, "data_source"),
            (gene_header, "gene_symbol"),
        ]
        for row, label in expected_labels:
            if not row or row[0].strip() != label:
                observed = row[0] if row else "<empty>"
                raise ValueError(
                    f"Expected first-column label '{label}', but found '{observed}'."
                )

        total_columns = len(gene_header)
        for label, row in (
            ("model_id", model_id_row),
            ("model_name", model_name_row),
            ("data_source", source_row),
        ):
            if len(row) != total_columns:
                raise ValueError(
                    f"Header row {label} has {len(row)} columns; expected {total_columns}."
                )

        model_ids = [value.strip() for value in model_id_row[3:]]
        model_names = [value.strip() for value in model_name_row[3:]]
        data_sources = [value.strip() for value in source_row[3:]]

        if not (len(model_ids) == len(model_names) == len(data_sources)):
            raise ValueError("Model ID, model name, and data-source counts disagree.")

        model_manifest = [
            {
                "SANGER_MODEL_ID": model_id,
                "MODEL_NAME": model_name,
                "EXPRESSION_DATA_SOURCE": data_source,
            }
            for model_id, model_name, data_source in zip(
                model_ids, model_names, data_sources
            )
        ]

        gene_rows = 0
        malformed_rows = 0
        missing_gene_symbols = 0
        missing_ensembl_ids = 0
        missing_sanger_gene_ids = 0
        gene_symbols: set[str] = set()
        duplicate_gene_symbols: set[str] = set()
        ensembl_ids: set[str] = set()
        duplicate_ensembl_ids: set[str] = set()
        sanger_gene_ids: set[str] = set()
        duplicate_sanger_gene_ids: set[str] = set()

        sampled_numeric_values = 0
        sampled_missing_values = 0
        sampled_non_numeric_values = 0
        sampled_min: float | None = None
        sampled_max: float | None = None
        sample_gene_limit = 250

        for row in reader:
            gene_rows += 1
            if len(row) != total_columns:
                malformed_rows += 1
                continue

            gene_symbol = row[0].strip()
            ensembl_id = row[1].strip()
            sanger_gene_id = row[2].strip()

            if gene_symbol:
                count_seen(gene_symbol, gene_symbols, duplicate_gene_symbols)
            else:
                missing_gene_symbols += 1

            if ensembl_id:
                count_seen(ensembl_id, ensembl_ids, duplicate_ensembl_ids)
            else:
                missing_ensembl_ids += 1

            if sanger_gene_id:
                count_seen(sanger_gene_id, sanger_gene_ids, duplicate_sanger_gene_ids)
            else:
                missing_sanger_gene_ids += 1

            if gene_rows <= sample_gene_limit:
                for value in row[3:]:
                    value = value.strip()
                    if value == "":
                        sampled_missing_values += 1
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        sampled_non_numeric_values += 1
                        continue
                    sampled_numeric_values += 1
                    sampled_min = number if sampled_min is None else min(sampled_min, number)
                    sampled_max = number if sampled_max is None else max(sampled_max, number)

            if gene_rows % 10_000 == 0:
                print(f"Processed {gene_rows:,} gene rows...")

    source_counts = Counter(data_sources)
    inventory = {
        "expression_file": path.name,
        "expression_file_bytes": path.stat().st_size,
        "total_columns": total_columns,
        "annotation_columns": 3,
        "expression_models": len(model_ids),
        "unique_model_ids": len(set(model_ids)),
        "duplicate_model_id_extra_columns": len(model_ids) - len(set(model_ids)),
        "sanger_expression_models": source_counts.get("Sanger", 0),
        "broad_expression_models": source_counts.get("Broad", 0),
        "other_expression_source_models": len(model_ids)
        - source_counts.get("Sanger", 0)
        - source_counts.get("Broad", 0),
        "gene_rows": gene_rows,
        "unique_gene_symbols": len(gene_symbols),
        "duplicate_gene_symbols": len(duplicate_gene_symbols),
        "missing_gene_symbols": missing_gene_symbols,
        "unique_ensembl_gene_ids": len(ensembl_ids),
        "duplicate_ensembl_gene_ids": len(duplicate_ensembl_ids),
        "missing_ensembl_gene_ids": missing_ensembl_ids,
        "unique_sanger_gene_ids": len(sanger_gene_ids),
        "duplicate_sanger_gene_ids": len(duplicate_sanger_gene_ids),
        "missing_sanger_gene_ids": missing_sanger_gene_ids,
        "malformed_gene_rows": malformed_rows,
        "sampled_genes_for_value_check": min(gene_rows, sample_gene_limit),
        "sampled_numeric_values": sampled_numeric_values,
        "sampled_missing_values": sampled_missing_values,
        "sampled_non_numeric_values": sampled_non_numeric_values,
        "sampled_expression_min": sampled_min,
        "sampled_expression_max": sampled_max,
    }
    return inventory, model_manifest


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    arguments = parse_arguments()
    project = arguments.project_root.resolve()
    expression_path = (
        project / "data" / "raw" / "gdsc" / EXPRESSION_FILENAME
    )
    response_path = project / "results" / "tables" / RESPONSE_FILENAME
    output_folder = project / "results" / "tables"
    output_folder.mkdir(parents=True, exist_ok=True)

    require_file(expression_path)
    require_file(response_path)

    response_records = read_response_records(response_path)
    inventory, manifest = audit_expression(expression_path)

    expression_ids = {row["SANGER_MODEL_ID"] for row in manifest}
    source_by_model = {
        row["SANGER_MODEL_ID"]: row["EXPRESSION_DATA_SOURCE"] for row in manifest
    }

    response_ids_by_dataset: dict[str, set[str]] = {}
    overlap_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []

    for dataset in ("GDSC1", "GDSC2"):
        dataset_records = response_records[dataset]
        response_ids = {
            row["SANGER_MODEL_ID"].strip()
            for row in dataset_records
            if row["SANGER_MODEL_ID"].strip()
        }
        response_ids_by_dataset[dataset] = response_ids
        matched_ids = response_ids.intersection(expression_ids)
        unmatched_ids = response_ids.difference(expression_ids)

        overlap_rows.append(
            {
                "dataset": dataset,
                "trametinib_response_models": len(response_ids),
                "matched_expression_models": len(matched_ids),
                "unmatched_expression_models": len(unmatched_ids),
                "expression_coverage_percent": round(
                    100 * len(matched_ids) / len(response_ids), 2
                )
                if response_ids
                else 0,
                "matched_sanger_expression_models": sum(
                    source_by_model[model_id] == "Sanger" for model_id in matched_ids
                ),
                "matched_broad_expression_models": sum(
                    source_by_model[model_id] == "Broad" for model_id in matched_ids
                ),
            }
        )

        cell_name_by_id = {
            row["SANGER_MODEL_ID"].strip(): row["CELL_LINE_NAME"].strip()
            for row in dataset_records
        }
        for model_id in sorted(unmatched_ids):
            unmatched_rows.append(
                {
                    "dataset": dataset,
                    "SANGER_MODEL_ID": model_id,
                    "CELL_LINE_NAME": cell_name_by_id.get(model_id, ""),
                }
            )

    for row in manifest:
        model_id = row["SANGER_MODEL_ID"]
        row["IN_GDSC1_TRAMETINIB"] = str(
            model_id in response_ids_by_dataset["GDSC1"]
        )
        row["IN_GDSC2_TRAMETINIB"] = str(
            model_id in response_ids_by_dataset["GDSC2"]
        )

    inventory_path = output_folder / "gdsc_expression_inventory.csv"
    overlap_path = output_folder / "gdsc_trametinib_expression_overlap.csv"
    unmatched_path = output_folder / "gdsc_trametinib_unmatched_expression_models.csv"
    manifest_path = output_folder / "gdsc_expression_model_manifest.csv"

    write_dict_rows(inventory_path, [inventory])
    write_dict_rows(overlap_path, overlap_rows)
    write_dict_rows(unmatched_path, unmatched_rows)
    write_dict_rows(manifest_path, manifest)

    print("\nGDSC EXPRESSION INVENTORY")
    for key, value in inventory.items():
        print(f"{key}: {value}")

    print("\nTRAMETINIB EXPRESSION COVERAGE")
    for row in overlap_rows:
        print(
            f"{row['dataset']}: "
            f"{row['matched_expression_models']}/"
            f"{row['trametinib_response_models']} models matched "
            f"({row['expression_coverage_percent']}%); "
            f"{row['unmatched_expression_models']} unmatched."
        )

    print("\nSaved:")
    print(inventory_path)
    print(overlap_path)
    print(unmatched_path)
    print(manifest_path)
    print("\nGDSC EXPRESSION AUDIT COMPLETED")


if __name__ == "__main__":
    main()
