# Data acquisition and placement

Raw source data are intentionally excluded from version control. Download each file directly from its provider, accept any applicable terms, and preserve the original filename.

## Required directory structure

```text
data/
└── raw/
    ├── depmap/
    │   ├── OmicsExpressionProteinCodingGenesTPMLogp1.csv
    │   └── Model.csv
    ├── prism/
    │   ├── primary-screen-replicate-collapsed-logfold-change.csv
    │   ├── primary-screen-replicate-collapsed-treatment-info.csv
    │   └── primary-screen-cell-line-info.csv
    ├── gdsc/
    │   ├── GDSC1_fitted_dose_response_27Oct23.xlsx
    │   ├── GDSC2_fitted_dose_response_27Oct23.xlsx
    │   └── rnaseq_merged_rsem_tpm_20260323.csv
    └── msigdb/
        └── h.all.v2026.1.Hs.symbols.gmt
```

## Providers

| Source | Role | Access point |
|---|---|---|
| DepMap | Baseline protein-coding expression and model crosswalk | [DepMap Data Portal](https://depmap.org/portal/data_page/) |
| PRISM | Training drug-response outcome and treatment/cell metadata | [PRISM Repurposing](https://depmap.org/repurposing/) |
| GDSC | External trametinib response | [GDSC downloads](https://www.cancerrxgene.org/downloads/bulk_download) |
| Cell Model Passports | Sanger/Broad RNA-seq TPM expression | [Downloads](https://cellmodelpassports.sanger.ac.uk/downloads) |
| MSigDB | Human Hallmark gene-symbol sets | [Hallmark collection](https://www.gsea-msigdb.org/gsea/msigdb/human/genesets.jsp?collection=H) |

Release-specific filenames can change. The filenames above document the exact releases used for the committed results. If a provider supplies a newer version, record the new filename and date and treat the rerun as a new analysis version.

## Do not commit raw files

The repository `.gitignore` excludes `data/raw/`, spreadsheets, Parquet/HDF files, and GMT files. Before every push, run:

```bash
git status
git ls-files data/raw
```

The second command should return nothing.

## Integrity checklist

After download:

1. Confirm every expected file exists and opens normally.
2. Preserve the original filenames.
3. Record the download date, release/version, source URL, file size, and checksum in a private data manifest.
4. Run `python scripts/00_check_setup.py` and `python scripts/03_inventory_real_data.py` before fitting a model.
5. Compare row counts and headers with [docs/DATA_DICTIONARY.md](../docs/DATA_DICTIONARY.md).

Do not redistribute provider-controlled data merely because they can be downloaded publicly. Cite and follow the terms displayed by each provider.

