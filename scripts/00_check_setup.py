"""Verify the beginner research environment before any analysis."""

from pathlib import Path
import platform
import sys


REQUIRED_DIRS = [
    Path("data/raw/depmap"),
    Path("data/raw/prism"),
    Path("data/demo"),
    Path("data/processed"),
    Path("results/figures"),
    Path("results/tables"),
]


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Operating system: {platform.platform()}")

    for path in REQUIRED_DIRS:
        path.mkdir(parents=True, exist_ok=True)
        print(f"OK directory: {path}")

    imports = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pyarrow": "pyarrow",
        "duckdb": "duckdb",
    }

    failed = []
    for label, module_name in imports.items():
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"OK package: {label} {version}")
        except Exception as exc:  # beginner-facing diagnostic
            failed.append((label, str(exc)))
            print(f"MISSING package: {label} -> {exc}")

    if failed:
        print("\nSETUP CHECK FAILED")
        print("Run: conda env update -f environment.yml --prune")
        raise SystemExit(1)

    print("\nSETUP CHECK PASSED")


if __name__ == "__main__":
    main()

