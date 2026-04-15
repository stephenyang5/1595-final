"""CLI: build ICU cohort (LOS ≥ threshold) and write to results/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.cohort import build_cohort, load_delirium_hadm_ids
from src.mimic_paths import (
    admissions_path,
    diagnoses_icd_path,
    icustays_path,
    patients_path,
    mimic_root,
)


def _default_output_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "results"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MIMIC-IV ICU cohort with LOS filter.")
    p.add_argument(
        "--mimic-root",
        type=Path,
        default=None,
        help="Override MIMIC root (else MIMIC_ROOT env or Oscar default).",
    )
    p.add_argument("--min-los-hours", type=float, default=24.0)
    p.add_argument(
        "--skip-delirium",
        action="store_true",
        help="Do not scan diagnoses_icd (faster; delirium_icd column null).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (.csv or .csv.gz). Default: results/cohort_icu_los_ge{hours}.csv.gz",
    )
    args = p.parse_args(argv)

    if args.mimic_root is not None:
        import os

        os.environ["MIMIC_ROOT"] = str(args.mimic_root)

    root = mimic_root()
    icu_path = icustays_path()
    adm_path = admissions_path()
    pat_path = patients_path()

    for path, label in (
        (icu_path, "icustays"),
        (adm_path, "admissions"),
        (pat_path, "patients"),
    ):
        if not path.is_file():
            print(f"Missing {label} file: {path}", file=sys.stderr)
            print(f"MIMIC root resolved to: {root}", file=sys.stderr)
            return 1

    print(f"Loading icustays from {icu_path}")
    icustays = pd.read_csv(icu_path, compression="gzip")
    print(f"Loading admissions from {adm_path}")
    admissions = pd.read_csv(adm_path)
    print(f"Loading patients from {pat_path}")
    patients = pd.read_csv(pat_path)

    scan_delirium = not args.skip_delirium
    delirium_ids: set[int] | None = None
    if scan_delirium:
        dpath = diagnoses_icd_path()
        if not dpath.is_file():
            print(f"Missing diagnoses_icd: {dpath}", file=sys.stderr)
            return 1
        print("Scanning diagnoses_icd for delirium-related ICD codes …")
        delirium_ids = load_delirium_hadm_ids()

    cohort = build_cohort(
        icustays,
        admissions,
        patients,
        delirium_hadm_ids=delirium_ids if scan_delirium else None,
        delirium_labels_known=scan_delirium,
        min_los_hours=args.min_los_hours,
    )

    out = args.output
    if out is None:
        out_dir = _default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        los = args.min_los_hours
        h = int(los) if los == int(los) else los
        out = out_dir / f"cohort_icu_los_ge{h}h.csv.gz"

    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out, index=False, compression="infer")
    print(f"Wrote {len(cohort):,} rows to {out}")
    if scan_delirium:
        pos = int(cohort["delirium_icd"].sum())
        print(f"delirium_icd positive (hospitalization-level): {pos:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
