"""Build ICU stay cohort with LOS filter and optional delirium labels from ICD."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mimic_paths import diagnoses_icd_path

# ICD-10-CM: F05* delirium; R41.82 altered mental status (common proxy in claims).
# ICD-9-CM: 293.0, 293.1, 780.97 — match without punctuation via string prefixes.
DELIRIUM_ICD10_PREFIXES = ("F05", "R4182")


def _normalize_icd_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
        .replace({"NAN": ""})
    )


def delirium_mask_vectorized(icd_code: pd.Series, icd_version: pd.Series) -> pd.Series:
    n = _normalize_icd_series(icd_code)
    v = pd.to_numeric(icd_version, errors="coerce")
    icd10 = v == 10
    icd9 = v == 9
    d10 = icd10 & (
        n.str.startswith(DELIRIUM_ICD10_PREFIXES[0])
        | n.str.startswith(DELIRIUM_ICD10_PREFIXES[1])
    )
    d9 = icd9 & (
        n.str.startswith("2930")
        | n.str.startswith("2931")
        | n.str.startswith("78097")
    )
    return d10 | d9


def load_delirium_hadm_ids(
    diag_path: Path | None = None,
    chunksize: int = 500_000,
) -> set[int]:
    """hadm_ids with any delirium-related ICD row (hospitalization-level billing diagnoses)."""
    path = diag_path or diagnoses_icd_path()
    hadm_delirium: set[int] = set()
    for chunk in pd.read_csv(
        path,
        compression="infer",
        chunksize=chunksize,
        usecols=["hadm_id", "icd_code", "icd_version"],
    ):
        m = delirium_mask_vectorized(chunk["icd_code"], chunk["icd_version"])
        if m.any():
            for h in chunk.loc[m, "hadm_id"].dropna().astype(int):
                hadm_delirium.add(int(h))
    return hadm_delirium


def build_cohort(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    delirium_hadm_ids: set[int] | None = None,
    *,
    delirium_labels_known: bool = True,
    min_los_hours: float = 24.0,
    first_icu_only: bool = True,
    exclude_early_death_hours: float = 48.0,
) -> pd.DataFrame:
    """Build ICU cohort with DeLLiriuM-compatible inclusion/exclusion criteria.

    Parameters
    ----------
    first_icu_only:
        Keep only each patient's first ICU admission (DeLLiriuM criterion).
    exclude_early_death_hours:
        Drop stays where the patient dies within this many hours of ICU admission
        (DeLLiriuM excludes deaths within 48 h).
    """
    icu = icustays.copy()
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
    icu["los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0
    icu = icu[icu["los_hours"] >= min_los_hours].copy()

    adm_cols = [
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
    ]
    adm_keep = [c for c in adm_cols if c in admissions.columns]
    adm = admissions[adm_keep].drop_duplicates(subset=["hadm_id"])

    out = icu.merge(adm, on="hadm_id", how="left")
    pat_cols = [
        c
        for c in ["subject_id", "gender", "anchor_age", "anchor_year", "dod"]
        if c in patients.columns
    ]
    pat = patients[pat_cols].drop_duplicates(subset=["subject_id"])
    out = out.merge(pat, on="subject_id", how="left")

    adm_year = pd.to_datetime(out["admittime"], errors="coerce").dt.year
    out["age_at_admission"] = (
        out["anchor_age"] + (adm_year - out["anchor_year"])
    ).clip(upper=91)

    # --- First ICU admission per patient (DeLLiriuM criterion) ---
    if first_icu_only:
        out = out.sort_values("intime").groupby("subject_id", as_index=False).first()

    # --- Exclude early in-hospital deaths (DeLLiriuM: exclude deaths within 48 h) ---
    if exclude_early_death_hours > 0 and "deathtime" in out.columns:
        deathtime = pd.to_datetime(out["deathtime"], errors="coerce")
        hours_to_death = (deathtime - out["intime"]).dt.total_seconds() / 3600.0
        early_death = deathtime.notna() & (hours_to_death < exclude_early_death_hours)
        out = out[~early_death].copy()

    if not delirium_labels_known:
        out["delirium_icd"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    else:
        ids = delirium_hadm_ids or set()
        out["delirium_icd"] = out["hadm_id"].isin(ids).astype("int8")

    return out
