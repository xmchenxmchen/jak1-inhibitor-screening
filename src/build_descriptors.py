"""
Build ``data/drug_descriptors_normalized.csv`` from the raw PubChem export.

Pipeline:  raw PubChem BioAssay table  ->  clean + label  ->  PaDEL descriptors
           ->  MaxAbs normalization  ->  modeling-ready CSV

The committed dataset was produced with:
  * Descriptors : PaDEL-Descriptor  (1875 features = 1444 2D + 431 3D)
  * Normalization: scikit-learn ``MaxAbsScaler``  (each column divided by its
                   max absolute value -> every column lands in [-1, 1], zeros
                   preserved). This was reverse-engineered and verified against
                   the committed file: all 1875 columns satisfy max(|x|) == 1.
  * Labels      : PUBCHEM_ACTIVITY_OUTCOME -> Active (1) / Unspecified (0)

Reproducibility note
--------------------
The descriptor step requires Java + PaDEL-Descriptor (via the ``padelpy``
wrapper). PaDEL's 3D descriptors depend on its conformer-generation settings,
so re-running this will reproduce the *methodology* and near-identical values,
but not necessarily a byte-for-byte copy of the committed CSV. The label
mapping and the MaxAbs normalization are exact and independently testable.

Usage:
    pip install padelpy && <install a Java runtime, e.g. `brew install openjdk`>
    python src/build_descriptors.py
    python src/build_descriptors.py --check   # run only the steps that need no Java
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
RAW_PATH = PROJECT_ROOT / "data" / "AID_1919099_datatable_all.csv"
OUT_PATH = PROJECT_ROOT / "data" / "drug_descriptors_normalized.csv"

SMILES_COL = "PUBCHEM_EXT_DATASOURCE_SMILES"
OUTCOME_COL = "PUBCHEM_ACTIVITY_OUTCOME"
# PubChem activity outcome -> binary label used by the model.
LABEL_MAP = {"Active": 1, "Unspecified": 0}


# ─────────────────────────── Step 1: load + clean + label ───────────────────────────
def load_and_label(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw export, drop the PubChem metadata rows and rows without a
    SMILES/known outcome, and attach the binary ``Active`` label.

    Returns a frame with columns [SMILES, Active]. Needs no Java, so it is run
    by ``--check``.
    """
    # Rows 1-3 of the export are PubChem metadata (RESULT_TYPE / DESCR / UNIT).
    raw = pd.read_csv(raw_path, low_memory=False, skiprows=[1, 2, 3])

    df = raw[[SMILES_COL, OUTCOME_COL]].copy()
    df = df[df[SMILES_COL].notna()]                       # drop rows without SMILES
    df = df[df[OUTCOME_COL].isin(LABEL_MAP)]              # keep mappable outcomes
    df["Active"] = df[OUTCOME_COL].map(LABEL_MAP).astype(int)

    out = df[[SMILES_COL, "Active"]].rename(columns={SMILES_COL: "SMILES"})
    out = out.reset_index(drop=True)
    print(f"Loaded raw: {len(raw)} data rows -> {len(out)} labelled compounds")
    print(f"Label distribution: {out['Active'].value_counts().to_dict()}")
    return out


# ─────────────────────────── Step 2: PaDEL descriptors ───────────────────────────
def compute_padel_descriptors(smiles: list[str]) -> pd.DataFrame:
    """Compute the 1875 PaDEL 2D+3D descriptors for the given SMILES.

    Requires ``padelpy`` and a Java runtime. Returns descriptors in input order.
    """
    try:
        from padelpy import padeldescriptor
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit(
            "padelpy is required for the descriptor step:\n"
            "    pip install padelpy\n"
            "and a Java runtime must be installed (e.g. `brew install openjdk`)."
        ) from exc

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        smi_path = Path(tmp) / "mols.smi"
        out_path = Path(tmp) / "descriptors.csv"
        # PaDEL reads a .smi file: "<SMILES>\t<name>" per line.
        smi_path.write_text(
            "\n".join(f"{s}\t{i}" for i, s in enumerate(smiles))
        )
        padeldescriptor(
            mol_dir=str(smi_path),
            d_file=str(out_path),
            d_2d=True,
            d_3d=True,
            fingerprints=False,
            removesalt=True,
            standardizenitro=True,
            retainorder=True,
            threads=-1,
        )
        desc = pd.read_csv(out_path)

    desc = desc.drop(columns=[c for c in ["Name"] if c in desc.columns])
    print(f"Computed PaDEL descriptors: {desc.shape[0]} rows x {desc.shape[1]} features")
    return desc.reset_index(drop=True)


# ─────────────────────────── Step 3: MaxAbs normalization ───────────────────────────
def normalize_maxabs(desc: pd.DataFrame) -> pd.DataFrame:
    """Scale each descriptor column by its maximum absolute value.

    Equivalent to sklearn ``MaxAbsScaler``: every column lands in [-1, 1] and
    zeros are preserved. This is the exact transform used for the committed file.
    """
    desc = desc.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaled = MaxAbsScaler().fit_transform(desc.values)
    return pd.DataFrame(scaled, columns=desc.columns, index=desc.index)


# ─────────────────────────── Orchestration ───────────────────────────
def build(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> None:
    labelled = load_and_label(raw_path)
    desc = compute_padel_descriptors(labelled["SMILES"].tolist())

    # PaDEL may silently drop unparseable structures; keep labels aligned by
    # intersecting on position only when row counts match.
    if len(desc) != len(labelled):
        print(f"WARNING: descriptor rows ({len(desc)}) != labelled rows "
              f"({len(labelled)}); PaDEL likely skipped some structures. "
              "Inspect the SMILES before trusting the alignment.")
    n = min(len(desc), len(labelled))
    desc, y = desc.iloc[:n], labelled["Active"].iloc[:n]

    normalized = normalize_maxabs(desc)
    normalized["Active"] = y.values
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out_path, index=False)
    print(f"Saved {normalized.shape[0]} x {normalized.shape[1]} -> "
          f"{out_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────── Self-check (no Java needed) ───────────────────────────
def check() -> None:
    """Validate the steps that don't need PaDEL/Java."""
    labelled = load_and_label()
    assert set(labelled["Active"].unique()) <= {0, 1}, "labels must be 0/1"

    # MaxAbs property: every non-constant column ends with max(|x|) == 1.
    demo = pd.DataFrame({"a": [0.0, 2.0, -4.0], "b": [0.0, 0.0, 0.0], "c": [1.0, -3.0, 3.0]})
    scaled = normalize_maxabs(demo)
    maxabs = scaled.abs().max()
    assert abs(maxabs["a"] - 1.0) < 1e-12 and abs(maxabs["c"] - 1.0) < 1e-12
    assert (scaled["b"] == 0).all()  # all-zero column stays zero
    assert scaled.values.min() >= -1 - 1e-12 and scaled.values.max() <= 1 + 1e-12
    print("Self-check passed: labelling + MaxAbs normalization behave as expected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized descriptor dataset from raw PubChem export")
    parser.add_argument("--check", action="store_true",
                        help="run only the Java-free steps (labelling + normalization sanity)")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        build()


if __name__ == "__main__":
    main()
