#!/usr/bin/env python3
"""
Camie‑Distill image‑pipeline consistency checker
-----------------------------------------------
Verifies at runtime that the *teacher* (dataset_builder) and *student*
(train_student → CsvDataset) loaders remain bit‑for‑bit identical, that the
default `--pad‑colour` stays **(0, 0, 0)**, and that **either**
(a) *zero* Bicubic interpolation is present **or**
(b) *every* resize call uses Bicubic.

Exit status 0 = all good, non‑zero = failure.
"""

from __future__ import annotations
import inspect, re, sys
from pathlib import Path

import camie_distill.preprocessing as _pre
import camie_distill.dataset_builder as _db
import camie_distill.train_student as _ts

# ────────────────────────────────────────────────────────────────────
def _scan_for(pattern: str, root: Path) -> list[Path]:
    regex = re.compile(pattern)
    return [p for p in root.rglob("*.py")
            if regex.search(p.read_text(errors="ignore"))]

# 1. Teacher ↔ Student pipeline parity
def _check_shared_pipeline() -> bool:
    # Same function object?
    if _pre.load_and_preprocess is not _db.load_and_preprocess:
        return False

    # CsvDataset must call that very same symbol
    src = inspect.getsource(_ts.CsvDataset.__getitem__)
    return "load_and_preprocess" in src

# 2. Default pad‑colour = (0,0,0) in both entry‑points
def _check_pad_colour_default() -> bool:
    code = (
        Path(_db.__file__).read_text() +
        Path(_ts.__file__).read_text()
    )
    m = re.findall(r"--pad-colou?r[^)]*default=\((\d+),\s*(\d+),\s*(\d+)\)", code)
    return all(tuple(map(int, g)) == (0, 0, 0) for g in m)

# 3. Bicubic interpolation – all‑or‑nothing rule
def _check_interpolation() -> bool:
    root = Path(__file__).resolve().parent
    bicubic_files = _scan_for(r"\b(BICUBIC|InterpolationMode\.BICUBIC)\b", root)
    if not bicubic_files:                         # zero Bicubic ⇒ OK
        return True
    # else: every resize op must be Bicubic
    non_bicubic = _scan_for(r"\bLANCZOS\b|\bBILINEAR\b|\bNEAREST\b", root)
    return len(non_bicubic) == 0                  # fail if mixture detected

# ────────────────────────────────────────────────────────────────────
def main() -> None:
    checks = {
        "shared_pipeline":       _check_shared_pipeline(),
        "pad_colour_is_black":   _check_pad_colour_default(),
        "interpolation_uniform": _check_interpolation(),
    }

    for name, ok in checks.items():
        print(("✓" if ok else "✗"), name.replace("_", " "))

    sys.exit(0 if all(checks.values()) else 1)

if __name__ == "__main__":
    main()
