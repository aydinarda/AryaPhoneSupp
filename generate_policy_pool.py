#!/usr/bin/env python3
"""Generate a large pool of candidate government policies and export to Excel.

Default = full Cartesian enumeration:
- 6 multiplier knobs ∈ {1, 5, 10}  (Low/Mid/High)
- 2 binary penalties ∈ {0, 1}      (No/Yes)

This yields 3^6 * 2^2 = 2916 unique policies.

IMPORTANT:
- Sheet 'policies' contains ONLY numeric columns + policy_id so you can safely load it back into the app.
- Sheet 'readable' contains extra human-friendly label columns (Low/Mid/High, Yes/No).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import List, Tuple

import pandas as pd


MULT_KEYS = [
    "env_mult",
    "social_mult",
    "cost_mult",
    "strategic_mult",
    "improvement_mult",
    "low_quality_mult",
]
PEN_KEYS = ["child_labor_penalty", "banned_chem_penalty"]

DEFAULT_LEVELS = [1, 5, 10]   # Low/Mid/High
DEFAULT_PENALTY_VALS = [0, 1] # No/Yes


def _parse_int_list(s: str) -> List[int]:
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return [int(p) for p in parts]


def generate_pool(levels: List[int], penalty_vals: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for mult_vals in product(levels, repeat=len(MULT_KEYS)):
        for pen_vals in product(penalty_vals, repeat=len(PEN_KEYS)):
            d = dict(zip(MULT_KEYS, mult_vals))
            d.update(dict(zip(PEN_KEYS, pen_vals)))

            payload = json.dumps(d, sort_keys=True)
            d["policy_id"] = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
            rows.append(d)

    df = pd.DataFrame(rows)

    # numeric-only (safe to load back into Policy object)
    df_num = df[["policy_id"] + MULT_KEYS + PEN_KEYS].copy()

    # readable view
    level_name = {1: "Low", 5: "Mid", 10: "High"}
    df_read = df_num.copy()
    for k in MULT_KEYS:
        df_read[k + "_level"] = df_read[k].map(level_name).fillna(df_read[k].astype(str))
    for k in PEN_KEYS:
        df_read[k + "_yn"] = df_read[k].map({0: "No", 1: "Yes"}).fillna(df_read[k].astype(str))

    return df_num, df_read


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="policy_pool.xlsx", help="Output Excel path")
    ap.add_argument(
        "--levels",
        default=",".join(map(str, DEFAULT_LEVELS)),
        help="Comma/space-separated multiplier levels, e.g. '1,5,10' or '1 5 10'",
    )
    ap.add_argument(
        "--penalties",
        default=",".join(map(str, DEFAULT_PENALTY_VALS)),
        help="Comma/space-separated penalty values, e.g. '0,1'",
    )
    args = ap.parse_args()

    levels = _parse_int_list(args.levels)
    penalty_vals = _parse_int_list(args.penalties)

    df_num, df_read = generate_pool(levels, penalty_vals)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_num.to_excel(writer, sheet_name="policies", index=False)
        df_read.to_excel(writer, sheet_name="readable", index=False)

    print(f"Wrote {len(df_num)} policies -> {out_path.resolve()}")
    print("Numeric columns:", ", ".join(df_num.columns))


if __name__ == "__main__":
    main()
