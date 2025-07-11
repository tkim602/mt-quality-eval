#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import pandas as pd

DEFAULT_INPUT  = Path(r"c:\Exception\term_base_2024.xlsx")
DEFAULT_OUTPUT = Path("samples/term_base_map.json")

def main(input_xlsx: Path = DEFAULT_INPUT, output_json: Path = DEFAULT_OUTPUT):
    df = pd.read_excel(input_xlsx, engine="openpyxl", keep_default_na=False)

    for col in ("en-US", "ko"):
        if col not in df.columns:
            raise KeyError(f"no '{col}' column exists.")

    df = df.dropna(subset=["ko"])
    df = df[df["ko"].astype(str).str.strip().ne("")] 

    mask = df["ko"].astype(str).str.len() <= 10
    df_short = df.loc[mask, ["en-US", "ko"]]

    df_short = df_short[df_short["en-US"] != df_short["ko"]]

    records = df_short.to_dict(orient="records")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"done: {len(records)} records → {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter XLSX → JSON (ko length ≤ 10, no NaNs)")
    parser.add_argument("xls", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.xls, args.out)
