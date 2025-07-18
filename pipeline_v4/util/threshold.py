import json, pandas as pd
from pathlib import Path

JSON_PATH = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\filtered_ko_total_no_duplicates.json")
TOP_PCT   = 0.50
OUT_DIR   = JSON_PATH.parent

SUMMARY_FILE = OUT_DIR / "threshold_summary.json"
AND_FILE     = OUT_DIR / "and_pass_records.json"
OR_FILE      = OUT_DIR / "or_pass_records.json"  

df = pd.read_json(JSON_PATH)

cos_thr   = df["cos"  ].quantile(1 - TOP_PCT)
comet_thr = df["comet"].quantile(1 - TOP_PCT)

m_cos   = df["cos"]   >= cos_thr
m_comet = df["comet"] >= comet_thr

total      = len(df)
and_pass   = df[m_cos & m_comet]
or_pass_df = df[m_cos | m_comet]

summary = {
    "top_pct": TOP_PCT,
    "thresholds": {"cos": round(cos_thr, 4), "comet": round(comet_thr, 4)},
    "counts": {
        "total": total,
        "and_pass": len(and_pass),
        "or_pass": len(or_pass_df)
    }
}
SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"summary → {SUMMARY_FILE}")

AND_FILE.write_text(
    and_pass.to_json(orient="records", force_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"AND passed {len(and_pass):,} → {AND_FILE}")

OR_FILE.write_text(
    or_pass_df.to_json(orient="records", force_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"OR passed {len(or_pass_df):,} → {OR_FILE}")
