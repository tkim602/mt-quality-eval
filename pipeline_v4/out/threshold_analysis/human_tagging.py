import pandas as pd, numpy as np
from pathlib import Path

JSON     = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\filtered_ko_total_no_duplicates.json")
TOP_PCT  = 0.30 
COUNTS   = {"AND":30, "OR":30, "FAIL":30}
OUT_FILE = Path("sample_for_manual_tagging3.json")

df = pd.read_json(JSON)

cos_thr = df["cos"  ].quantile(1 - TOP_PCT)
com_thr = df["comet"].quantile(1 - TOP_PCT)

m_cos   = df["cos"]   >= cos_thr
m_comet = df["comet"] >= com_thr
df["stratum"] = np.select(
    [m_cos & m_comet,  m_cos ^ m_comet],
    ["AND",           "OR"],
    default="FAIL"
)

samples = []
for s, k in COUNTS.items():
    sub = df[df["stratum"] == s]
    if len(sub) < k:
        raise ValueError(f"{s} 구간에 {k}줄이 부족합니다 (현재 {len(sub)})")
    samples.append(sub.sample(n=k))

sample_df = pd.concat(samples).sample(frac=1)
sample_df["manual"] = ""

sample_df[["key","src","mt","stratum","manual"]].to_json(
    OUT_FILE, orient="records", force_ascii=False, indent=2
)
print(f"saved → {OUT_FILE}")
print(sample_df["stratum"].value_counts())
