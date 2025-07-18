import pandas as pd, random, json, os
from pathlib import Path

PATH = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\filtered_ko_total_no_duplicates.json")
OUT_DIR = PATH.parent / "gap_samples"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_json(PATH)
df["abs_diff"] = (df["cos"] - df["comet"]).abs()
df["direction"] = df.apply(lambda r: "cos_gthan_comet" if r["cos"] > r["comet"] else "comet_gthan_cos", axis=1)

def sample_group(df_sub, n=20):
    return df_sub if len(df_sub) <= n else df_sub.sample(n, random_state=42)

ranges = [(0.10, 0.20), (0.20, 0.30), (0.30, 0.40)]

for lo, hi in ranges:
    df_range = df[(df["abs_diff"] >= lo) & (df["abs_diff"] < hi)]

    for direction in ["cos_gthan_comet", "comet_gthan_cos"]:        # ← 문자열 맞춤
        subset = df_range[df_range["direction"] == direction]
        sample = sample_group(subset, n=20)
        if sample.empty:
            continue

        out_path = OUT_DIR / f"gap_{lo:.2f}_{hi:.2f}_{direction}.json"
        sample.to_json(out_path, orient="records", force_ascii=False, indent=2)
        print(f"{len(sample):2} saved to {out_path}")

print("\ndone")
