import json, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

JSON_PATH = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\filtered_ko_total_no_duplicates.json")

df = pd.read_json(JSON_PATH)

assert "bucket" in df.columns, "no bucket in json file"

TOP_PCT = 0.333
q_val   = 1 - TOP_PCT

cuts = (df.groupby("bucket")
          .agg(cos_cut   =("cos",   lambda s: s.quantile(q_val)),
               comet_cut =("comet", lambda s: s.quantile(q_val)))
          .to_dict("index"))

for metric in ["cos", "comet"]:
    color = "#4C72B0" if metric == "cos" else "#55A868"

    for b in ["very_short", "short", "medium", "long", "very_long"]:
        sub   = df[df["bucket"] == b][metric]
        cut   = cuts[b][f"{metric}_cut"]
        left  = (sub < cut).sum()
        right = (sub >= cut).sum()

        plt.figure(figsize=(6,4))
        plt.hist(sub, bins=30, density=False, color=color, alpha=0.65, edgecolor="white")
        plt.axvline(cut, color="red", lw=2.5, ls="--", zorder=10,
                    label=f"cut {cut:.3f}")
        # 텍스트 annotate
        plt.text(0.02, 0.95, f"< cut: {left}\n>= cut: {right}",
                 transform=plt.gca().transAxes,
                 verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        plt.title(f"{metric.upper()} – {b.replace('_',' ').title()}")
        plt.xlabel(metric); plt.ylabel("count"); plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"cut {cut:.3f}")
