import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

JSON_PATH = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\filtered_ko_total_no_duplicates.json")

with JSON_PATH.open(encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)[["bucket", "cos", "comet"]]

bins   = np.arange(0.0, 1.01, 0.1)                       
labels = [f"{b:.1f}–{b+0.1:.1f}" for b in bins[:-1]]

df["cos_bin"]   = pd.cut(df["cos"],   bins=bins, labels=labels,
                         right=False, include_lowest=True)
df["comet_bin"] = pd.cut(df["comet"], bins=bins, labels=labels,
                         right=False, include_lowest=True)

hist = (
    pd.concat(
        [
            df["cos_bin"].value_counts(sort=False).rename("cos_count"),
            df["comet_bin"].value_counts(sort=False).rename("comet_count")
        ],
        axis=1
    )
    .fillna(0)
    .astype(int)
)

print("\nCos / COMET frequency by 0.1‑wide bins")
print(hist.to_string())

stats = (
    df.groupby("bucket")
      .agg(mean_cos   = ("cos",   "mean"),
           std_cos    = ("cos",   "std"),
           mean_comet = ("comet", "mean"),
           std_comet  = ("comet", "std"),
           count      = ("cos",   "size"))
      .round(4)
      .reset_index()
      .sort_values("bucket")
)

print("\nBucket‑level statistics")
print(stats.to_string(index=False))

corr_all = df["cos"].corr(df["comet"])
print(f"\nOverall Pearson correlation (cos, comet): {corr_all:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))

buckets = sorted(df["bucket"].unique())
colors  = plt.cm.get_cmap("tab10", len(buckets))

for i, b in enumerate(buckets):
    sub = df[df["bucket"] == b]
    ax.scatter(sub["cos"], sub["comet"],
               label=b, alpha=0.6, s=20,
               marker='o', edgecolors='none')

ax.scatter(stats["mean_cos"], stats["mean_comet"],
           color='black', marker='X', s=100, zorder=5)

ax.set_xlabel("Cosine Similarity")
ax.set_ylabel("COMET Score")
ax.set_title("Cosine vs COMET by Bucket (not noun phrases)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.legend(title="Bucket", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
