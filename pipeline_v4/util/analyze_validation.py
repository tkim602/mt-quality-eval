import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

df = pd.read_json("out/validation.json")

bucket_order = ["short", "medium", "long", "very_long"]
df["bucket"] = pd.Categorical(df["bucket"], categories=bucket_order, ordered=True)

metrics = ["cos", "comet", "gemba"]
corr_records = []
for bucket in bucket_order + ["all"]:
    subset = df if bucket == "all" else df[df["bucket"] == bucket]
    for m in metrics:
        pear = subset[m].corr(subset["human_mqm"], method="pearson")
        spear = subset[m].corr(subset["human_mqm"], method="spearman")
        corr_records.append({
            "bucket": bucket,
            "metric": m,
            "pearson": round(pear,4),
            "spearman": round(spear,4),
        })
corr_df = pd.DataFrame(corr_records)

MQM_PASS = 3
df["label"] = (df["human_mqm"] <= MQM_PASS).astype(int)

thr_ranges = {
    "cos":   [x/100 for x in range(50,96)],   # 0.50–0.95
    "comet": [x/100 for x in range(60,96)],   # 0.60–0.95
    "gemba": [x/10  for x in range(6,101)],   # 0.6–10.0 (matches validation scale)
}
thr_records = []
for bucket in bucket_order + ["all"]:
    subset = df if bucket == "all" else df[df["bucket"] == bucket]
    for m in metrics:
        best_thr, best_f1 = None, -1.0
        for thr in thr_ranges[m]:
            preds = (subset[m] >= thr).astype(int)
            f1 = f1_score(subset["label"], preds)
            if f1 > best_f1:
                best_thr, best_f1 = thr, f1
        thr_records.append({
            "bucket": bucket,
            "metric": m,
            "best_threshold": best_thr,
            "best_f1": round(best_f1,4),
        })
thr_df = pd.DataFrame(thr_records)
thr_df["bucket"] = pd.Categorical(thr_df["bucket"], categories=bucket_order+["all"], ordered=True)

print("\n=== Correlations (short→very_long, then all) ===")
print(corr_df.to_string(index=False))
print("\n=== Best Thresholds & F1 by Bucket ===")
print(thr_df.to_string(index=False))

for kind in ["pearson","spearman"]:
    plt.figure()
    for m in metrics:
        data = corr_df[corr_df["metric"] == m]
        plt.plot(data["bucket"], data[kind], marker='o', label=m)
    plt.title(f"{kind.title()} Correlation with human_mqm")
    plt.xlabel("Bucket")
    plt.ylabel(f"{kind.title()} ρ")
    plt.legend()
    plt.tight_layout()

plt.figure()
for m in metrics:
    data = thr_df[thr_df["metric"] == m]
    plt.plot(data["bucket"], data["best_threshold"], marker='o', label=m)
plt.title("Optimal Thresholds by Bucket")
plt.xlabel("Bucket")
plt.ylabel("Threshold")
plt.legend()
plt.tight_layout()

plt.show()

