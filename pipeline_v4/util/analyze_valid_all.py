#!/usr/bin/env python3
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1) validation.json 로드
df = pd.read_json("out/validation.json")

# 2) 전체 상관계수 계산
metrics = ["cos", "comet", "gemba"]
corrs = {"metric": [], "pearson": [], "spearman": []}
for m in metrics:
    corr_pear = df[m].corr(df["human_mqm"], method="pearson")
    corr_spear = df[m].corr(df["human_mqm"], method="spearman")
    corrs["metric"].append(m)
    corrs["pearson"].append(round(corr_pear,4))
    corrs["spearman"].append(round(corr_spear,4))
corr_df = pd.DataFrame(corrs)

# 3) human_mqm ≤15 → PASS(1), else FAIL(0)
df["label"] = (df["human_mqm"] <= 5).astype(int)

# 4) 전체 최적 임계값 & F1 탐색
thr_ranges = {
    "cos":   [x/100 for x in range(50,96)],   # 0.50–0.95
    "comet": [x/100 for x in range(60,96)],   # 0.60–0.95
    "gemba": [x/10  for x in range(6,101)],   # 0.6–10.0 (validation 스케일)
}
thr_records = {"metric": [], "best_threshold": [], "best_f1": []}
for m in metrics:
    best_thr, best_f1 = None, -1
    for thr in thr_ranges[m]:
        preds = (df[m] >= thr).astype(int)
        f1 = f1_score(df["label"], preds)
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    thr_records["metric"].append(m)
    thr_records["best_threshold"].append(best_thr)
    thr_records["best_f1"].append(round(best_f1,4))
thr_df = pd.DataFrame(thr_records)

# 5) 결과 출력
print("=== 전체 상관계수 ===")
print(corr_df.to_string(index=False))
print("\n=== 전체 최적 Threshold & F1 ===")
print(thr_df.to_string(index=False))

# 6) 시각화
# 상관계수 바 차트
plt.figure(figsize=(6,4))
x = range(len(metrics))
plt.bar(x, corr_df["pearson"], width=0.4, label="Pearson", align="center")
plt.bar([i+0.4 for i in x], corr_df["spearman"], width=0.4, label="Spearman", align="center")
plt.xticks([i+0.2 for i in x], metrics)
plt.ylabel("Correlation ρ")
plt.title("Overall Correlation with human_mqm")
plt.legend()
plt.tight_layout()

# Threshold & F1 라인 차트
fig, ax1 = plt.subplots(figsize=(6,4))
ax1.plot(metrics, thr_df["best_threshold"], marker='o', label="Threshold")
ax1.set_ylabel("Threshold")
ax2 = ax1.twinx()
ax2.plot(metrics, thr_df["best_f1"], marker='s', color='gray', label="F1")
ax2.set_ylabel("Best F1")
ax1.set_title("Overall Optimal Thresholds & F1")
ax1.set_xlabel("Metric")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()

plt.show()
