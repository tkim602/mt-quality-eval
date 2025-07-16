import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 1) 데이터 로드 & 스케일링
df = pd.read_json("out/validation.json")
df["gemba_norm"] = df["gemba"] / 10.0

# 2) composite score 계산(예시 가중치)
w_cos, w_comet, w_gemba = 0.2, 0.4, 0.4
df["score"] = w_cos * df["cos"] + w_comet * df["comet"] + w_gemba * df["gemba_norm"]

# 3) human_mqm 후보 범위: 5–95 백분위 사이
h5, h95 = np.percentile(df["human_mqm"], [5, 95])
human_thrs = np.unique(df["human_mqm"])
human_thrs = human_thrs[(human_thrs >= h5) & (human_thrs <= h95)]

# 4) composite score 임계값 후보: 5–95 백분위 사이
s5, s95 = np.percentile(df["score"], [5, 95])
metric_thrs = np.linspace(s5, s95, 100)

best = {"h_thr": None, "m_thr": None, "f1": 0}
records = []

for h_thr in human_thrs:
    labels = (df["human_mqm"] <= h_thr).astype(int)
    prop = labels.mean()
    if prop < 0.05 or prop > 0.95:
        continue  # 극단 레이블 배제
    
    for m_thr in metric_thrs:
        preds = (df["score"] >= m_thr).astype(int)
        prop_p = preds.mean()
        if prop_p < 0.05 or prop_p > 0.95:
            continue  # 극단 예측 배제
        
        f1 = f1_score(labels, preds)
        records.append((h_thr, m_thr, f1))
        if f1 > best["f1"]:
            best.update({"h_thr": h_thr, "m_thr": m_thr, "f1": f1})

# 결과 출력
print("=== 최적 human_mqm & composite 임계값 (비극단) ===")
print(f"human_mqm ≤ {best['h_thr']} → 레이블 분리")
print(f"composite ≥ {best['m_thr']:.4f} → 예측 Good")
print(f"Best F1: {best['f1']:.4f}")

# 시각화: human_thr vs 최고 F1
grouped = {}
for h_thr, m_thr, f1 in records:
    grouped.setdefault(h_thr, []).append(f1)
hrs = sorted(grouped)
best_f1s = [max(grouped[h]) for h in hrs]

plt.figure(figsize=(6,4))
plt.plot(hrs, best_f1s, marker='o')
plt.axvline(best["h_thr"], color='red', linestyle='--', label=f"Best h={best['h_thr']}")
plt.title("human_mqm Threshold vs Best F1 (5-95% 제한)")
plt.xlabel("human_mqm Threshold")
plt.ylabel("Best F1")
plt.legend()
plt.tight_layout()
plt.show()
