import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, roc_curve
)

with open("scores.json", encoding="utf-8") as f:
    data = json.load(f)

thresholds = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
rows, roc_rows = [], []

for model_name, m in data.items():
    s_high = np.asarray(m["scores_high"], dtype=np.float32)
    s_mid  = np.asarray(m["scores_mid"],  dtype=np.float32)
    s_low  = np.asarray(m["scores_low"],  dtype=np.float32)

    y_true  = np.concatenate([
        np.ones_like(s_high),               # high → 1
        np.zeros_like(s_mid),               # mid  → 0
        np.zeros_like(s_low)                # low  → 0
    ])
    y_score = np.concatenate([s_high, s_mid, s_low])

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_rows.append(dict(model=model_name, fpr=fpr, tpr=tpr, auc=auc))

    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        rows.append({
            "Model"      : model_name,
            "Threshold"  : t,
            "Precision"  : precision_score(y_true, y_pred),
            "Recall"     : recall_score(y_true, y_pred),
            "F1"         : f1_score(y_true, y_pred),
            "Accuracy"   : accuracy_score(y_true, y_pred),
            "AUC"        : auc,
            "MidPassRate": (s_mid >= t).mean()   
        })

plt.figure(figsize=(8, 5))
for r in roc_rows:
    plt.plot(r["fpr"], r["tpr"], label=f'{r["model"]} (AUC={r["auc"]:.3f})')
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve per Model")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

summary = pd.DataFrame(rows)
print(summary.to_string(index=False, float_format="{:.3f}".format))
