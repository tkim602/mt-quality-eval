import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score

df = pd.read_json("sample_tagged1.json")  # cos, comet, human_pass(1/0)

grid = np.arange(0.95, 0.80, -0.01)   # 0.95 → 0.81
best = None

for c_thr in grid:               # cos
    for m_thr in grid:           # comet
        pred = (df["cos"] >= c_thr) & (df["comet"] >= m_thr)
        prec = precision_score(df["human_pass"], pred)
        rec  = recall_score(df["human_pass"], pred)
        if prec >= 0.95:
            best = (c_thr, m_thr, prec, rec)
            break
    if best: break

print("best:", best)
