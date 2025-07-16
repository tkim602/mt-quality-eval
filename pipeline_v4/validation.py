import pandas as pd
from sklearn.metrics import f1_score

df = pd.read_json("out/validation.json")

MQM_PASS = 7
df["label"] = (df["human_mqm"] <= MQM_PASS).astype(int)

weights = {"cos": 0.1, "comet": 0.5, "gemba": 0.4}

df["composite"] = (
    df["cos"]   * weights["cos"] +
    df["comet"] * weights["comet"] +
    df["gemba"] * weights["gemba"]
)

THR = 0.7
df["pred"] = (df["composite"] >= THR).astype(int)

print("Fixed weights:", weights, "Threshold:", THR)
print("F1-score:", f1_score(df["label"], df["pred"]))
