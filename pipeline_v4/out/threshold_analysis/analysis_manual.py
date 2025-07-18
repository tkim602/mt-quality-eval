import pandas as pd, numpy as np
df = pd.read_json("sample_tagged3.json")

for s in ["AND","OR","FAIL"]:
    part = df[df["stratum"] == s]
    tp = (part["manual"] == "Pass").sum()
    fp = (part["manual"] == "Fail").sum()
    prec = tp / (tp + fp) if (tp+fp) else 0
    print(f"{s}: precision={prec:.3%}  (TP={tp}, FP={fp})")
