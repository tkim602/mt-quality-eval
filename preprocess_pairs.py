import pandas as pd
import json
from pathlib import Path

SRC_XLSX = r"c:\Exception\308.AI_허브_데이터_활용을_위한_기계_번역앱_구축과_번역기_평가_및_신규_말뭉치_구축\01-1.정식개방데이터\Training\01.원천데이터\한영_번역기평가_컴퓨터과학_7726.xlsx"
N_SAMPLES = 10000

out_true  = Path("true_pairs.json")
out_false = Path("false_pairs.json")

df = pd.read_excel(SRC_XLSX, engine="openpyxl")
df = df[["source_sentence", "mt_sentence"]].dropna()

max_pairs = len(df) // 2
if N_SAMPLES > max_pairs:
    N_SAMPLES = max_pairs

if len(df) < 2 * N_SAMPLES:
    raise ValueError("Not enough rows to build the requested pairs")

df_sample = df.sample(n=2 * N_SAMPLES, random_state=42).reset_index(drop=True)
df_true   = df_sample.iloc[:N_SAMPLES].copy()
df_false  = df_sample.iloc[N_SAMPLES:].reset_index(drop=True).copy()

true_pairs = list(zip(df_true["mt_sentence"], df_true["source_sentence"]))

eng_shuffled = df_false["mt_sentence"].sample(frac=1, random_state=1).reset_index(drop=True)
for i in range(N_SAMPLES):
    if eng_shuffled.iat[i] == df_false["mt_sentence"].iat[i]:      
        j = (i + 1) % N_SAMPLES
        eng_shuffled.iat[i], eng_shuffled.iat[j] = eng_shuffled.iat[j], eng_shuffled.iat[i]

false_pairs = list(zip(eng_shuffled, df_false["source_sentence"]))

with out_true.open("w", encoding="utf-8") as f:
    json.dump(true_pairs, f, ensure_ascii=False, indent=2)
with out_false.open("w", encoding="utf-8") as f:
    json.dump(false_pairs, f, ensure_ascii=False, indent=2)

print(f"saved {len(true_pairs)} true pairs → {out_true}")
print(f"saved {len(false_pairs)} false pairs → {out_false}")
