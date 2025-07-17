import pandas as pd
import json

KO_FILE = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\ko_checker.json"
EN_FILE = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\en-US_checker.json"
OUT_CSV = "merged_output.csv"

with open(KO_FILE, "r", encoding="utf-8") as f:
    ko_data = json.load(f)

with open(EN_FILE, "r", encoding="utf-8") as f:
    en_data = json.load(f)

records = []
for key in ko_data.keys():
    records.append({
        "key":   key,
        "en":    en_data.get(key, ""),
        "ko":    ko_data[key],
        "label": "" 
    })

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"'{OUT_CSV}' created")
