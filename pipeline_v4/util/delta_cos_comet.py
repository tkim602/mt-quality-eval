import json, math
from pathlib import Path

JSON_PATH = Path(r"c:\Users\tkim602_global\Desktop\mt_eval\pipeline_v4\out\filtered_ko_total_no_duplicates.json")
THR       = 0.10   
OUT       = JSON_PATH.with_stem(JSON_PATH.stem + "_delta_over_0.1")

def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    selected = [
        {**rec, "delta": abs(rec["cos"] - rec["comet"])}
        for rec in data
        if abs(rec["cos"] - rec["comet"]) >= THR
    ]

    OUT.with_suffix(".json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{len(selected)} where abs(cos-comet) >= 0.1 {THR}")
    for rec in selected:
        print(f"{rec['key']}: cos={rec['cos']:.4f}, comet={rec['comet']:.4f}, Δ={rec['delta']:.4f}")

if __name__ == "__main__":
    main()
