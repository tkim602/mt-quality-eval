from pathlib import Path
import json, time, os
from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

PAIR_DIR   = Path("data_pairs_by_gpt")
BATCH_SIZE = 64
MODEL_ID   = "Unbabel/wmt22-cometkiwi-da"

def load_pairs(name, n=500):
    pairs = json.loads((PAIR_DIR / name).read_text(encoding="utf-8"))
    pairs = pairs[:n]
    return [{"src": ko, "mt": en} for en, ko in pairs]

low_pairs  = load_pairs("low_pairs.json")
mid_pairs  = load_pairs("mid_pairs.json")
high_pairs = load_pairs("high_pairs.json")

model = load_from_checkpoint(download_model(MODEL_ID))

start = time.time()
scores_low  = model.predict(low_pairs,  batch_size=BATCH_SIZE, gpus=0)["scores"]
scores_mid  = model.predict(mid_pairs,  batch_size=BATCH_SIZE, gpus=0)["scores"]
scores_high = model.predict(high_pairs, batch_size=BATCH_SIZE, gpus=0)["scores"]
elapsed = time.time() - start

res = {
    MODEL_ID: {
        "scores_low":  scores_low,
        "scores_mid":  scores_mid,
        "scores_high": scores_high,
        "mean_low":  float(sum(scores_low)  / len(scores_low)),
        "mean_mid":  float(sum(scores_mid)  / len(scores_mid)),
        "mean_high": float(sum(scores_high) / len(scores_high)),
        "execution_time": elapsed
    }
}

out = Path("comet_scores.json")
out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

print(
    f"model: {MODEL_ID}\n"
    f"mean_low:  {res[MODEL_ID]['mean_low']}\n"
    f"mean_mid:  {res[MODEL_ID]['mean_mid']}\n"
    f"mean_high: {res[MODEL_ID]['mean_high']}\n"
    f"execution_time: {elapsed}"
)