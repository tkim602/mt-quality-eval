from pathlib import Path
import json, numpy as np, orjson   
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm 

from comet import download_model, load_from_checkpoint
from dotenv import load_dotenv
from huggingface_hub import login
import os

load_dotenv() 
login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

KO_JSON = Path("../samples/ko_checker.json")
EN_JSON = Path("../samples/en-US_checker.json")
EMBED_MODEL = "sentence-transformers/LaBSE"
MODEL_COMET = "Unbabel/wmt22-cometkiwi-da"
BATCH_SIZE = 128         
OUT_JSON  = Path("out/tagged.json")

THRESHOLD = 0.83

load  = lambda p: json.loads(p.read_text(encoding="utf-8"))
ko_d  = load(KO_JSON)
en_d  = load(EN_JSON)

common = ko_d.keys() & en_d.keys()      
if not common:
    raise ValueError("No overlapping keys.")

def pick(dct, k):
    v = dct[k]
    return v if isinstance(v, str) else v.get("sentence", "")

limit = 3000
keys = list(common)

if limit:
    keys = keys[:limit]

srcs = [pick(ko_d, k) for k in keys]
mts  = [pick(en_d, k) for k in keys]

model = SentenceTransformer(EMBED_MODEL)
emb_ko = model.encode(srcs, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
emb_en = model.encode(mts,  batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
cos_sim  = (emb_ko * emb_en).sum(axis=1)     

pairs = [{"src": src, "mt": mt} for src, mt in zip(srcs, mts)]
comet = load_from_checkpoint(download_model(MODEL_COMET))
comet_score = comet.predict(pairs, batch_size=BATCH_SIZE)["scores"]

passes = cos_sim >= THRESHOLD
tagged = [
    {
        "key": key,
        "src": src,
        "mt": mt,
        "cosine_score": float(cos_score),
        "comet_score": float(cmt_score),
        "labse_tag": "pass" if p else "fail",
    }
    for key, src, mt, cos_score, cmt_score, p in zip(keys, srcs, mts, cos_sim, comet_score, passes)
]

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_bytes(orjson.dumps(tagged, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))

print(f"(pass={passes.sum()}, fail={len(passes)-passes.sum()})")
