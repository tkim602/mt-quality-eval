from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import time

PAIR_DIR = Path("data_pairs_by_gpt")
BATCH_SIZE = 64
models = [
    # "sentence-transformers/LaBSE",
    # "nlpai-lab/KoE5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]

# def cos_sim(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_pairs(fname, limit=None):
    pairs = [tuple(pair) for pair in json.loads((PAIR_DIR / fname).read_text(encoding="utf-8"))]
    if limit:
        pairs = pairs[:limit]
    return pairs

# true_pairs = load_pairs("true_pairs.json")
# false_pairs = load_pairs("false_pairs_gpt.json")
# true_en, true_ko = zip(*true_pairs)
# false_en, false_ko = zip(*false_pairs)

## 너무 오래걸려서 limit 추가 
low_pairs = load_pairs("low_pairs.json", limit=500)
mid_pairs = load_pairs("mid_pairs.json", limit=500)
high_pairs = load_pairs("high_pairs.json", limit=500)

low_en, low_ko = zip(*low_pairs)
mid_en, mid_ko = zip(*mid_pairs)
high_en, high_ko = zip(*high_pairs)


results = {}

for name in models:
    model = SentenceTransformer(name, trust_remote_code=True)
    start_time = time.time()
    emb_en = model.encode(low_en, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    emb_ko = model.encode(low_ko, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    scores_low = (emb_en * emb_ko).sum(axis=1)
    emb_en = model.encode(mid_en, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    emb_ko = model.encode(mid_ko, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    scores_mid = (emb_en * emb_ko).sum(axis=1)
    emb_en = model.encode(high_en, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    emb_ko = model.encode(high_ko, batch_size=BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    scores_high = (emb_en * emb_ko).sum(axis=1)

    mean_low = float(scores_low.mean())
    mean_mid = float(scores_mid.mean())
    mean_high = float(scores_high.mean())

    execution_t = time.time() - start_time
    results[name] = dict(
        scores_low=scores_low.tolist(),
        scores_mid=scores_mid.tolist(),
        scores_high=scores_high.tolist(),        
        mean_low=float(scores_low.mean()),
        mean_mid=float(scores_mid.mean()),
        mean_high=float(scores_high.mean())
    )
    print(f"model: {name}\nmean_low: {mean_low}\nmean_mid: {mean_mid}\nmean_high: {mean_high}\nexecution_time: {execution_t}\n")
    del model, emb_en, emb_ko, scores_low, scores_mid, scores_high
    
out_file = Path("scores11.json")
with out_file.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)