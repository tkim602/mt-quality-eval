import json, orjson, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from comet import load_from_checkpoint, download_model
from tqdm import tqdm
import cfg


def bucket(sentence: str) -> str:
    n = len(sentence.split())
    if n <= 4:
        return "short"
    elif n <= 8:
        return "medium"
    elif n <= 15:
        return "long"
    else:
        return "very_long"


def main() -> None:
    labse = SentenceTransformer(cfg.LABSE_MODEL)
    comet = load_from_checkpoint(download_model(cfg.COMET_CKPT))

    ko = json.load(open(cfg.KO_JSON, encoding="utf-8"))
    en = json.load(open(cfg.EN_JSON, encoding="utf-8"))
    keys = list(ko.keys())[: cfg.LIMIT] if cfg.LIMIT else list(ko.keys())

    src = [ko[k] for k in keys]
    mt  = [en[k] for k in keys]

    e_src = labse.encode(src, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    e_mt  = labse.encode(mt , batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    cos   = (e_src * e_mt).sum(1)

    scores = []
    BATCH = 64
    for i in tqdm(range(0, len(src), BATCH), desc="COMET"):
        batch_data = [{"src": s, "mt": t} for s, t in zip(src[i : i + BATCH], mt[i : i + BATCH])]
        scores.extend(comet.predict(batch_data, batch_size=BATCH)["scores"])
    com = np.array(scores, dtype=np.float32)

    records = []
    for k, s, m, c, q in zip(keys, src, mt, cos, com):
        records.append(
            {
                "key":    k,
                "src":    s,
                "mt":     m,
                "cos":    float(c),
                "comet":  float(q),
                "bucket": bucket(s),
            }
        )

    Path(cfg.OUT_DIR).mkdir(exist_ok=True, parents=True)
    (cfg.OUT_DIR / "filtered.json").write_bytes(
        orjson.dumps(records, option=orjson.OPT_INDENT_2)
    )

if __name__ == "__main__":
    main()
