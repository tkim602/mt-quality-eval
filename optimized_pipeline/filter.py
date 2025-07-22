import json
import orjson
import numpy as np
import torch
import os
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from comet import load_from_checkpoint, download_model
from tqdm import tqdm
import cfg
import random
from validation import run_all_validations


def bucket(sentence: str) -> str:
    n = len(sentence.strip())
    if n <= 9:
        return "very_short"            
    elif n <= 15:
        return "short"            
    elif n <= 45:
        return "medium"               
    elif n <= 100:
        return "long"                   
    else:
        return "very_long"            


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    labse = SentenceTransformer(cfg.LABSE_MODEL).to(device)
    comet = load_from_checkpoint(download_model(cfg.COMET_CKPT)).to(device)

    ko = json.load(open(cfg.KO_JSON, encoding="utf-8"))
    en = json.load(open(cfg.EN_JSON, encoding="utf-8"))

    all_keys = list(ko.keys())

    if cfg.LIMIT:
        random.seed(getattr(cfg, "seed", 42))
        keys = random.sample(all_keys, min(cfg.LIMIT, len(all_keys)))
    else:
        keys = all_keys


    src = [ko[k] for k in keys]

    # mt  = [en[k] for k in keys]
    mt = [en.get(k, "") for k in keys]

    e_src = labse.encode(src, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    e_mt  = labse.encode(mt , batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    cos   = (e_src * e_mt).sum(1)

    scores = []
    BATCH = 64
    for i in tqdm(range(0, len(src), BATCH), desc="COMET"):
        batch_data = [{"src": s, "mt": t} for s, t in zip(src[i : i + BATCH], mt[i : i + BATCH])]
        scores.extend(comet.predict(batch_data, batch_size=BATCH)["scores"])
    com = np.array(scores, dtype=np.float32)

    # Load termbase if available
    termbase = {}
    if hasattr(cfg, 'TERMBASE_PATH') and Path(cfg.TERMBASE_PATH).exists():
        with open(cfg.TERMBASE_PATH, 'r', encoding='utf-8') as f:
            termbase_data = json.load(f)
            # Convert from array of objects to simple key-value dictionary
            if isinstance(termbase_data, list):
                termbase = {item['ko']: item['en-US'] for item in termbase_data if 'ko' in item and 'en-US' in item}
            else:
                termbase = termbase_data
    else:
        print("Warning: Termbase not found or path not configured in cfg.py. Skipping term consistency check.")


    records = []
    for k, s, m, c, q in zip(keys, src, mt, cos, com):
        validation_results = run_all_validations(s, m, termbase)
        
        # Apply enhanced quality decision with confidence scoring
        tag, passed, failed, confidence = cfg.make_quality_decision_enhanced(c, q, 0, bucket(s), k)
        
        records.append(
            {
                "key":    k,
                "src":    s,
                "mt":     m,
                "cos":    float(c),
                "comet":  float(q),
                "bucket": bucket(s),
                "tag": tag,
                "passed_checks": passed,
                "failed_checks": failed,
                "confidence": confidence,
                "string_type": cfg.get_string_type(k),
                "validation": validation_results,
            }
        )

    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        output_path = Path(run_dir) / cfg.FILTER_OUTPUT_FILENAME
        output_path.parent.mkdir(exist_ok=True, parents=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(cfg.OUT_DIR).mkdir(exist_ok=True, parents=True)
        output_path = cfg.OUT_DIR / f"filtered_{timestamp}.json"
    
    output_path.write_bytes(
        orjson.dumps(records, option=orjson.OPT_INDENT_2)
    )
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
