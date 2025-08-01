import json
import orjson
import numpy as np
import torch
import os
import hashlib
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from comet import load_from_checkpoint, download_model
from tqdm import tqdm
import cfg
import random
from validation import run_all_validations


def get_cache_key(texts: list) -> str:
    content = ''.join(texts)
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_cos_cache(cache_key: str) -> tuple:
    cache_dir = Path(cfg.OUT_DIR) / "cache"
    cache_file = cache_dir / f"koe5_{cache_key}.npz"
    
    if cache_file.exists():
        data = np.load(cache_file)
        return data['e_src'], data['e_mt'], True
    return None, None, False

def save_cos_cache(cache_key: str, e_src: np.ndarray, e_mt: np.ndarray):
    cache_dir = Path(cfg.OUT_DIR) / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"koe5_{cache_key}.npz"
    
    np.savez_compressed(cache_file, e_src=e_src, e_mt=e_mt)
    print(f"Saved KoE5 cache: {cache_file}")

def load_comet_cache(cache_key: str) -> tuple:
    cache_dir = Path(cfg.OUT_DIR) / "cache"
    cache_file = cache_dir / f"comet_{cache_key}.npy"
    
    if cache_file.exists():
        scores = np.load(cache_file)
        return scores, True
    return None, False

def save_comet_cache(cache_key: str, scores: np.ndarray):
    cache_dir = Path(cfg.OUT_DIR) / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"comet_{cache_key}.npy"
    
    np.save(cache_file, scores)
    print(f"Saved COMET cache: {cache_file}")

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


def detect_domain_from_filename(filename: str) -> str:
    """
    Detect domain from input filename pattern.
    Examples:
    - ko_checker.json -> sparrow
    - en_conversation.json -> conversation 
    """
    filename = filename.lower()
    
    domain_patterns = {
        'checker': 'sparrow',
        'sparrow': 'sparrow',
        'conversation': 'conversation',
        'academic': 'academic', 
        'news': 'news_article',
        'fiction': 'fiction',
        'poetry': 'poetry',
        'sns': 'sns'
    }
    
    for pattern, domain in domain_patterns.items():
        if pattern in filename:
            return domain
    
    return 'sparrow'  # default

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Detect domain from input filenames
    ko_domain = detect_domain_from_filename(cfg.KO_JSON)
    en_domain = detect_domain_from_filename(cfg.EN_JSON)
    
    # Use the domain that's not 'sparrow' if they differ, otherwise use either
    detected_domain = ko_domain if ko_domain != 'sparrow' else en_domain
    print(f"🎯 Detected domain: {detected_domain} (from {cfg.KO_JSON}, {cfg.EN_JSON})")
    
    # Store domain in environment for subsequent stages
    os.environ['PIPELINE_DOMAIN'] = detected_domain
    
    ko = json.load(open(cfg.KO_JSON, encoding="utf-8"))
    en = json.load(open(cfg.EN_JSON, encoding="utf-8"))

    all_keys = list(ko.keys())

    if cfg.LIMIT:
        if cfg.SEED is not None:
            random.seed(cfg.SEED)
            np.random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)
            print(f"🎲 Random seed set to {cfg.SEED} for reproducible sampling")
        keys = random.sample(all_keys, min(cfg.LIMIT, len(all_keys)))
        print(f"📊 Selected {len(keys)} samples from {len(all_keys)} total keys")
        print(f"🔍 First 5 selected keys: {keys[:5]}")
    else:
        keys = all_keys

    src = [ko[k] for k in keys]
    mt = [en.get(k, "") for k in keys]

    cache_key = get_cache_key(src + mt)
    print(f"Cache key: {cache_key}")

    force_fresh = os.getenv('FORCE_FRESH', 'false').lower() == 'true'
    if force_fresh:
        print("FORCE_FRESH mode: Computing fresh embeddings and scores...")

    e_src, e_mt, cache_hit = load_cos_cache(cache_key)
    
    if cache_hit and not force_fresh:
        print("OK: Loaded koe5 embeddings from cache")
        cos = (e_src * e_mt).sum(1)
    else:
        print("Computing koe5 embeddings...")
        koe5 = SentenceTransformer(cfg.COS_MODEL).to(device)
        
        e_src = koe5.encode(src, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        e_mt  = koe5.encode(mt , batch_size=128, normalize_embeddings=True, show_progress_bar=True)
        cos   = (e_src * e_mt).sum(1)
        
        if getattr(cfg, 'ENABLE_CACHING', True) and not force_fresh:
            save_cos_cache(cache_key, e_src, e_mt)
    
    com, comet_cache_hit = load_comet_cache(cache_key)
    
    if comet_cache_hit and not force_fresh:
        print("OK: Loaded COMET scores from cache")
    else:
        print("Computing COMET scores...")
        comet = load_from_checkpoint(download_model(cfg.COMET_CKPT)).to(device)
        
        scores = []
        BATCH = 64
        for i in tqdm(range(0, len(src), BATCH), desc="COMET"):
            batch_data = [{"src": s, "mt": t} for s, t in zip(src[i : i + BATCH], mt[i : i + BATCH])]
            scores.extend(comet.predict(batch_data, batch_size=BATCH)["scores"])
        com = np.array(scores, dtype=np.float32)
        
        if getattr(cfg, 'ENABLE_CACHING', True) and not force_fresh:
            save_comet_cache(cache_key, com)

    termbase = {}
    if hasattr(cfg, 'TERMBASE_PATH') and Path(cfg.TERMBASE_PATH).exists():
        with open(cfg.TERMBASE_PATH, 'r', encoding='utf-8') as f:
            termbase_data = json.load(f)
            if isinstance(termbase_data, list):
                termbase = {item['ko']: item['en-US'] for item in termbase_data if 'ko' in item and 'en-US' in item}
            else:
                termbase = termbase_data
    else:
        print("Warning: Termbase not found or path not configured in cfg.py. Skipping term consistency check.")


    records = []
    for k, s, m, c, q in zip(keys, src, mt, cos, com):
        validation_results = run_all_validations(s, m, termbase)
        
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
                "confidence": float(confidence),
                "string_type": cfg.get_string_type(k),
                "validation": validation_results,
                "domain": detected_domain,  # Add domain info to each record
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
