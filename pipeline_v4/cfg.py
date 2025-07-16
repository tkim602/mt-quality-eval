from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent
OUT_DIR    = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

KO_JSON    = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\ko_checker.json"
EN_JSON    = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\en-US_checker.json"

LIMIT      = 100        

LABSE_MODEL   = "sentence-transformers/LaBSE"
COMET_CKPT    = "Unbabel/wmt22-cometkiwi-da"
GEMBA_MODEL   = "gpt-3.5-turbo-16k"            
APE_MODEL     = "gpt-4o-2024-11-20"          

TERMBASE = {
    "스레드": "thread",
    "이벤트": "event",
    "세션": "session",
    "취약점": "vulnerability",
    "취약점": "vulnerabilities"
}
GEMBA_PASS   = 80
GEMBA_BATCH  = 16     

COS_THR = {
    "short":      0.83,
    "medium":     0.85,
    "long":       0.87,
    "very_long":  0.88,
}

COMET_THR = 0.80   
