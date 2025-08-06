from pathlib import Path
import os
import numpy as np
from typing import Dict, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent

USE_Q_SCORE = True

Q_SCORE_WEIGHTS = {
    'cos': 0.20,
    'comet': 0.30,
    'gemba': 0.50
}

Q_SCORE_QUANTILES = [-float('inf'), -0.6, -0.1, 0.25, 0.6, float('inf')]

Q_TEMP_GRADES = ['Fail_temp', 'Soft_temp', 'Mid_temp', 'Good_temp', 'Strict_temp']

Q_FINAL_GRADE_MAPPING = {
    'Fail_temp': 'fail',
    'Soft_temp': 'soft_pass',
    'Mid_temp': 'soft_pass', 
    'Good_temp': 'strict_pass',
    'Strict_temp': 'strict_pass'
}

BUCKET_SAFETY_BELT = {
    'enabled': True,
    'gemba_threshold': 50, 
    'apply_to_grades': ['Good_temp', 'Strict_temp'], 
    'downgrade_to': 'Soft_temp'
}

OUTPUT_DIR = "out"
OUT_DIR = Path(OUTPUT_DIR)
OUT_DIR.mkdir(exist_ok=True)

KO_JSON = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\ko_checker_dedup.json"
EN_JSON = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\en-US_checker.json"

# KO_JSON = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\ko-KR.json"
# EN_JSON = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\ja-JP.json"

LIMIT = 5 
import time
# SEED = int(time.time()) % 10000  
SEED = 8084

COS_MODEL = "sentence-transformers/LaBSE"
COMET_CKPT = "Unbabel/wmt22-cometkiwi-da"
GEMBA_MODEL = "gpt-4o-mini"  
APE_MODEL = "gpt-4o-mini"
BT_MODEL = "gpt-4o-mini"  

FILTER_OUTPUT_FILENAME = "filtered.json"
GEMBA_OUTPUT_FILENAME = "gemba.json"
APE_OUTPUT_FILENAME = "ape_evidence.json"

TERMBASE_PATH = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\term_base_simple.json"

ENCODE_BATCH_SIZE = 128  
COMET_BATCH_SIZE = 128    
GEMBA_BATCH = 4        
APE_CONCURRENCY = 8 

DEVICE = "cpu"

TERMBASE = {
    "스레드": "thread",
    "이벤트": "event",
    "세션": "session",
    "취약점": "vulnerability",
    "취약성": "vulnerabilities",
    "보안": "security",
    "인증": "authentication",
    "권한": "permission",
    "액세스": "access",
    "로그": "log",
    "모니터링": "monitoring",
    "검증": "validation",
    "암호화": "encryption"
}

ENABLE_CACHING = True
ENABLE_LENGTH_ANALYSIS = True
ENABLE_TERMINOLOGY_CHECK = True
SAVE_INTERMEDIATE_RESULTS = True

def get_output_filename(stage: str, version: str = "v3", model: str = "4o") -> str:
    return f"{stage}_{version}_{model}.json"

VALIDATION_RULES = {
    "min_src_length": 3,
    "min_mt_length": 3,
    "max_length_ratio": 5.0,
    "min_length_ratio": 0.2,  
}
