from pathlib import Path
import os
import numpy as np
from typing import Dict, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

KO_JSON = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\ko_checker_dedup.json"
EN_JSON = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\en-US_checker.json"

LIMIT = 100  
SEED = 42

LABSE_MODEL = "sentence-transformers/LaBSE"
COMET_CKPT = "Unbabel/wmt22-cometkiwi-da"
GEMBA_MODEL = "gpt-3.5-turbo-16k"
APE_MODEL = "gpt-4o-mini"
BT_MODEL = "gpt-4o-mini"  

FILTER_OUTPUT_FILENAME = "filtered.json"
GEMBA_OUTPUT_FILENAME = "gemba.json"
APE_OUTPUT_FILENAME = "ape_evidence.json"

TERMBASE_PATH = r"c:\Users\tkim602_global\Desktop\mt_eval\data\samples\term_base_simple.json"

ENCODE_BATCH_SIZE = 128  
COMET_BATCH_SIZE = 64    
GEMBA_BATCH = 4        
APE_CONCURRENCY = 8     

DEVICE = "cuda" if os.getenv("USE_GPU", "true").lower() == "true" else "cpu"

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

# Optimized thresholds based on analysis
COS_THR = {
    "very_short": 0.820,  # Reduced from 0.840
    "short": 0.800,       # Reduced from 0.850 (71% failure rate was too high)
    "medium": 0.820,
    "long": 0.820,
    "very_long": 0.830
}

COMET_THR = {
    "very_short": 0.830,  # Reduced from 0.850
    "short": 0.780,       # Reduced from 0.830 (71% failure rate was too high)
    "medium": 0.840, 
    "long": 0.830,
    "very_long": 0.830
}

GEMBA_PASS = 65  # Lowered from 70 for better balance

# Business-specific thresholds for different string types
BUSINESS_RULES = {
    "critical_strings": {  # UI labels, error messages
        "cos_boost": 0.05,
        "comet_boost": 0.05,
        "required_confidence": 0.90
    },
    "help_text": {  # Help text, descriptions
        "cos_penalty": -0.03,
        "comet_penalty": -0.03,
        "required_confidence": 0.70
    },
    "default": {
        "required_confidence": 0.75
    }
} 

STRICT_COS_THR = {
    "very_short": 0.900,
    "short": 0.885,
    "medium": 0.910,
    "long": 0.915,
    "very_long": 0.920
}

STRICT_COMET_THR = {
    "very_short": 0.880,
    "short": 0.875,
    "medium": 0.885,
    "long": 0.870,
    "very_long": 0.850
}

ENABLE_CACHING = True
ENABLE_STRICT_MODE = False 
ENABLE_LENGTH_ANALYSIS = True
ENABLE_TERMINOLOGY_CHECK = True
SAVE_INTERMEDIATE_RESULTS = True

def get_output_filename(stage: str, version: str = "v3", model: str = "4o") -> str:
    return f"{stage}_{version}_{model}.json"

def calculate_confidence(cos: float, comet: float, gemba: float, bucket: str) -> float:
    """Calculate confidence in quality decision based on metric agreement and values"""
    # Normalize GEMBA to 0-1 scale
    gemba_norm = gemba / 100.0
    
    # Calculate agreement between metrics (lower std = higher agreement)
    metrics = [cos, comet, gemba_norm]
    agreement = 1.0 - np.std(metrics)  # Higher agreement = higher confidence
    
    # Base confidence from minimum metric value
    base_confidence = min(cos, comet, gemba_norm)
    
    # Bucket-specific confidence adjustments
    bucket_weights = {
        "very_short": 0.95,  # High confidence for very short texts
        "short": 0.90,
        "medium": 1.0,
        "long": 1.0,
        "very_long": 0.85   # Slightly lower confidence for very long texts
    }
    
    bucket_weight = bucket_weights.get(bucket, 1.0)
    
    # Combine factors
    confidence = (base_confidence * 0.6 + agreement * 0.4) * bucket_weight
    return min(1.0, max(0.0, confidence))

def get_string_type(key: str) -> str:
    """Determine string type based on key patterns"""
    key_lower = key.lower()
    
    # Critical strings (UI labels, buttons, errors)
    critical_patterns = [
        'button', 'label', 'error', 'warning', 'title', 'menu',
        'save', 'cancel', 'ok', 'apply', 'submit', 'delete', 'remove',
        'add', 'create', 'new', 'edit', 'update', 'confirm', 'yes', 'no'
    ]
    if any(pattern in key_lower for pattern in critical_patterns):
        return "critical_strings"
    
    # Help text (descriptions, help, tooltips)
    help_patterns = [
        'help', 'description', 'tooltip', 'hint', 'info', 'instruction',
        'click here', 'more information', 'learn more', 'see details'
    ]
    if any(pattern in key_lower for pattern in help_patterns):
        return "help_text"
    
    return "default"

def make_quality_decision_enhanced(cos: float, comet: float, gemba: float, bucket: str, key: str = "", strict_mode: bool = False) -> Tuple[str, list, list, float]:
    """Enhanced quality decision with confidence scoring and business rules"""
    
    # Get business rules for this string type
    string_type = get_string_type(key)
    business_rule = BUSINESS_RULES.get(string_type, BUSINESS_RULES["default"])
    
    # Apply business rule adjustments
    adjusted_cos = cos + business_rule.get("cos_boost", 0) + business_rule.get("cos_penalty", 0)
    adjusted_comet = comet + business_rule.get("comet_boost", 0) + business_rule.get("comet_penalty", 0)
    
    # Clamp values to valid range
    adjusted_cos = min(1.0, max(0.0, adjusted_cos))
    adjusted_comet = min(1.0, max(0.0, adjusted_comet))
    
    # Get thresholds
    cos_thr = STRICT_COS_THR if strict_mode else COS_THR
    comet_thr = STRICT_COMET_THR if strict_mode else COMET_THR
    
    # Calculate confidence
    confidence = calculate_confidence(adjusted_cos, adjusted_comet, gemba, bucket)
    
    # Enhanced decision logic with confidence consideration
    checks = [
        adjusted_cos >= cos_thr[bucket],
        adjusted_comet >= comet_thr[bucket],
        gemba >= GEMBA_PASS,
    ]
    
    keys = ("cosine", "comet", "gemba")
    passed = [k for k, ok in zip(keys, checks) if ok]
    failed = [k for k in keys if k not in passed]
    
    # Decision logic with confidence weighting
    required_confidence = business_rule["required_confidence"]
    
    if adjusted_cos == 1.0:  # Perfect cosine similarity
        tag = "strict_pass"
    elif len(passed) == 3 and confidence >= required_confidence:
        tag = "strict_pass"
    elif len(passed) >= 2 and confidence >= (required_confidence - 0.1):
        tag = "soft_pass"
    elif len(passed) >= 1 and confidence >= (required_confidence - 0.2) and bucket in ["very_short", "short"]:
        # More lenient for short texts with reasonable confidence
        tag = "soft_pass"
    else:
        tag = "fail"
    
    return tag, passed, failed, confidence

def make_quality_decision(cos: float, comet: float, gemba: float, bucket: str, strict_mode: bool = False) -> tuple:
    """Legacy function - maintained for backward compatibility"""
    tag, passed, failed, _ = make_quality_decision_enhanced(cos, comet, gemba, bucket, "", strict_mode)
    return tag, passed, failed

VALIDATION_RULES = {
    "min_src_length": 3,
    "min_mt_length": 3,
    "max_length_ratio": 5.0,
    "min_length_ratio": 0.2,  
}
