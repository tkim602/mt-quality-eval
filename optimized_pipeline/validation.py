#!/usr/bin/env python3
"""
Validation functions for the MT pipeline
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import textstat

def check_term_consistency(src: str, mt: str, termbase: List[Dict[str, str]]) -> Dict[str, Any]:
    
    if not termbase:
        return {"score": 1.0, "mismatches": []}
    
    mismatches = []
    
    try:
        for term_pair in termbase:
            if not isinstance(term_pair, dict):
                continue
                
            ko_term = term_pair.get("ko", "").strip()
            en_term = term_pair.get("en-US", "").strip()
            
            if not ko_term or not en_term:
                continue
            
            # Check if Korean term exists in source
            if ko_term.lower() in src.lower():
                # Check if corresponding English term exists in MT
                if en_term.lower() not in mt.lower():
                    mismatches.append({
                        "src_term": ko_term,
                        "expected_mt": en_term,
                        "found_in_src": True,
                        "found_in_mt": False
                    })
    
    except Exception as e:
        # Handle any errors gracefully
        pass
    
    score = 1.0 if len(mismatches) == 0 else max(0.0, 1.0 - len(mismatches) * 0.2)
    
    return {
        "score": score,
        "mismatches": mismatches
    }

def check_technical_formats(src: str, mt: str) -> Dict[str, Any]:
    """Check if technical formats are preserved correctly"""
    issues = []
    
    # Version numbers (e.g., "v1.2.3", "TLS 1.2")
    version_pattern = r'\b[vV]?\d+\.\d+(?:\.\d+)?\b'
    src_versions = set(re.findall(version_pattern, src))
    mt_versions = set(re.findall(version_pattern, mt))
    if src_versions != mt_versions:
        issues.append(f"Version mismatch: {list(src_versions)} vs {list(mt_versions)}")
    
    # URLs and file paths
    url_pattern = r'https?://[^\s]+|/[^\s]*|\\[^\s]*'
    src_urls = set(re.findall(url_pattern, src))
    mt_urls = set(re.findall(url_pattern, mt))
    if src_urls != mt_urls:
        issues.append(f"URL/path mismatch: {list(src_urls)} vs {list(mt_urls)}")
    
    # Error codes and technical identifiers
    error_pattern = r'\b(?:HTTP|CVE|SSL|TLS|ID:|SPDX)\s*[-:]?\s*[\w\d.-]+\b'
    src_codes = set(re.findall(error_pattern, src, re.IGNORECASE))
    mt_codes = set(re.findall(error_pattern, mt, re.IGNORECASE))
    if src_codes != mt_codes:
        issues.append(f"Technical identifier mismatch: {list(src_codes)} vs {list(mt_codes)}")
    
    # Special characters and symbols that should be preserved
    special_pattern = r'[_\-(){}[\]<>@#$%^&*+=|\\/:;\"\'`~]'
    src_special = set(re.findall(special_pattern, src))
    mt_special = set(re.findall(special_pattern, mt))
    
    # Only flag if important technical symbols are missing
    important_missing = src_special - mt_special
    critical_chars = {'_', '/', '\\', ':', '.', '-'}
    if important_missing.intersection(critical_chars):
        issues.append(f"Critical technical symbols missing: {list(important_missing.intersection(critical_chars))}")
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "score": 1.0 if len(issues) == 0 else max(0.0, 1.0 - len(issues) * 0.25)
    }

def check_length_consistency(src: str, mt: str) -> Dict[str, Any]:
    """Check if translation length is reasonable"""
    src_len = len(src.strip())
    mt_len = len(mt.strip())
    
    if src_len == 0:
        return {"score": 0.0, "ratio": 0, "issue": "Empty source"}
    
    ratio = mt_len / src_len
    
    # Korean to English typically expands by 1.2-2.5x
    if ratio < 0.5:
        issue = "Translation too short"
        score = 0.3
    elif ratio > 4.0:
        issue = "Translation too long" 
        score = 0.3
    elif ratio < 0.8 or ratio > 3.0:
        issue = "Length ratio suspicious"
        score = 0.7
    else:
        issue = None
        score = 1.0
    
    return {
        "score": score,
        "ratio": ratio,
        "issue": issue
    }

def get_readability_score(text: str) -> float:
    """Calculate readability score for English text"""
    try:
        # Use Flesch Reading Ease for English
        score = textstat.flesch_reading_ease(text)
        return max(0.0, score)  # Ensure non-negative
    except:
        return 50.0  # Default neutral score

def validate_translation(src: str, mt: str, termbase: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Validation function for Korean→English translation
    """
    
    validation_results = {
        "term_consistency": check_term_consistency(src, mt, termbase or []),
        "technical_formats": check_technical_formats(src, mt),
        "length_consistency": check_length_consistency(src, mt),
        "readability_score": get_readability_score(mt)
    }
    
    # Calculate overall validation score
    scores = []
    for check_name, result in validation_results.items():
        if isinstance(result, dict) and "score" in result:
            scores.append(result["score"])
        elif check_name == "readability_score":
            # Normalize readability score to 0-1 range
            normalized = min(1.0, max(0.0, result / 100.0))
            scores.append(normalized)
    
    overall_score = sum(scores) / len(scores) if scores else 0.0
    validation_results["overall_score"] = overall_score
    
    return validation_results

def run_all_validations(src: str, mt: str, termbase: dict = None) -> Dict[str, Any]:
    """
    Wrapper function for filter.py compatibility
    Converts termbase dict to list format if needed
    """
    if termbase and isinstance(termbase, dict):
        # Convert dict format to list format
        termbase_list = [{"ko": k, "en-US": v} for k, v in termbase.items()]
    elif termbase and isinstance(termbase, list):
        termbase_list = termbase
    else:
        termbase_list = []
    
    return validate_translation(src, mt, termbase_list)
