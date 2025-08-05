"""
Q-Score calculation module for MT quality evaluation
Implements Q-score with bucket safety-belt logic
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import cfg

Q_WEIGHTS = {
    'cos': 0.20,
    'comet': 0.30, 
    'gemba': 0.50
}

Q_WEIGHTS_NO_PARSE = {
    'cos': 0.40,
    'comet': 0.60
}

Q_QUANTILES = cfg.Q_SCORE_QUANTILES

FAIL_GATE_CONDITIONS = {
    'gemba_min': 50,
    'comet_min': 0.50,
    'cos_min': 0.50
}

TEMP_GRADES = ['Fail_temp', 'Soft_temp', 'Mid_temp', 'Good_temp', 'Strict_temp']

def compute_global_stats(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    metrics = ['cos', 'comet', 'gemba']
    stats = {}
    
    for metric in metrics:
        values = [record[metric] for record in data if record.get(metric) is not None]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return stats

def z_standardize(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std

def calculate_q_score(z_cos: float, z_comet: float, z_gemba: float, is_no_parse: bool = False) -> float:
    if is_no_parse:
        # No-parse 케이스: COMET 0.6, COS 0.4 가중치 + 5% 페널티
        base_score = (Q_WEIGHTS_NO_PARSE['cos'] * z_cos + 
                      Q_WEIGHTS_NO_PARSE['comet'] * z_comet)
        return 0.95 * base_score  # 5% 페널티 적용
    else:
        # 일반 케이스: 기존 가중치 사용
        return (Q_WEIGHTS['cos'] * z_cos + 
                Q_WEIGHTS['comet'] * z_comet + 
                Q_WEIGHTS['gemba'] * z_gemba)

def check_fail_gate(cos: float, comet: float, gemba: float, is_no_parse: bool = False) -> bool:
    if (comet <= FAIL_GATE_CONDITIONS['comet_min'] or 
        cos <= FAIL_GATE_CONDITIONS['cos_min']):
        return True 

    if not is_no_parse and gemba <= FAIL_GATE_CONDITIONS['gemba_min']:
        return True
        
    return False 

def q_to_temp_grade(q_score: float, passed_fail_gate: bool = False) -> str:
    """Convert Q-score to temporary grade
    
    Args:
        q_score: The calculated Q-score
        passed_fail_gate: True if the record passed fail gate conditions
    
    Returns:
        Temporary grade string
    """
    for i, threshold in enumerate(Q_QUANTILES[1:], 0):
        if q_score < threshold:
            grade = TEMP_GRADES[i]
            # 하한게이트를 통과한 케이스는 최소한 Soft_temp 보장
            if passed_fail_gate and grade == 'Fail_temp':
                return 'Soft_temp'
            return grade
    return TEMP_GRADES[-1] 

def compute_bucket_percentiles(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets = {}
    for record in data:
        bucket = record.get('bucket', 'medium')
        if bucket not in buckets:
            buckets[bucket] = {'cos': [], 'comet': []}
        
        if record.get('cos') is not None:
            buckets[bucket]['cos'].append(record['cos'])
        if record.get('comet') is not None:
            buckets[bucket]['comet'].append(record['comet'])
    
    bucket_q20 = {}
    for bucket, values in buckets.items():
        bucket_q20[bucket] = {
            'cos': np.percentile(values['cos'], 20) if values['cos'] else 0.0,
            'comet': np.percentile(values['comet'], 20) if values['comet'] else 0.0
        }
    
    return bucket_q20

def apply_bucket_safety_belt(temp_grade: str, cos: float, comet: float, gemba: float, 
                           bucket: str, bucket_q20: Dict[str, Dict[str, float]]) -> str:
    if temp_grade not in ['Good_temp', 'Strict_temp']:
        return temp_grade
    q20_cos = bucket_q20.get(bucket, {}).get('cos', 0.0)
    q20_comet = bucket_q20.get(bucket, {}).get('comet', 0.0)
    
    if (cos < q20_cos or 
        comet < q20_comet or 
        gemba < 65):
        return 'Soft_temp'
    
    return temp_grade

def temp_grade_to_final_grade(temp_grade: str) -> str:
    mapping = {
        'Fail_temp': 'fail',
        'Soft_temp': 'soft_pass', 
        'Mid_temp': 'soft_pass', 
        'Good_temp': 'strict_pass',
        'Strict_temp': 'strict_pass'
    }
    return mapping.get(temp_grade, 'fail')

def process_q_score_grading(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def is_no_parse_case(record):
        if record.get('flag', {}).get('gemba_reason') == 'no_parse':
            return True
        if record.get('_ev') == 'no_parse' or record.get('gemba_reason') == 'no_parse':
            return True
        return False
    
    normal_data = [record for record in data if not is_no_parse_case(record)]
    no_parse_data = [record for record in data if is_no_parse_case(record)]
    
    print(f"Normal records: {len(normal_data)}, No-parse records: {len(no_parse_data)}")
    
    for record in no_parse_data:
        print(f"DEBUG no_parse: {record.get('key', 'unknown')} - COMET:{record.get('comet', 0):.3f}, COS:{record.get('cos', 0):.3f}")
    
    if len(normal_data) == 0:
        print("Warning: No normal data found, using all data for global stats")
        global_stats = compute_global_stats(data)
        bucket_q20 = compute_bucket_percentiles(data)
    else:
        global_stats = compute_global_stats(normal_data)
        bucket_q20 = compute_bucket_percentiles(normal_data)
    
    print(f"Global stats: {global_stats}")
    print(f"Bucket 20th percentiles: {bucket_q20}")
    
    processed_data = []
    
    for record in normal_data:
        cos = record.get('cos', 0.0)
        comet = record.get('comet', 0.0)
        gemba = record.get('gemba', 0.0)
        bucket = record.get('bucket', 'medium')
        
        z_cos = z_standardize(cos, global_stats['cos']['mean'], global_stats['cos']['std'])
        z_comet = z_standardize(comet, global_stats['comet']['mean'], global_stats['comet']['std'])
        z_gemba = z_standardize(gemba, global_stats['gemba']['mean'], global_stats['gemba']['std'])
        
        q_score = calculate_q_score(z_cos, z_comet, z_gemba)
        
        if check_fail_gate(cos, comet, gemba, is_no_parse=False):
            temp_grade = 'Fail_temp'
            final_temp_grade = 'Fail_temp'
            final_tag = 'fail'
            print(f"Record {record.get('key', 'unknown')}: Fail gate triggered (G:{gemba:.0f}, C:{comet:.3f}, S:{cos:.3f})")
        else:
            temp_grade = q_to_temp_grade(q_score, passed_fail_gate=True)
            
            final_temp_grade = apply_bucket_safety_belt(temp_grade, cos, comet, gemba, bucket, bucket_q20)
            
            final_tag = temp_grade_to_final_grade(final_temp_grade)
        
        record_copy = record.copy()
        record_copy.update({
            'z_cos': z_cos,
            'z_comet': z_comet,
            'z_gemba': z_gemba,
            'q_score': q_score,
            'temp_grade': temp_grade,
            'final_temp_grade': final_temp_grade,
            'tag': final_tag,
            'q_score_info': {
                'global_stats': global_stats,
                'bucket_q20': bucket_q20.get(bucket, {}),
                'downgraded': temp_grade != final_temp_grade
            }
        })
        
        processed_data.append(record_copy)
    
    for record in no_parse_data:
        cos = record.get('cos', 0.0)
        comet = record.get('comet', 0.0)
        bucket = record.get('bucket', 'medium')
        
        print(f"Processing no_parse record {record.get('key', 'unknown')}: COMET={comet:.3f}, COS={cos:.3f}")
        
        if check_fail_gate(cos, comet, 0.0, is_no_parse=True):
            temp_grade = 'Fail_temp'
            final_temp_grade = 'Fail_temp'
            final_tag = 'fail'
            q_score_no_gemba = None
            z_cos = None
            z_comet = None
            z_gemba = None
            print(f"  → Fail gate triggered for no_parse (C:{comet:.3f}, S:{cos:.3f})")
        else:
            z_cos = z_standardize(cos, global_stats['cos']['mean'], global_stats['cos']['std'])
            z_comet = z_standardize(comet, global_stats['comet']['mean'], global_stats['comet']['std'])
            z_gemba = None  
            q_score_no_gemba = calculate_q_score(z_cos, z_comet, 0.0, is_no_parse=True)
            
            print(f"  → Q-score (no GEMBA): {q_score_no_gemba:.3f}")
            
            temp_grade = q_to_temp_grade(q_score_no_gemba, passed_fail_gate=True)
            
            final_temp_grade = apply_bucket_safety_belt(temp_grade, cos, comet, 0.0, bucket, bucket_q20)
            
            final_tag = temp_grade_to_final_grade(final_temp_grade)
        
        print(f"  → Final tag: {final_tag} (temp: {temp_grade} → {final_temp_grade})")
        print(f"  → Q-score recalculated: {q_score_no_gemba:.3f} (expected ~0.4 for this case)")
        
        record_copy = record.copy()
        record_copy.update({
            'z_cos': z_cos,
            'z_comet': z_comet,
            'z_gemba': z_gemba,
            'q_score': q_score_no_gemba,
            'temp_grade': temp_grade,
            'final_temp_grade': final_temp_grade,
            'tag': final_tag,
            'q_score_info': {
                'global_stats': global_stats,
                'bucket_q20': bucket_q20.get(bucket, {}),
                'downgraded': temp_grade != final_temp_grade,
                'no_parse_handling': True, 
                'weights_used': {'cos': 0.4, 'comet': 0.6, 'gemba': 0.0}
            }
        })
        
        processed_data.append(record_copy)
    
    return processed_data


def save_q_score_stats(data: List[Dict[str, Any]], output_path: Path):
    stats = {
        'total_records': len(data),
        'grade_distribution': {},
        'q_score_distribution': {
            'mean': np.mean([r['q_score'] for r in data]),
            'std': np.std([r['q_score'] for r in data]),
            'min': np.min([r['q_score'] for r in data]),
            'max': np.max([r['q_score'] for r in data])
        },
        'downgrades': sum(1 for r in data if r.get('q_score_info', {}).get('downgraded', False))
    }
    
    for record in data:
        grade = record.get('tag', 'unknown')
        stats['grade_distribution'][grade] = stats['grade_distribution'].get(grade, 0) + 1
    
    stats_file = output_path.parent / f"{output_path.stem}_q_score_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Q-score statistics saved to: {stats_file}")
    return stats
