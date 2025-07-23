#!/usr/bin/env python3

import json
from pathlib import Path
import cfg

def analyze_current_performance():
    """Analyze current threshold performance to suggest conservative adjustments."""
    
    # Load latest pipeline results
    out_dir = Path(cfg.OUT_DIR)
    version_dirs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
    latest_version = max(version_dirs, key=lambda x: int(x.name[1:]))
    file_path = latest_version / cfg.APE_OUTPUT_FILENAME
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Current Threshold Analysis")
    print("=" * 50)
    print(f"Analyzing {len(data)} records from {latest_version}")
    
    # Analyze by bucket
    buckets = {}
    for record in data:
        bucket = record['bucket']
        if bucket not in buckets:
            buckets[bucket] = {'total': 0, 'strict_pass': 0, 'soft_pass': 0, 'fail': 0}
        
        buckets[bucket]['total'] += 1
        buckets[bucket][record['tag']] += 1
    
    print("\nCurrent Distribution:")
    print("Bucket       Total  Strict  Soft   Fail   Pass Rate")
    print("-" * 55)
    
    for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
        if bucket in buckets:
            stats = buckets[bucket]
            pass_rate = (stats['strict_pass'] + stats['soft_pass']) / stats['total'] * 100
            print(f"{bucket:12} {stats['total']:5} {stats['strict_pass']:6} {stats['soft_pass']:5} {stats['fail']:5}   {pass_rate:5.1f}%")
    
    return buckets

def suggest_conservative_thresholds():
    """Suggest more conservative (higher) thresholds."""
    
    current_cos = cfg.COS_THR
    current_comet = cfg.COMET_THR
    
    # Conservative increases: +0.05 to +0.10
    conservative_adjustments = {
        'very_short': {'cos': 0.05, 'comet': 0.05},  # Light adjustment for very short
        'short': {'cos': 0.07, 'comet': 0.07},       # Moderate adjustment
        'medium': {'cos': 0.08, 'comet': 0.05},      # Higher COS adjustment
        'long': {'cos': 0.05, 'comet': 0.08},        # Higher COMET adjustment
        'very_long': {'cos': 0.10, 'comet': 0.10},   # Stronger adjustment for long texts
    }
    
    print("\nConservative Threshold Suggestions:")
    print("=" * 60)
    print("Bucket       Current COS  →  Suggested    Current COMET  →  Suggested")
    print("-" * 70)
    
    suggested_cos = {}
    suggested_comet = {}
    
    for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
        curr_cos = current_cos[bucket]
        curr_comet = current_comet[bucket]
        
        adj = conservative_adjustments[bucket]
        sugg_cos = min(0.90, curr_cos + adj['cos'])  # Cap at 0.90
        sugg_comet = min(0.90, curr_comet + adj['comet'])  # Cap at 0.90
        
        suggested_cos[bucket] = sugg_cos
        suggested_comet[bucket] = sugg_comet
        
        print(f"{bucket:12} {curr_cos:.3f}     →   {sugg_cos:.3f}        {curr_comet:.3f}      →   {sugg_comet:.3f}")
    
    return suggested_cos, suggested_comet

def update_config_with_conservative_thresholds(cos_thr, comet_thr):
    """Update cfg.py with conservative thresholds."""
    cfg_path = Path(__file__).parent / "cfg.py"
    backup_path = cfg_path.with_suffix('.py.conservative_bak')
    
    # Create backup
    if backup_path.exists():
        backup_path.unlink()
    cfg_path.rename(backup_path)
    print(f"\nBacked up original config to {backup_path.name}")
    
    # Read original file
    with open(backup_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update COS_THR
    cos_dict_str = "{\n"
    for bucket, value in cos_thr.items():
        cos_dict_str += f'    "{bucket}": {value:.3f},\n'
    cos_dict_str += "}"
    
    # Update COMET_THR  
    comet_dict_str = "{\n"
    for bucket, value in comet_thr.items():
        comet_dict_str += f'    "{bucket}": {value:.3f},\n'
    comet_dict_str += "}"
    
    # Replace in content
    import re
    content = re.sub(
        r'COS_THR = \{[^}]*\}',
        f'COS_THR = {cos_dict_str}',
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r'COMET_THR = \{[^}]*\}',
        f'COMET_THR = {comet_dict_str}',
        content,
        flags=re.DOTALL
    )
    
    # Write updated file
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Updated cfg.py with conservative thresholds")

def main():
    print("Conservative Threshold Adjustment Tool")
    print("=" * 50)
    print("This tool increases thresholds to reduce false positives.")
    
    # Analyze current performance
    buckets = analyze_current_performance()
    
    # Show suggested conservative thresholds
    suggested_cos, suggested_comet = suggest_conservative_thresholds()
    
    print("\nBenefits of Conservative Thresholds:")
    print("✓ Fewer false positives (questionable translations marked as good)")
    print("✓ Higher precision in quality assessment")
    print("✓ More reliable strict_pass classifications")
    print("\nTrade-offs:")
    print("⚠ Some good translations may be marked as soft_pass or fail")
    print("⚠ Lower recall (may miss some acceptable translations)")
    
    # Ask for confirmation
    update = input("\nDo you want to apply these conservative thresholds? (y/n): ")
    if update.lower() == 'y':
        update_config_with_conservative_thresholds(suggested_cos, suggested_comet)
        print("\nConservative thresholds applied!")
        print("Run 'python run_pipeline.py' to test the new thresholds.")
    else:
        print("No changes made.")

if __name__ == "__main__":
    main()
