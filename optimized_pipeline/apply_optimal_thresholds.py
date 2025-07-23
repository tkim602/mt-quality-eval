#!/usr/bin/env python3

import cfg
from pathlib import Path
import shutil

def apply_reference_optimal_thresholds():
    """Apply the reference optimal thresholds from the 100 dataset analysis"""
    
    # Reference optimal thresholds based on your data
    optimal_cos = {
        "very_short": 0.840,
        "short": 0.850,
        "medium": 0.820,
        "long": 0.820,
        "very_long": 0.830
    }
    
    optimal_comet = {
        "very_short": 0.850,
        "short": 0.830,
        "medium": 0.840,
        "long": 0.830,
        "very_long": 0.830
    }
    
    print("Applying Reference Optimal Thresholds")
    print("="*40)
    print("Length Bucket    Optimal COS    Optimal COMET    F1 Score")
    print("-"*55)
    print("very_short       0.840          0.850            98.39%")
    print("short            0.850          0.830            95.65%")
    print("medium           0.820          0.840            87.39%")
    print("long             0.820          0.830            87.80%")
    print("very_long        0.830          0.830            93.33%")
    print("="*40)
    
    # Backup current config
    cfg_path = Path('cfg.py')
    backup_path = cfg_path.with_suffix('.py.backup')
    shutil.copy2(cfg_path, backup_path)
    print(f"✓ Backed up current config to {backup_path}")
    
    # Read current config
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create new threshold strings
    cos_thr_str = "COS_THR = {\n"
    for bucket, threshold in optimal_cos.items():
        cos_thr_str += f'    "{bucket}": {threshold},\n'
    cos_thr_str += "}"
    
    comet_thr_str = "COMET_THR = {\n"
    for bucket, threshold in optimal_comet.items():
        comet_thr_str += f'    "{bucket}": {threshold},\n'
    comet_thr_str += "}"
    
    # Replace thresholds using regex
    import re
    content = re.sub(r'COS_THR = \{[^}]+\}', cos_thr_str, content, flags=re.DOTALL)
    content = re.sub(r'COMET_THR = \{[^}]+\}', comet_thr_str, content, flags=re.DOTALL)
    
    # Write updated config
    with open(cfg_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Applied reference optimal thresholds to cfg.py")
    
    # Show comparison
    print("\nThreshold Changes:")
    print("-"*50)
    print(f"{'Bucket':<12} {'COS Change':<15} {'COMET Change':<15}")
    print("-"*50)
    
    for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
        current_cos = getattr(cfg.COS_THR, bucket, 0)
        current_comet = getattr(cfg.COMET_THR, bucket, 0)
        new_cos = optimal_cos[bucket]
        new_comet = optimal_comet[bucket]
        
        cos_change = f"{current_cos:.3f}→{new_cos:.3f}"
        comet_change = f"{current_comet:.3f}→{new_comet:.3f}"
        
        print(f"{bucket:<12} {cos_change:<15} {comet_change:<15}")
    
    print("\n✓ Reference optimal thresholds applied successfully!")
    print("Run 'python run_pipeline.py' to test with these optimized thresholds.")

def main():
    print("Reference Optimal Threshold Applicator")
    print("This will apply the optimal thresholds from your 100-dataset analysis.")
    print("These thresholds achieved F1 scores ranging from 87.39% to 98.39%.")
    
    response = input("\nApply reference optimal thresholds? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        apply_reference_optimal_thresholds()
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
