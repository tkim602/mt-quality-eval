#!/usr/bin/env python3
"""
Seed 재현성 테스트
"""

import json
import random
import numpy as np
import torch
import cfg

def test_seed_reproducibility():
    print("🧪 Seed 재현성 테스트 시작")
    print(f"🎲 SEED = {cfg.SEED}")
    
    # 한국어 데이터 로드
    with open(cfg.KO_JSON, 'r', encoding='utf-8') as f:
        ko = json.load(f)
    
    all_keys = list(ko.keys())
    print(f"📊 총 키 개수: {len(all_keys)}")
    
    # 첫 번째 샘플링
    print("\n🔄 첫 번째 샘플링:")
    if cfg.SEED is not None:
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
    
    keys1 = random.sample(all_keys, min(cfg.LIMIT, len(all_keys)))
    print(f"선택된 키 개수: {len(keys1)}")
    print(f"첫 10개 키: {keys1[:10]}")
    
    # 두 번째 샘플링 (같은 seed)
    print("\n🔄 두 번째 샘플링 (같은 seed):")
    if cfg.SEED is not None:
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        torch.manual_seed(cfg.SEED)
    
    keys2 = random.sample(all_keys, min(cfg.LIMIT, len(all_keys)))
    print(f"선택된 키 개수: {len(keys2)}")
    print(f"첫 10개 키: {keys2[:10]}")
    
    # 세 번째 샘플링 (다른 seed)
    print("\n🔄 세 번째 샘플링 (다른 seed=123):")
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    
    keys3 = random.sample(all_keys, min(cfg.LIMIT, len(all_keys)))
    print(f"선택된 키 개수: {len(keys3)}")
    print(f"첫 10개 키: {keys3[:10]}")
    
    # 결과 비교
    print("\n📋 결과 분석:")
    if keys1 == keys2:
        print("✅ 같은 seed로 완전히 동일한 샘플 선택됨")
    else:
        print("❌ 같은 seed인데 다른 샘플 선택됨")
        
    if keys1 == keys3:
        print("❌ 다른 seed인데 같은 샘플 선택됨 (문제)")
    else:
        print("✅ 다른 seed로 다른 샘플 선택됨")
    
    # 중복 확인
    common_12 = len(set(keys1) & set(keys2))
    common_13 = len(set(keys1) & set(keys3))
    
    print(f"📊 1번과 2번 샘플 공통 개수: {common_12}/{len(keys1)}")
    print(f"📊 1번과 3번 샘플 공통 개수: {common_13}/{len(keys1)}")

if __name__ == "__main__":
    test_seed_reproducibility()
