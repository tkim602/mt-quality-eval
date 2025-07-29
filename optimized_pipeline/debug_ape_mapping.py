#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemba_batch import gemba_batch
import json
import asyncio

async def debug_ape_gemba_mapping():
    """디버그: 파이프라인에서 사용하는 것과 동일한 방식으로 APE GEMBA 계산"""
    
    # v15 데이터 로드
    with open('out/v15/ape_evidence.json', 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    # 파이프라인과 동일한 로직으로 targets_idx 생성
    targets_idx = [i for i, r in enumerate(items) if r.get("tag") in ("soft_pass", "fail")]
    print(f"Total items: {len(items)}")
    print(f"Target indices (soft_pass/fail): {len(targets_idx)}")
    print(f"First few target indices: {targets_idx[:5]}")
    
    # APE가 있는 항목들 찾기
    ape_items = []
    for i in targets_idx:
        if 'ape' in items[i]:
            ape_items.append(i)
    
    print(f"Items with APE: {len(ape_items)}")
    print(f"First few APE item indices: {ape_items[:5]}")
    
    if len(ape_items) >= 2:
        # 첫 번째 APE 항목 확인
        idx1 = ape_items[0]
        src1 = items[idx1]["src"]
        ape1 = items[idx1]["ape"]
        current_score1 = items[idx1].get("ape_gemba", "N/A")
        
        print(f"\n=== Item {idx1} ===")
        print(f"Current ape_gemba: {current_score1}")
        print(f"Source: {src1}")
        print(f"APE: {ape1}")
        
        # 개별적으로 GEMBA 평가
        result = await gemba_batch([src1], [ape1])
        print(f"Individual GEMBA result: {result}")
        
        # 두 번째 APE 항목도 확인
        idx2 = ape_items[1]
        src2 = items[idx2]["src"]
        ape2 = items[idx2]["ape"]
        current_score2 = items[idx2].get("ape_gemba", "N/A")
        
        print(f"\n=== Item {idx2} ===")
        print(f"Current ape_gemba: {current_score2}")
        print(f"Source: {src2}")
        print(f"APE: {ape2}")
        
        # 개별적으로 GEMBA 평가
        result2 = await gemba_batch([src2], [ape2])
        print(f"Individual GEMBA result: {result2}")
        
        # 파이프라인에서 사용하는 방식으로 배치 평가 시뮬레이션
        print(f"\n=== Batch Evaluation Simulation ===")
        src_txt = [items[i]["src"] for i in targets_idx if 'ape' in items[i]]
        ape_txt = [items[i]["ape"] for i in targets_idx if 'ape' in items[i]]
        
        print(f"Source texts count: {len(src_txt)}")
        print(f"APE texts count: {len(ape_txt)}")
        
        if len(src_txt) > 0 and len(ape_txt) > 0:
            # 첫 번째 2개만 테스트
            test_src = src_txt[:2]
            test_ape = ape_txt[:2]
            
            batch_result = await gemba_batch(test_src, test_ape)
            print(f"Batch GEMBA result: {batch_result}")
            
            for i, result in enumerate(batch_result):
                print(f"  Result {i}: {result}")

if __name__ == "__main__":
    asyncio.run(debug_ape_gemba_mapping())
