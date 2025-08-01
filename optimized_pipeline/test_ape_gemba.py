#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemba_batch import gemba_batch
import json
import asyncio

async def test_ape_gemba():
    # v15 데이터에서 APE 문장들을 가져와서 직접 GEMBA 평가
    with open('out/v15/ape_evidence.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # APE가 있는 케이스들 찾기
    ape_cases = [item for item in data if 'ape' in item]
    
    print(f"Found {len(ape_cases)} APE cases")
    
    # 첫 번째 케이스 테스트
    if ape_cases:
        case = ape_cases[0]
        src_txt = case['src']
        ape_txt = case['ape']
        
        print(f"\nTesting first case:")
        print(f"Source: {src_txt}")
        print(f"APE: {ape_txt}")
        print(f"Current ape_gemba in data: {case.get('ape_gemba', 'N/A')}")
        
        # GEMBA로 직접 평가
        print(f"\nEvaluating APE text with GEMBA...")
        result = await gemba_batch([src_txt], [ape_txt])
        print(f"GEMBA result: {result}")
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], tuple):
                overall, adequacy, fluency, evidence = result[0]
                print(f"Overall: {overall}, Adequacy: {adequacy}, Fluency: {fluency}")
            else:
                print(f"Overall score: {result[0]}")

    # 두 번째 케이스도 테스트
    if len(ape_cases) > 1:
        case = ape_cases[1]
        src_txt = case['src']
        ape_txt = case['ape']
        
        print(f"\n\nTesting second case:")
        print(f"Source: {src_txt}")
        print(f"APE: {ape_txt}")
        print(f"Current ape_gemba in data: {case.get('ape_gemba', 'N/A')}")
        
        # GEMBA로 직접 평가
        print(f"\nEvaluating APE text with GEMBA...")
        result = await gemba_batch([src_txt], [ape_txt])
        print(f"GEMBA result: {result}")
        
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], tuple):
                overall, adequacy, fluency, evidence = result[0]
                print(f"Overall: {overall}, Adequacy: {adequacy}, Fluency: {fluency}")
            else:
                print(f"Overall score: {result[0]}")

if __name__ == "__main__":
    asyncio.run(test_ape_gemba())
