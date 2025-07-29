#!/usr/bin/env python3
"""
Delta GEMBA 계산 실행 스크립트

이 스크립트는 기존 v13 APE 결과에 delta_gemba를 추가합니다.
COS, COMET 점수는 다시 계산하지 않고 GEMBA 점수만 추가로 계산합니다.
"""

import asyncio
import sys
import os
from pathlib import Path

def main():
    print("🚀 Delta GEMBA 계산을 시작합니다...")
    print()
    print("📋 작업 내용:")
    print("  1. v13/ape_evidence.json에서 APE 개선된 텍스트 확인")
    print("  2. APE 텍스트에 대해서만 GEMBA 점수 재계산")
    print("  3. delta_gemba = ape_gemba - original_gemba 계산")
    print("  4. 결과를 v13/ape_evidence_with_delta_gemba.json에 저장")
    print()
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    expected_files = [
        "add_delta_gemba.py",
        "gemba_batch.py", 
        "cfg.py",
        "out/v13/ape_evidence.json"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not (current_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 다음 파일들이 없습니다:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print()
        print("optimized_pipeline 디렉토리에서 실행해주세요.")
        return 1
    
    # 출력 파일 확인
    output_file = current_dir / "out/v13/ape_evidence_with_delta_gemba.json"
    if output_file.exists():
        print("⚠️  출력 파일이 이미 존재합니다:")
        print(f"   {output_file}")
        response = input("\n덮어쓸까요? (y/N): ").strip().lower()
        if response != 'y':
            print("작업을 취소했습니다.")
            return 0
        force_flag = "--force"
    else:
        force_flag = ""
    
    print("\n🔄 계산을 시작합니다... (몇 분 소요될 수 있습니다)")
    
    # add_delta_gemba.py 실행
    cmd = f"python add_delta_gemba.py --version v13 {force_flag}"
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("\n🎉 Delta GEMBA 계산이 완료되었습니다!")
        print()
        print("📁 결과 파일:")
        print(f"   {output_file}")
        print()
        print("🌐 API 서버를 다시 시작하면 새로운 데이터가 적용됩니다:")
        print("   cd api")
        print("   python main.py")
        print()
        print("💡 이제 프론트엔드에서 정확한 품질 등급 변화를 확인할 수 있습니다!")
        return 0
    else:
        print("\n❌ 계산 중 오류가 발생했습니다.")
        print("로그를 확인해주세요.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
