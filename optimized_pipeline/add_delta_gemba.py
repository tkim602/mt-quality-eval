#!/usr/bin/env python3
"""
기존 APE 결과에 delta_gemba 점수를 추가하는 스크립트

v13의 ape_evidence.json과 gemba.json을 사용해서
APE 개선 후 GEMBA 점수를 계산하고 delta_gemba를 추가합니다.

사용법:
python add_delta_gemba.py --version v13
"""

import asyncio
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from gemba_batch import gemba_batch
import cfg

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

class DeltaGembaCalculator:
    def __init__(self, version: str):
        self.version = version
        self.out_dir = Path("out") / version
        self.ape_evidence_file = self.out_dir / "ape_evidence.json"
        self.gemba_file = self.out_dir / "gemba.json"
        self.output_file = self.out_dir / "ape_evidence_with_delta_gemba.json"
        
    def load_data(self) -> tuple[List[Dict], Dict[str, Any]]:
        """기존 데이터 로드"""
        logger.info(f"Loading data from {self.out_dir}")
        
        # APE evidence 데이터 로드
        if not self.ape_evidence_file.exists():
            raise FileNotFoundError(f"APE evidence file not found: {self.ape_evidence_file}")
            
        with open(self.ape_evidence_file, 'r', encoding='utf-8') as f:
            ape_data = json.load(f)
        logger.info(f"Loaded {len(ape_data)} APE records")
        
        # 원본 GEMBA 데이터 로드 (참조용)
        gemba_data = {}
        if self.gemba_file.exists():
            with open(self.gemba_file, 'r', encoding='utf-8') as f:
                gemba_list = json.load(f)
                gemba_data = {item['key']: item for item in gemba_list}
            logger.info(f"Loaded {len(gemba_data)} original GEMBA scores")
        
        return ape_data, gemba_data
    
    async def calculate_ape_gemba_scores(self, ape_data: List[Dict]) -> List[Dict]:
        """APE 개선된 텍스트에 대해 GEMBA 점수 계산"""
        logger.info("APE 개선된 텍스트에 대해 GEMBA 점수 계산 중...")
        
        # APE가 적용된 레코드만 필터링
        ape_records = []
        src_texts = []
        ape_texts = []
        
        for record in ape_data:
            if record.get('ape') and record.get('ape') != record.get('mt'):
                ape_records.append(record)
                src_texts.append(record['src'])
                ape_texts.append(record['ape'])
        
        if not ape_records:
            logger.warning("APE가 적용된 레코드가 없습니다.")
            return ape_data
        
        logger.info(f"APE가 적용된 {len(ape_records)}개 레코드에 대해 GEMBA 점수 계산")
        
        # 배치별로 나누어 병렬 처리
        from gemba_batch import _score
        import asyncio
        from tqdm.asyncio import tqdm as tqdm_asyncio
        
        # 배치 생성
        batch_size = cfg.GEMBA_BATCH
        batches = []
        
        for i in range(0, len(ape_records), batch_size):
            batch_records = ape_records[i:i + batch_size]
            batch_data = []
            for record in batch_records:
                batch_data.append({
                    "src": record["src"],
                    "mt": record["ape"]  # APE 결과를 평가
                })
            batches.append((batch_records, batch_data))
        
        # 병렬 처리를 위한 세마포어
        semaphore = asyncio.Semaphore(4)  # 4개 동시 처리
        
        async def process_batch(batch_records, batch_data):
            async with semaphore:
                scores = await _score(batch_data)
                for record, (ov, adq, flu, ev) in zip(batch_records, scores):
                    original_gemba = record.get('gemba', 0.0)
                    ape_gemba = float(ov)
                    
                    record['ape_gemba'] = ape_gemba
                    record['delta_gemba'] = float(ape_gemba - original_gemba)
                    
                    # 추가 상세 점수도 저장
                    record['ape_gemba_adequacy'] = float(adq)
                    record['ape_gemba_fluency'] = float(flu)
        
        # 모든 배치를 병렬로 처리
        await tqdm_asyncio.gather(
            *(process_batch(batch_records, batch_data) for batch_records, batch_data in batches),
            desc="GEMBA 배치 처리"
        )
        
        logger.info("GEMBA 점수 계산 완료")
        return ape_data
    
    def calculate_statistics(self, data: List[Dict]) -> Dict[str, Any]:
        """delta_gemba 통계 계산"""
        delta_gemba_values = [
            record.get('delta_gemba', 0) 
            for record in data 
            if 'delta_gemba' in record and record.get('delta_gemba') is not None
        ]
        
        if not delta_gemba_values:
            return {"count": 0, "mean": 0, "positive_count": 0, "negative_count": 0}
        
        positive_count = sum(1 for d in delta_gemba_values if d > 0)
        negative_count = sum(1 for d in delta_gemba_values if d < 0)
        zero_count = len(delta_gemba_values) - positive_count - negative_count
        
        return {
            "count": len(delta_gemba_values),
            "mean": sum(delta_gemba_values) / len(delta_gemba_values),
            "min": min(delta_gemba_values),
            "max": max(delta_gemba_values),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
            "improvement_rate": (positive_count / len(delta_gemba_values)) * 100
        }
    
    def save_results(self, data: List[Dict]):
        """결과 저장"""
        logger.info(f"결과를 {self.output_file}에 저장 중...")
        
        # 메타데이터 추가
        result = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source_version": self.version,
                "source_files": [
                    str(self.ape_evidence_file),
                    str(self.gemba_file) if self.gemba_file.exists() else None
                ],
                "delta_gemba_added": True,
                "script": __file__
            },
            "statistics": self.calculate_statistics(data),
            "records": data
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 결과 저장 완료: {self.output_file}")
        
        # 통계 출력
        stats = result["statistics"]
        logger.info("📊 Delta GEMBA 통계:")
        logger.info(f"  - 총 APE 레코드: {stats['count']}")
        logger.info(f"  - 평균 개선도: {stats['mean']:.3f}")
        logger.info(f"  - 개선된 레코드: {stats['positive_count']} ({stats['improvement_rate']:.1f}%)")
        logger.info(f"  - 악화된 레코드: {stats['negative_count']}")
        logger.info(f"  - 변화없음: {stats['zero_count']}")
        logger.info(f"  - 최대 개선: +{stats['max']:.3f}")
        logger.info(f"  - 최대 악화: {stats['min']:.3f}")

async def main():
    parser = argparse.ArgumentParser(description="기존 APE 결과에 delta_gemba 추가")
    parser.add_argument("--version", default="v13", help="처리할 버전 (기본값: v13)")
    parser.add_argument("--force", action="store_true", help="기존 출력 파일 덮어쓰기")
    
    args = parser.parse_args()
    
    calculator = DeltaGembaCalculator(args.version)
    
    # 출력 파일 존재 확인
    if calculator.output_file.exists() and not args.force:
        logger.error(f"출력 파일이 이미 존재합니다: {calculator.output_file}")
        logger.error("--force 옵션을 사용하여 덮어쓰거나 파일을 삭제해주세요.")
        return
    
    try:
        # 1. 기존 데이터 로드
        ape_data, gemba_data = calculator.load_data()
        
        # 2. APE 텍스트에 대해 GEMBA 점수 계산
        updated_data = await calculator.calculate_ape_gemba_scores(ape_data)
        
        # 3. 결과 저장
        calculator.save_results(updated_data)
        
        logger.info("🎉 Delta GEMBA 계산 완료!")
        
    except Exception as e:
        logger.error(f"❌ 처리 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
