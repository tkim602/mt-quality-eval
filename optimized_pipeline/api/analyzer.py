#!/usr/bin/env python3
"""
MT Quality 문제 분석 도구
특정 키나 패턴에 대해 상세 분석
"""

import requests
import json
from typing import Dict, List, Any

class QualityAnalyzer:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
    
    def check_key(self, key: str) -> Dict[str, Any]:
        """특정 키의 상세 정보 확인"""
        try:
            response = requests.get(f"{self.api_url}/records/{key}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Key '{key}' not found"}
        except Exception as e:
            return {"error": str(e)}
    
    def find_similar_failures(self, key: str) -> List[Dict[str, Any]]:
        """비슷한 실패 패턴 찾기"""
        record = self.check_key(key)
        if "error" in record:
            return []
        
        # 같은 bucket, 같은 실패 이유인 것들 찾기
        bucket = record.get("bucket")
        failed_checks = record.get("flag", {}).get("failed", [])
        
        params = {
            "tag": "fail",
            "bucket": bucket,
            "limit": 50
        }
        
        response = requests.get(f"{self.api_url}/records", params=params)
        if response.status_code != 200:
            return []
        
        similar_records = []
        for r in response.json().get("records", []):
            r_failed = r.get("flag", {}).get("failed", [])
            # 실패한 검사 항목이 비슷한 것들
            if set(failed_checks).intersection(set(r_failed)):
                similar_records.append(r)
        
        return similar_records[:10]  # 상위 10개만
    
    def analyze_problem(self, key: str) -> Dict[str, Any]:
        """문제 종합 분석"""
        record = self.check_key(key)
        if "error" in record:
            return record
        
        analysis = {
            "key": key,
            "basic_info": {
                "source": record.get("src", ""),
                "translation": record.get("mt", ""),
                "ape_version": record.get("ape", "N/A"),
                "bucket": record.get("bucket", ""),
                "tag": record.get("tag", "")
            },
            "scores": {
                "gemba": record.get("gemba", 0),
                "gemba_adequacy": record.get("gemba_adequacy", 0),
                "gemba_fluency": record.get("gemba_fluency", 0),
                "comet": record.get("comet", 0),
                "cosine": record.get("cos", 0)
            },
            "issues": {
                "failed_checks": record.get("flag", {}).get("failed", []),
                "gemba_feedback": record.get("flag", {}).get("gemba_reason", ""),
                "validation_issues": self._extract_validation_issues(record)
            },
            "improvement": self._analyze_improvement(record)
        }
        
        # 비슷한 실패 사례들
        similar = self.find_similar_failures(key)
        analysis["similar_failures"] = len(similar)
        analysis["similar_examples"] = [
            {
                "key": r.get("key", ""),
                "src": r.get("src", "")[:50] + "...",
                "gemba": r.get("gemba", 0),
                "issues": r.get("flag", {}).get("failed", [])
            }
            for r in similar[:3]
        ]
        
        return analysis
    
    def _extract_validation_issues(self, record: Dict[str, Any]) -> List[str]:
        """검증 문제점 추출"""
        issues = []
        validation = record.get("validation", {})
        
        # 용어 일관성 문제
        term_consistency = validation.get("term_consistency", {})
        if term_consistency.get("score", 1.0) < 1.0:
            mismatches = term_consistency.get("mismatches", [])
            for mismatch in mismatches:
                issues.append(f"용어 불일치: '{mismatch.get('src_term')}' → 예상 '{mismatch.get('expected_mt')}'")
        
        # 가독성 문제
        readability = validation.get("readability_score", 100)
        if readability < 50:
            issues.append(f"가독성 낮음: {readability:.0f}점")
        
        # 길이 일관성 문제
        length_consistency = validation.get("length_consistency", {})
        if length_consistency.get("issue"):
            issues.append(f"길이 문제: {length_consistency.get('issue')}")
        
        return issues
    
    def _analyze_improvement(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """개선 효과 분석"""
        improvement = {"has_ape": "ape" in record}
        
        if improvement["has_ape"]:
            improvement.update({
                "comet_delta": record.get("delta_comet", 0),
                "cosine_delta": record.get("delta_cos", 0),
                "ape_improved": record.get("delta_comet", 0) > 0 or record.get("delta_cos", 0) > 0
            })
        
        return improvement
    
    def suggest_fixes(self, key: str) -> List[str]:
        """개선 제안"""
        record = self.check_key(key)
        if "error" in record:
            return ["키를 찾을 수 없습니다."]
        
        suggestions = []
        failed_checks = record.get("flag", {}).get("failed", [])
        gemba_reason = record.get("flag", {}).get("gemba_reason", "")
        
        # 실패한 검사별 제안
        if "gemba" in failed_checks:
            if "grammar" in gemba_reason.lower():
                suggestions.append("문법 교정이 필요합니다")
            if "terminology" in gemba_reason.lower() or "term" in gemba_reason.lower():
                suggestions.append("용어 일관성 확인이 필요합니다")
            if "fluency" in gemba_reason.lower():
                suggestions.append("자연스러운 표현으로 수정이 필요합니다")
        
        if "comet" in failed_checks:
            suggestions.append("번역 품질 전반적 검토가 필요합니다")
        
        if "cosine" in failed_checks:
            suggestions.append("의미 유사성 개선이 필요합니다")
        
        # APE가 있으면 APE 버전 확인 제안
        if "ape" in record:
            suggestions.append(f"APE 개선 버전 확인: '{record['ape']}'")
        
        return suggestions if suggestions else ["구체적인 개선 제안을 위해서는 더 많은 분석이 필요합니다."]


def main():
    """대화형 분석 도구"""
    analyzer = QualityAnalyzer()
    
    print("🔍 MT Quality 문제 분석 도구")
    print("사용법: 키 입력하면 상세 분석 결과를 보여줍니다")
    print("종료: 'quit' 입력")
    print()
    
    while True:
        key = input("🔑 분석할 키를 입력하세요: ").strip()
        
        if key.lower() in ['quit', 'exit', 'q']:
            print("👋 분석 도구를 종료합니다.")
            break
        
        if not key:
            continue
        
        print(f"\n📊 '{key}' 분석 중...")
        analysis = analyzer.analyze_problem(key)
        
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            continue
        
        # 기본 정보
        print(f"\n📝 기본 정보:")
        print(f"  원문: {analysis['basic_info']['source']}")
        print(f"  번역: {analysis['basic_info']['translation']}")
        if analysis['basic_info']['ape_version'] != "N/A":
            print(f"  APE: {analysis['basic_info']['ape_version']}")
        print(f"  길이: {analysis['basic_info']['bucket']}")
        print(f"  결과: {analysis['basic_info']['tag']}")
        
        # 점수
        scores = analysis['scores']
        print(f"\n📈 점수:")
        print(f"  GEMBA: {scores['gemba']:.0f} (적절성: {scores['gemba_adequacy']:.0f}, 유창성: {scores['gemba_fluency']:.0f})")
        print(f"  COMET: {scores['comet']:.3f}")
        print(f"  Cosine: {scores['cosine']:.3f}")
        
        # 문제점
        issues = analysis['issues']
        print(f"\n⚠️  문제점:")
        print(f"  실패한 검사: {', '.join(issues['failed_checks'])}")
        if issues['gemba_feedback']:
            print(f"  GEMBA 피드백: {issues['gemba_feedback']}")
        if issues['validation_issues']:
            for issue in issues['validation_issues']:
                print(f"  검증 문제: {issue}")
        
        # 개선 효과
        improvement = analysis['improvement']
        if improvement['has_ape']:
            print(f"\n🚀 APE 개선 효과:")
            print(f"  COMET 개선: {improvement['comet_delta']:+.3f}")
            print(f"  Cosine 개선: {improvement['cosine_delta']:+.3f}")
            print(f"  개선됨: {'✅' if improvement['ape_improved'] else '❌'}")
        
        # 개선 제안
        suggestions = analyzer.suggest_fixes(key)
        print(f"\n💡 개선 제안:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        
        # 비슷한 실패 사례
        if analysis['similar_failures'] > 0:
            print(f"\n🔗 비슷한 실패 사례 {analysis['similar_failures']}개 발견:")
            for example in analysis['similar_examples']:
                print(f"  - {example['key']}: {example['src']} (GEMBA: {example['gemba']:.0f})")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
