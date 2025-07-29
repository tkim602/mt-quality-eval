#!/usr/bin/env python3
"""
GEMBA 개선 효과 분석 스크립트
프롬프트와 모델 변경 후 결과 분석
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class GembaAnalyzer:
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.data = None
        
    def load_results(self) -> bool:
        """결과 파일들 로드"""
        try:
            # GEMBA 결과 파일 로드
            gemba_file = self.result_dir / "gemba.json"
            if not gemba_file.exists():
                print(f"❌ GEMBA 결과 파일이 없습니다: {gemba_file}")
                return False
            
            with open(gemba_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            print(f"✅ {len(self.data)}개 레코드 로드됨")
            return True
            
        except Exception as e:
            print(f"❌ 파일 로드 오류: {e}")
            return False
    
    def analyze_gemba_scores(self) -> Dict[str, Any]:
        """GEMBA 점수 분석"""
        if not self.data:
            return {}
        
        # 기본 통계
        overall_scores = [item['gemba'] for item in self.data]
        adequacy_scores = [item['gemba_adequacy'] for item in self.data]
        fluency_scores = [item['gemba_fluency'] for item in self.data]
        
        # 길이별 분석
        bucket_analysis = {}
        for bucket in ['very_short', 'short', 'medium', 'long', 'very_long']:
            bucket_items = [item for item in self.data if item['bucket'] == bucket]
            if bucket_items:
                bucket_scores = [item['gemba'] for item in bucket_items]
                bucket_analysis[bucket] = {
                    'count': len(bucket_items),
                    'avg_score': statistics.mean(bucket_scores),
                    'pass_rate': len([s for s in bucket_scores if s >= 75]) / len(bucket_scores) * 100
                }
        
        return {
            'total_samples': len(self.data),
            'overall_stats': {
                'avg_overall': statistics.mean(overall_scores),
                'avg_adequacy': statistics.mean(adequacy_scores),
                'avg_fluency': statistics.mean(fluency_scores),
                'median_overall': statistics.median(overall_scores),
                'std_overall': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0
            },
            'score_distribution': {
                'excellent_90+': len([s for s in overall_scores if s >= 90]),
                'good_80_89': len([s for s in overall_scores if 80 <= s < 90]),
                'acceptable_70_79': len([s for s in overall_scores if 70 <= s < 80]),
                'poor_60_69': len([s for s in overall_scores if 60 <= s < 70]),
                'failing_below_60': len([s for s in overall_scores if s < 60])
            },
            'bucket_analysis': bucket_analysis
        }
    
    def analyze_quality_decisions(self) -> Dict[str, Any]:
        """품질 결정 분석"""
        if not self.data:
            return {}
        
        # 태그 분포
        tag_counts = {}
        for item in self.data:
            tag = item.get('tag', 'unknown')
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 실패한 케이스 분석
        failed_items = [item for item in self.data if item.get('tag') == 'fail']
        fail_reasons = {}
        
        for item in failed_items:
            failed_checks = item.get('flag', {}).get('failed', [])
            for reason in failed_checks:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        
        return {
            'tag_distribution': tag_counts,
            'pass_rate': (tag_counts.get('strict_pass', 0) + tag_counts.get('soft_pass', 0)) / len(self.data) * 100,
            'strict_pass_rate': tag_counts.get('strict_pass', 0) / len(self.data) * 100,
            'soft_pass_rate': tag_counts.get('soft_pass', 0) / len(self.data) * 100,
            'fail_rate': tag_counts.get('fail', 0) / len(self.data) * 100,
            'failure_reasons': fail_reasons
        }
    
    def analyze_evidence_quality(self) -> Dict[str, Any]:
        """Evidence 품질 분석"""
        if not self.data:
            return {}
        
        evidence_lengths = []
        evidence_types = {
            'perfect_excellent': 0,
            'specific_issues': 0,
            'generic_feedback': 0,
            'empty_or_minimal': 0
        }
        
        for item in self.data:
            evidence = item.get('flag', {}).get('gemba_reason', '')
            evidence_lengths.append(len(evidence))
            
            # Evidence 유형 분류
            evidence_lower = evidence.lower()
            if any(word in evidence_lower for word in ['perfect', 'excellent', 'flawless']):
                evidence_types['perfect_excellent'] += 1
            elif len(evidence) > 50 and any(word in evidence_lower for word in 
                                          ['grammar', 'terminology', 'accuracy', 'fluency']):
                evidence_types['specific_issues'] += 1
            elif len(evidence) > 20:
                evidence_types['generic_feedback'] += 1
            else:
                evidence_types['empty_or_minimal'] += 1
        
        return {
            'avg_evidence_length': statistics.mean(evidence_lengths) if evidence_lengths else 0,
            'evidence_types': evidence_types,
            'detailed_feedback_rate': (evidence_types['specific_issues'] / len(self.data) * 100) if self.data else 0
        }
    
    def compare_with_baseline(self, baseline_stats: Dict = None) -> Dict[str, Any]:
        """기준선과 비교 (수동으로 입력하거나 이전 결과와 비교)"""
        
        # 일반적인 GPT-3.5-turbo 기준선 (추정값)
        if baseline_stats is None:
            baseline_stats = {
                'avg_overall': 72.5,
                'pass_rate': 65.0,
                'detailed_feedback_rate': 40.0,
                'avg_evidence_length': 35
            }
        
        current_stats = self.analyze_gemba_scores()
        quality_stats = self.analyze_quality_decisions()
        evidence_stats = self.analyze_evidence_quality()
        
        improvements = {
            'score_improvement': current_stats['overall_stats']['avg_overall'] - baseline_stats['avg_overall'],
            'pass_rate_improvement': quality_stats['pass_rate'] - baseline_stats['pass_rate'],
            'evidence_quality_improvement': evidence_stats['detailed_feedback_rate'] - baseline_stats['detailed_feedback_rate'],
            'evidence_length_improvement': evidence_stats['avg_evidence_length'] - baseline_stats['avg_evidence_length']
        }
        
        return {
            'baseline': baseline_stats,
            'current': {
                'avg_overall': current_stats['overall_stats']['avg_overall'],
                'pass_rate': quality_stats['pass_rate'],
                'detailed_feedback_rate': evidence_stats['detailed_feedback_rate'],
                'avg_evidence_length': evidence_stats['avg_evidence_length']
            },
            'improvements': improvements
        }
    
    def print_analysis_report(self):
        """분석 결과 출력"""
        print("🎯 GEMBA 개선 효과 분석 리포트")
        print("=" * 60)
        
        if not self.load_results():
            return
        
        # 1. GEMBA 점수 분석
        gemba_stats = self.analyze_gemba_scores()
        print(f"\n📊 GEMBA 점수 분석 (총 {gemba_stats['total_samples']}개 샘플)")
        print("-" * 40)
        overall = gemba_stats['overall_stats']
        print(f"평균 Overall 점수: {overall['avg_overall']:.1f}")
        print(f"평균 Adequacy: {overall['avg_adequacy']:.1f}")
        print(f"평균 Fluency: {overall['avg_fluency']:.1f}")
        print(f"중앙값: {overall['median_overall']:.1f}")
        print(f"표준편차: {overall['std_overall']:.1f}")
        
        # 점수 분포
        print(f"\n점수 분포:")
        dist = gemba_stats['score_distribution']
        print(f"  🌟 우수 (90+): {dist['excellent_90+']}개 ({dist['excellent_90+']/gemba_stats['total_samples']*100:.1f}%)")
        print(f"  ✅ 양호 (80-89): {dist['good_80_89']}개 ({dist['good_80_89']/gemba_stats['total_samples']*100:.1f}%)")
        print(f"  ⚠️  보통 (70-79): {dist['acceptable_70_79']}개 ({dist['acceptable_70_79']/gemba_stats['total_samples']*100:.1f}%)")
        print(f"  🔴 미흡 (60-69): {dist['poor_60_69']}개 ({dist['poor_60_69']/gemba_stats['total_samples']*100:.1f}%)")
        print(f"  ❌ 실패 (<60): {dist['failing_below_60']}개 ({dist['failing_below_60']/gemba_stats['total_samples']*100:.1f}%)")
        
        # 길이별 분석
        print(f"\n📏 길이별 성능:")
        bucket_stats = gemba_stats['bucket_analysis']
        for bucket, stats in bucket_stats.items():
            print(f"  {bucket:12}: {stats['count']:3}개, 평균 {stats['avg_score']:5.1f}, 통과율 {stats['pass_rate']:5.1f}%")
        
        # 2. 품질 결정 분석
        quality_stats = self.analyze_quality_decisions()
        print(f"\n🎯 품질 결정 분석")
        print("-" * 40)
        print(f"전체 통과율: {quality_stats['pass_rate']:.1f}%")
        print(f"Strict Pass: {quality_stats['strict_pass_rate']:.1f}%")
        print(f"Soft Pass: {quality_stats['soft_pass_rate']:.1f}%")
        print(f"실패율: {quality_stats['fail_rate']:.1f}%")
        
        if quality_stats['failure_reasons']:
            print(f"\n실패 원인:")
            for reason, count in quality_stats['failure_reasons'].items():
                print(f"  {reason}: {count}개")
        
        # 3. Evidence 품질 분석
        evidence_stats = self.analyze_evidence_quality()
        print(f"\n💬 Evidence 품질 분석")
        print("-" * 40)
        print(f"평균 Evidence 길이: {evidence_stats['avg_evidence_length']:.1f} 글자")
        print(f"구체적 피드백 비율: {evidence_stats['detailed_feedback_rate']:.1f}%")
        
        ev_types = evidence_stats['evidence_types']
        total = sum(ev_types.values())
        print(f"\nEvidence 유형별 분포:")
        print(f"  완벽/우수: {ev_types['perfect_excellent']}개 ({ev_types['perfect_excellent']/total*100:.1f}%)")
        print(f"  구체적 이슈: {ev_types['specific_issues']}개 ({ev_types['specific_issues']/total*100:.1f}%)")
        print(f"  일반적 피드백: {ev_types['generic_feedback']}개 ({ev_types['generic_feedback']/total*100:.1f}%)")
        print(f"  빈약/최소: {ev_types['empty_or_minimal']}개 ({ev_types['empty_or_minimal']/total*100:.1f}%)")
        
        # 4. 기준선 대비 개선도
        comparison = self.compare_with_baseline()
        print(f"\n🚀 개선 효과 (GPT-3.5-turbo 대비)")
        print("-" * 40)
        improvements = comparison['improvements']
        print(f"평균 점수 개선: {improvements['score_improvement']:+.1f}점")
        print(f"통과율 개선: {improvements['pass_rate_improvement']:+.1f}%")
        print(f"구체적 피드백 개선: {improvements['evidence_quality_improvement']:+.1f}%")
        print(f"Evidence 길이 개선: {improvements['evidence_length_improvement']:+.1f} 글자")
        
        # 5. 종합 평가
        print(f"\n📋 종합 평가")
        print("=" * 40)
        score_improvement = improvements['score_improvement']
        if score_improvement > 5:
            print("🎉 우수: 상당한 점수 개선 달성!")
        elif score_improvement > 2:
            print("✅ 양호: 의미있는 개선 확인")
        elif score_improvement > 0:
            print("⚠️  보통: 약간의 개선 있음")
        else:
            print("🔴 미흡: 개선 필요")
        
        # 추천 사항
        print(f"\n💡 추천 사항:")
        if evidence_stats['detailed_feedback_rate'] < 60:
            print("- Evidence 품질 더 향상 필요 (더 구체적인 피드백)")
        if quality_stats['fail_rate'] > 30:
            print("- 실패율이 높음: 임계값 조정 검토 필요")
        if gemba_stats['overall_stats']['std_overall'] > 15:
            print("- 점수 편차가 큼: 더 일관된 평가 필요")
        
        print(f"\n분석 완료! 🎯")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GEMBA 개선 효과 분석")
    parser.add_argument("--version", "-v", default="v10", help="분석할 버전 (기본: v10)")
    parser.add_argument("--output", "-o", help="결과를 JSON으로 저장할 경로")
    
    args = parser.parse_args()
    
    # 결과 디렉토리 설정 - optimized_pipeline의 out 디렉토리 확인
    base_dir = Path(__file__).parent
    result_dir = base_dir / "out" / args.version
    
    # optimized_pipeline도 확인
    optimized_result_dir = base_dir.parent / "optimized_pipeline" / "out" / args.version
    
    if result_dir.exists():
        target_dir = result_dir
    elif optimized_result_dir.exists():
        target_dir = optimized_result_dir
        print(f"📁 optimized_pipeline의 결과 사용: {target_dir}")
    else:
        print(f"❌ 결과 디렉토리가 없습니다:")
        print(f"   {result_dir}")
        print(f"   {optimized_result_dir}")
        return
    
    # 분석 실행
    analyzer = GembaAnalyzer(target_dir)
    analyzer.print_analysis_report()
    
    # JSON 출력 (선택사항)
    if args.output:
        if analyzer.load_results():
            analysis_data = {
                'gemba_scores': analyzer.analyze_gemba_scores(),
                'quality_decisions': analyzer.analyze_quality_decisions(),
                'evidence_quality': analyzer.analyze_evidence_quality(),
                'comparison': analyzer.compare_with_baseline()
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            print(f"📁 분석 결과가 {args.output}에 저장되었습니다.")

if __name__ == "__main__":
    main()
