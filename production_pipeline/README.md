# Production MT Quality Evaluation Pipeline

이 폴더는 **완성된 번역품질 평가 파이프라인**의 프로덕션 버전입니다.
분석/실험용 코드들을 제거하고 핵심 기능만 남겼습니다.

## 구조

```
production_pipeline/
├── run_pipeline.py          # 메인 파이프라인 실행
├── cfg.py                   # 설정 파일
├── filter.py                # 데이터 필터링
├── gemba_batch.py           # GEMBA 품질 평가
├── ape+back_translation.py  # APE 및 백번역
├── requirements.txt         # Python 의존성
├── api/                     # REST API 서버
│   ├── main.py             # FastAPI 메인 서버
│   ├── index.html          # 웹 인터페이스
│   └── requirements.txt    # API 의존성
└── data/                   # 입력/출력 데이터
    └── out/                # 파이프라인 실행 결과

## 실행 방법

### 1. 파이프라인 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 파이프라인 실행
python run_pipeline.py
```

### 2. API 서버 실행
```bash
cd api
pip install -r requirements.txt
python main.py
```

### 3. 웹 인터페이스 접속
브라우저에서 `http://localhost:8000` 접속

## 핵심 기능

### 파이프라인
- **필터링**: 중복 제거, 길이 기반 분류
- **GEMBA 평가**: GPT 기반 번역품질 평가
- **APE**: 자동 후편집으로 번역 개선
- **백번역**: 품질 검증 (선택사항)

### 웹 인터페이스
- 번역품질 데이터 조회 및 필터링
- APE 개선 효과 확인
- 품질 등급별 분류 및 통계
- 간단한 데이터 내보내기

## 설정

`cfg.py` 파일에서 다음을 설정할 수 있습니다:
- API 키 (OpenAI, Anthropic)
- 입력 데이터 경로
- 배치 크기 및 처리 옵션
- 품질 임계값

## 데이터 형식

### 입력 형식 (JSON Lines)
```json
{
  "key": "unique_id",
  "src": "source text",
  "mt": "machine translation"
}
```

### 출력 형식
```json
{
  "key": "unique_id",
  "src": "source text", 
  "mt": "machine translation",
  "ape": "improved translation",
  "gemba": 85.5,
  "comet": 0.82,
  "cos": 0.87,
  "delta_comet": 0.05,
  "delta_cos": 0.03,
  "tag": "strict_pass",
  "bucket": "medium"
}
```

## 제거된 분석/실험 코드들

다음 파일들은 프로덕션에서 제외되었습니다:
- `analyze.py` - 결과 분석
- `threshold_optimizer.py` - 임계값 최적화
- `iterative_optimizer.py` - 반복 최적화
- `precision_optimizer.py` - 정밀도 튜닝
- `test_*.py` - 테스트 스크립트들
- `*_analyzer.py` - 각종 분석 도구들
- 최적화 리포트 JSON 파일들
- 실험용 설정 백업 파일들

이러한 도구들은 `optimized_pipeline` 폴더에서 계속 사용할 수 있습니다.
