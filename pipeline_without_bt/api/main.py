from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import json
import statistics
from pathlib import Path
from enum import Enum

DELTA_GEMBA_FILE = Path("../out/validation/ko_en_conv_prompt.json")
ORIGINAL_FILE = Path("../out/validation/ko_en_conv_prompt.json")

# DELTA_GEMBA_FILE = Path("../out/KURE_v1/ape_evidence.json")
# ORIGINAL_FILE = Path("../out/KURE_v1/ape_evidence.json")

# DELTA_GEMBA_FILE = Path("../out/KoE5/ape_evidence.json")
# ORIGINAL_FILE = Path("../out/KoE5/ape_evidence.json")

DATA_FILE = DELTA_GEMBA_FILE if DELTA_GEMBA_FILE.exists() else ORIGINAL_FILE

class BucketType(str, Enum):
    very_short = "very_short"
    short = "short" 
    medium = "medium"
    long = "long"
    very_long = "very_long"

class TagType(str, Enum):
    strict_pass = "strict_pass"
    soft_pass = "soft_pass"
    fail = "fail"

evaluation_data: List[Dict[Any, Any]] = []

def get_quality_grade_by_scores(gemba: float, comet: float, cos: float) -> str:
    """GEMBA, COMET, Cosine 점수를 모두 고려한 품질등급 반환"""
    # HTML의 getQualityClassByScores 로직과 동일
    if gemba >= 85 and comet >= 0.80 and cos >= 0.85:
        return "excellent"  # 매우우수
    elif gemba >= 75 and comet >= 0.75 and cos >= 0.75:
        return "very_good"  # 우수
    elif gemba >= 65 and comet >= 0.70 and cos >= 0.70:
        return "good"       # 양호
    elif gemba >= 45 and comet >= 0.50 and cos >= 0.50:
        return "poor"       # 나쁨
    else:
        return "very_poor"  # 매우나쁨

def calculate_meaningful_improvement_rate() -> float:
    """품질등급이 상승한 케이스의 비율을 계산 - 3개 메트릭 종합 기준"""
    if not evaluation_data:
        return 0.0
    
    # APE가 적용된 레코드들만 필터링 (delta 값이 하나라도 있는 경우)
    ape_records = [r for r in evaluation_data if "ape" in r and 
                   ("delta_gemba" in r or "delta_comet" in r or "delta_cos" in r)]
    
    if not ape_records:
        return 0.0
    
    improved_count = 0
    debug_info = []
    
    for record in ape_records:
        # 원본 점수들
        original_gemba = record.get("gemba", 0)
        original_comet = record.get("comet", 0)
        original_cos = record.get("cos", 0)
        
        # Delta 값들 (기본값 0)
        delta_gemba = record.get("delta_gemba", 0)
        delta_comet = record.get("delta_comet", 0)
        delta_cos = record.get("delta_cos", 0)
        
        # 개선 후 점수들
        improved_gemba = original_gemba + delta_gemba
        improved_comet = original_comet + delta_comet
        improved_cos = original_cos + delta_cos
        
        # 원본 품질등급과 개선 후 품질등급 계산
        original_grade = get_quality_grade_by_scores(original_gemba, original_comet, original_cos)
        improved_grade = get_quality_grade_by_scores(improved_gemba, improved_comet, improved_cos)
        
        # 품질등급 순서 (숫자가 클수록 높은 등급)
        grade_order = {
            "very_poor": 1,
            "poor": 2,
            "good": 3,
            "very_good": 4,
            "excellent": 5
        }
        
        # 품질등급이 올라간 경우만 카운트
        grade_improved = grade_order.get(improved_grade, 0) > grade_order.get(original_grade, 0)
        if grade_improved:
            improved_count += 1
        
        # 디버깅 정보 수집 (모든 케이스 저장)
        debug_info.append({
            "key": record.get("key", "unknown"),
            "original_scores": f"G:{original_gemba:.0f}/C:{original_comet:.3f}/S:{original_cos:.3f}",
            "delta_scores": f"G:{delta_gemba:.0f}/C:{delta_comet:.3f}/S:{delta_cos:.3f}",
            "improved_scores": f"G:{improved_gemba:.0f}/C:{improved_comet:.3f}/S:{improved_cos:.3f}",
            "original_grade": original_grade,
            "improved_grade": improved_grade,
            "grade_improved": grade_improved
        })
    
    # 디버깅 정보 출력
    print(f"\n=== 품질등급 상승률 계산 디버깅 (3개 메트릭 종합) ===")
    print(f"총 APE 레코드 수: {len(ape_records)}")
    print(f"품질등급 상승 케이스: {improved_count}")
    print(f"상승률: {(improved_count / len(ape_records)) * 100:.1f}%")
    
    # 등급별 분포 계산
    grade_distribution = {}
    improvement_by_grade = {}
    
    for info in debug_info:
        original = info['original_grade']
        improved = info['improved_grade']
        
        # 원본 등급별 분포
        grade_distribution[original] = grade_distribution.get(original, 0) + 1
        
        # 원본 등급별 개선 케이스
        if info['grade_improved']:
            improvement_by_grade[original] = improvement_by_grade.get(original, 0) + 1
    
    print(f"\n등급별 분포 및 개선률:")
    for grade in ["very_poor", "poor", "good", "very_good", "excellent"]:
        total = grade_distribution.get(grade, 0)
        improved = improvement_by_grade.get(grade, 0)
        if total > 0:
            rate = (improved / total) * 100
            print(f"  {grade}: {total}개 중 {improved}개 개선 ({rate:.1f}%)")
    
    print(f"\n처음 30개 레코드 상세:")
    for i, info in enumerate(debug_info[:30]):
        print(f"  {info['key']}: {info['original_scores']} → {info['improved_scores']}")
        print(f"    등급: {info['original_grade']} → {info['improved_grade']} {'✓' if info['grade_improved'] else '✗'}")
    
    if len(debug_info) > 30:
        print(f"  ... (총 {len(debug_info)}개 중 30개만 표시)")
    
    print("="*50)
    
    return (improved_count / len(ape_records)) * 100 if ape_records else 0.0

def get_quality_distribution_before_after() -> Dict[str, Any]:
    """APE 이전과 이후의 품질등급 분포를 계산"""
    if not evaluation_data:
        return {"before": {}, "after": {}, "total_records": 0}
    
    before_distribution = {"very_poor": 0, "poor": 0, "good": 0, "very_good": 0, "excellent": 0}
    after_distribution = {"very_poor": 0, "poor": 0, "good": 0, "very_good": 0, "excellent": 0}
    
    # 모든 레코드에 대해 APE 이전/이후 품질등급 계산
    for record in evaluation_data:
        # 원본 점수들
        original_gemba = record.get("gemba", 0)
        original_comet = record.get("comet", 0)
        original_cos = record.get("cos", 0)
        
        # APE 이전 품질등급
        before_grade = get_quality_grade_by_scores(original_gemba, original_comet, original_cos)
        before_distribution[before_grade] += 1
        
        # APE가 적용된 경우 개선 후 점수 계산
        if "ape" in record and ("delta_gemba" in record or "delta_comet" in record or "delta_cos" in record):
            delta_gemba = record.get("delta_gemba", 0)
            delta_comet = record.get("delta_comet", 0)
            delta_cos = record.get("delta_cos", 0)
            
            improved_gemba = original_gemba + delta_gemba
            improved_comet = original_comet + delta_comet
            improved_cos = original_cos + delta_cos
            
            after_grade = get_quality_grade_by_scores(improved_gemba, improved_comet, improved_cos)
        else:
            # APE가 적용되지 않은 경우 원본과 동일
            after_grade = before_grade
            
        after_distribution[after_grade] += 1
    
    total_records = len(evaluation_data)
    
    return {
        "before": before_distribution,
        "after": after_distribution,
        "total_records": total_records,
        "ape_applied_count": len([r for r in evaluation_data if "ape" in r])
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global evaluation_data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'records' in data:
            evaluation_data = data['records']
            print(f"Loaded {len(evaluation_data)} evaluation records (with delta_gemba)")
            if 'metadata' in data and data['metadata'].get('delta_gemba_added'):
                print("Delta GEMBA 데이터 포함됨")
        else:
            evaluation_data = data
            print(f"Loaded {len(evaluation_data)} evaluation records (original format)")
            
    except FileNotFoundError:
        print(f"Data file not found: {DATA_FILE}")
        evaluation_data = []
    yield

app = FastAPI(
    title="MT Quality Evaluation API",
    description="API for accessing machine translation quality evaluation results",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API status and basic info"""
    return {
        "message": "MT Quality Evaluation API",
        "total_records": len(evaluation_data),
        "version": "1.0.0",
        "endpoints": [
            "/records - Get all records with filtering",
            "/records/{key} - Get specific record by key", 
            "/analytics - Get summary analytics",
            "/buckets/{bucket} - Get records by text length bucket",
            "/tags/{tag} - Get records by quality tag"
        ]
    }

@app.get("/records")
async def get_records(
    bucket: Optional[BucketType] = None,
    tag: Optional[TagType] = None,
    min_gemba: Optional[float] = Query(None, ge=0, le=100),
    max_gemba: Optional[float] = Query(None, ge=0, le=100),
    min_comet: Optional[float] = Query(None, ge=0, le=1),
    has_ape: Optional[bool] = None,
    limit: Optional[int] = Query(100, ge=1, le=10000),
    offset: Optional[int] = Query(0, ge=0)
):
    """Get evaluation records with optional filtering"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    filtered_data = evaluation_data
    
    if bucket:
        filtered_data = [r for r in filtered_data if r.get("bucket") == bucket]
    
    if tag:
        filtered_data = [r for r in filtered_data if r.get("tag") == tag]
    
    if min_gemba is not None:
        filtered_data = [r for r in filtered_data if r.get("gemba", 0) >= min_gemba]
    
    if max_gemba is not None:
        filtered_data = [r for r in filtered_data if r.get("gemba", 100) <= max_gemba]
    
    if min_comet is not None:
        filtered_data = [r for r in filtered_data if r.get("comet", 0) >= min_comet]
    
    if has_ape is not None:
        if has_ape:
            filtered_data = [r for r in filtered_data if "ape" in r]
        else:
            filtered_data = [r for r in filtered_data if "ape" not in r]
    
    total = len(filtered_data)
    paginated_data = filtered_data[offset:offset + limit]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "records": paginated_data
    }

@app.get("/records/{key}")
async def get_record_by_key(key: str):
    """Get a specific record by its key"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    record = next((r for r in evaluation_data if r.get("key") == key), None)
    
    if not record:
        raise HTTPException(status_code=404, detail=f"Record with key '{key}' not found")
    
    return record

@app.get("/analytics")
async def get_analytics():
    """Get summary analytics of all evaluation data"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Basic statistics
    gemba_scores = [r.get("gemba", 0) for r in evaluation_data if "gemba" in r]
    comet_scores = [r.get("comet", 0) for r in evaluation_data if "comet" in r]
    cos_scores = [r.get("cos", 0) for r in evaluation_data if "cos" in r]
    
    # Tag distribution
    tag_dist = {}
    for record in evaluation_data:
        tag = record.get("tag", "unknown")
        tag_dist[tag] = tag_dist.get(tag, 0) + 1
    
    # Bucket distribution
    bucket_dist = {}
    for record in evaluation_data:
        bucket = record.get("bucket", "unknown")
        bucket_dist[bucket] = bucket_dist.get(bucket, 0) + 1
    
    # APE effectiveness
    ape_records = [r for r in evaluation_data if "ape" in r]
    ape_improvements = {
        "delta_comet": [r.get("delta_comet", 0) for r in ape_records if "delta_comet" in r],
        "delta_cos": [r.get("delta_cos", 0) for r in ape_records if "delta_cos" in r],
        "delta_gemba": [r.get("delta_gemba", 0) for r in ape_records if "delta_gemba" in r]
    }
    
    return {
        "total_records": len(evaluation_data),
        "scores": {
            "gemba": {
                "mean": statistics.mean(gemba_scores) if gemba_scores else 0,
                "median": statistics.median(gemba_scores) if gemba_scores else 0,
                "stdev": statistics.stdev(gemba_scores) if len(gemba_scores) > 1 else 0,
                "min": min(gemba_scores) if gemba_scores else 0,
                "max": max(gemba_scores) if gemba_scores else 0
            },
            "comet": {
                "mean": statistics.mean(comet_scores) if comet_scores else 0,
                "median": statistics.median(comet_scores) if comet_scores else 0,
                "stdev": statistics.stdev(comet_scores) if len(comet_scores) > 1 else 0
            },
            "cosine": {
                "mean": statistics.mean(cos_scores) if cos_scores else 0,
                "median": statistics.median(cos_scores) if cos_scores else 0,
                "stdev": statistics.stdev(cos_scores) if len(cos_scores) > 1 else 0
            }
        },
        "distributions": {
            "tags": tag_dist,
            "buckets": bucket_dist
        },
        "ape_effectiveness": {
            "total_ape_records": len(ape_records),
            "avg_comet_improvement": statistics.mean(ape_improvements["delta_comet"]) if ape_improvements["delta_comet"] else 0,
            "avg_cosine_improvement": statistics.mean(ape_improvements["delta_cos"]) if ape_improvements["delta_cos"] else 0,
            "avg_gemba_improvement": statistics.mean(ape_improvements["delta_gemba"]) if ape_improvements["delta_gemba"] else 0,
            "meaningful_improvement_rate": calculate_meaningful_improvement_rate()
        },
        "quality_distribution": get_quality_distribution_before_after()
    }

@app.get("/buckets/{bucket}")
async def get_records_by_bucket(bucket: BucketType):
    """Get all records for a specific text length bucket"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    bucket_records = [r for r in evaluation_data if r.get("bucket") == bucket]
    
    if not bucket_records:
        raise HTTPException(status_code=404, detail=f"No records found for bucket '{bucket}'")
    
    # Compute bucket statistics
    gemba_scores = [r.get("gemba", 0) for r in bucket_records]
    
    return {
        "bucket": bucket,
        "total_records": len(bucket_records),
        "avg_gemba": statistics.mean(gemba_scores) if gemba_scores else 0,
        "pass_rate": len([r for r in bucket_records if r.get("tag") in ["strict_pass", "soft_pass"]]) / len(bucket_records) * 100,
        "records": bucket_records
    }

@app.get("/tags/{tag}")
async def get_records_by_tag(tag: TagType):
    """Get all records for a specific quality tag"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    tag_records = [r for r in evaluation_data if r.get("tag") == tag]
    
    if not tag_records:
        raise HTTPException(status_code=404, detail=f"No records found for tag '{tag}'")
    
    return {
        "tag": tag,
        "total_records": len(tag_records),
        "records": tag_records
    }

@app.get("/search")
async def search_records(
    q: str = Query(..., min_length=2, description="Search query for source or translation text"),
    limit: int = Query(50, ge=1, le=200)
):
    """Search records by text content in source or translation"""
    if not evaluation_data:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    query = q.lower()
    matches = []
    
    for record in evaluation_data:
        src = record.get("src", "").lower()
        mt = record.get("mt", "").lower()
        ape = record.get("ape", "").lower()
        
        if query in src or query in mt or query in ape:
            matches.append(record)
        
        if len(matches) >= limit:
            break
    
    return {
        "query": q,
        "total_matches": len(matches),
        "records": matches
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
