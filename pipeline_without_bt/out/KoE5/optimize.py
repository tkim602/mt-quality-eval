#!/usr/bin/env python
# check_quality.py
# ──────────────────────────────────────────────────────────
# ① 평균·표준편차 계산 & 저장
# ② Cos/COMET/GEMBA 가중치 최적화   (--optimize)
# ③ Q-점수 기반 새 품질 등급 리포트
# ──────────────────────────────────────────────────────────

import json, argparse, itertools, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
from tabulate import tabulate

# 하이퍼파라미터
T_FAIL, T_STRICT = -1.0, -0.10          # Q 임계치
STEP_W  = 0.1                            # 가중치 그리드 해상도 (빠르게)
STEP_T  = 0.2                            # threshold 그리드 해상도 for optimize (빠르게)

# 등급 경계
def grade_from_q(q):
    if q >= 0.6:  return 'excellent'
    if q >= 0.3:  return 'very_good'
    if q >= -0.1: return 'good'
    if q >= -0.5: return 'poor'
    return 'very_poor'

def disagree(row):
    return (row[['cos','comet','gemba']].max()
          - row[['cos','comet','gemba']].min()) > 0.2

# ──────────────────────────────────────────────────────────
def load_df(path):
    return pd.json_normalize(json.load(open(path, encoding='utf-8')))

def calc_z(df, mean, std):
    return (df[['cos','comet','gemba']] - pd.Series(mean)) / pd.Series(std)

def calc_q(z, w):
    return w['cos']*z['cos'] + w['comet']*z['comet'] + w['gemba']*z['gemba']

def tag_from_q(q):
    if q < T_FAIL:           return 'fail'
    if q >= T_STRICT:        return 'strict_pass'
    return 'soft_pass'

def optimize_weights_fast(z, true_labels):
    """빠른 최적화: 단계별 접근"""
    print("[INFO] Phase 1: Finding best weights (fixed thresholds)...")
    
    # Phase 1: 고정 threshold로 최적 가중치 찾기
    best_weights = None
    best_f1_phase1 = -1
    weights = np.arange(0, 1+1e-9, STEP_W)
    
    for w_cos, w_comet in itertools.product(weights, weights):
        if w_cos + w_comet > 1:
            continue
        w_gemba = 1 - w_cos - w_comet
        w = {'cos':w_cos, 'comet':w_comet, 'gemba':w_gemba}
        q = calc_q(z, w)
        
        # 고정 threshold 사용
        pred = np.where(q >= T_STRICT, 2,
               np.where(q < T_FAIL, 0, 1))
        f1 = f1_score(true_labels, pred, average='macro')
        
        if f1 > best_f1_phase1:
            best_f1_phase1 = f1
            best_weights = w
    
    print(f"[INFO] Phase 1 complete. Best weights: {best_weights}, F1: {best_f1_phase1:.4f}")
    
    # Phase 2: 최적 가중치로 threshold 최적화
    print("[INFO] Phase 2: Optimizing thresholds with best weights...")
    q = calc_q(z, best_weights)
    best_f1_final = -1
    best_thresholds = (T_STRICT, T_FAIL)
    
    # threshold 범위를 더 좁게 탐색
    for thr_strict in np.arange(-1, 1+1e-9, STEP_T):
        for thr_fail in np.arange(-2, thr_strict-0.1, STEP_T):
            pred = np.where(q >= thr_strict, 2,
                   np.where(q < thr_fail, 0, 1))
            try:
                f1 = f1_score(true_labels, pred, average='macro')
                if f1 > best_f1_final:
                    best_f1_final = f1
                    best_thresholds = (thr_strict, thr_fail)
            except:
                continue  # skip invalid predictions
    
    print(f"[INFO] Phase 2 complete. Best thresholds: T_STRICT={best_thresholds[0]:.2f}, T_FAIL={best_thresholds[1]:.2f}")
    print(f"[INFO] Final F1: {best_f1_final:.4f}")
    
    return best_f1_final, best_weights, best_thresholds[0], best_thresholds[1]

def save_stats(fp, mean, std, weights, thresholds=None):
    data = {'mean':mean, 'std':std, 'weights':weights}
    if thresholds:
        data['optimal_thresholds'] = thresholds
    fp.write_text(json.dumps(data, indent=2))

def load_stats(fp):
    return json.loads(fp.read_text()) if fp.exists() else None

# ──────────────────────────────────────────────────────────
def main(args):
    df = load_df(args.json)

    # 1) 통계 불러오거나 계산
    stats_fp = Path(args.stats)
    stats     = load_stats(stats_fp)
    if stats is None:
        mean = df[['cos','comet','gemba']].mean().to_dict()
        std  = df[['cos','comet','gemba']].std().to_dict()
        weights = {'cos':0.4,'comet':0.3,'gemba':0.3}   # 초기값
    else:
        mean, std, weights = stats['mean'], stats['std'], stats['weights']

    # 2) 가중치 최적화 (선택)
    z = calc_z(df, mean, std)
    true = df['tag'].map({'fail':0,'soft_pass':1,'strict_pass':2}).values
    if args.optimize:
        best_f1, weights, best_thr_strict, best_thr_fail = optimize_weights_fast(z, true)
        print(f"\n[OPT] BEST RESULT:")
        print(f"      macro-F1={best_f1:.4f}")
        print(f"      weights={weights}")
        print(f"      T_STRICT={best_thr_strict:.2f} (for strict_pass)")
        print(f"      T_FAIL={best_thr_fail:.2f} (for fail)")
        print(f"\n[RECOMMENDATION] Update your cfg.py with these optimal thresholds!")
        
        # 최적 임계값으로 전역 변수 업데이트
        global T_FAIL, T_STRICT
        T_FAIL, T_STRICT = best_thr_fail, best_thr_strict

    # 3) Q / 새 태그 / 배지
    df['Q'] = calc_q(z, weights)
    df['new_tag']       = df['Q'].apply(tag_from_q)
    df['overall_grade'] = df['Q'].apply(grade_from_q)
    df['disagree']      = df.apply(disagree, axis=1)

    # 4) 리포트
    def summary(title, series):
        print(f"\n=== {title} ===")
        print(tabulate(series.value_counts().rename_axis(title.lower())
                       .reset_index(name='count'),
                       headers=[title.lower(),'count'], tablefmt='github'))
    summary('Strict/Soft/Fail', df['new_tag'])
    summary('Quality Badge', df['overall_grade'])

    cm = confusion_matrix(true,
             df['new_tag'].map({'fail':0,'soft_pass':1,'strict_pass':2}),
             labels=[0,1,2])
    print("\n=== Confusion Matrix ===")
    print(tabulate(cm, headers=['pred_fail','pred_soft','pred_strict'],
                   showindex=['true_fail','true_soft','true_strict'],
                   tablefmt='github'))
    print("\nmacro F1 =", f1_score(true,
                                   df['new_tag'].map({'fail':0,'soft_pass':1,'strict_pass':2}),
                                   average='macro'))

    # 5) stats 저장
    if args.save_stats:
        optimal_thresholds = None
        if args.optimize:
            optimal_thresholds = {'T_STRICT': T_STRICT, 'T_FAIL': T_FAIL}
        save_stats(stats_fp, mean, std, weights, optimal_thresholds)
        print(f"\n[INFO] stats saved → {stats_fp}")
        if optimal_thresholds:
            print(f"[INFO] optimal thresholds saved: {optimal_thresholds}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('json',          help='ape_evidence.json 경로')
    ap.add_argument('--stats', default='stats.json',
                    help='평균·표준편차·가중치 저장/로드 파일')
    ap.add_argument('--save-stats',  action='store_true',
                    help='계산한 통계를 stats.json에 저장')
    ap.add_argument('--optimize',    action='store_true',
                    help='가중치/threshold 그리드 서치로 최적화')
    main(ap.parse_args())
