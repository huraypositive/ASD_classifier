"""
RFMiD 데이터셋을 이용한 ASD / Non-ASD 분류 테스트 스크립트

레이블 정의:
  ASD     : ODC==1 OR TV==1 OR ODE==1, 나머지 disease 컬럼은 전부 0
  Non-ASD : Disease_Risk == 0

평가 대상: Training / Validation / Test set 전체
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.model_ResNet import resnext_scratch
from src.image_prep import maybe_fix_bgr_rgb, get_mask, mask_image, remove_back_area, supplemental_black_area
from main import preprocess, load_models


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
SPLITS = {
    'training':   ('RFMiD/Training_set/RFMiD_Training_Labels.csv',   'RFMiD/Training_set'),
    'validation': ('RFMiD/Validation_set/RFMiD_Validation_Labels.csv', 'RFMiD/Validation_set'),
    'test':       ('RFMiD/Test_set/RFMiD_Testing_Labels.csv',         'RFMiD/Test_set'),
}

DISEASE_COLS = [
    'DR','ARMD','MH','DN','MYA','BRVO','TSLN','ERM','LS','MS','CSR',
    'ODC','CRVO','TV','AH','ODP','ODE','ST','AION','PT','RT','RS','CRS',
    'EDN','RPEC','MHL','RP','CWS','CB','ODPM','PRH','MNF','HR','CRAO',
    'TD','CME','PTCR','CF','VH','MCA','VS','BRAO','PLQ','HPED','CL'
]
ASD_COLS        = ['ODC', 'TV', 'ODE']
# ASD_COLS        = ['ODC', 'TV']
OTHER_COLS      = [c for c in DISEASE_COLS if c not in ASD_COLS]

# True  : Non-ASD 수를 ASD 수에 맞게 다운샘플링하여 균형 잡힌 데이터로 테스트
# False : 원본 비율 그대로 테스트
BALANCE_DATASET = True


# ---------------------------------------------------------------------------
# 레이블 파싱
# ---------------------------------------------------------------------------
def parse_labels(csv_path: str, image_dir: str) -> pd.DataFrame:
    """
    CSV에서 ASD(1) / Non-ASD(0) 샘플만 추출한다.
    이미지 파일이 실제로 존재하는 샘플만 반환한다.

    반환: DataFrame columns = [ID, image_path, label]
    """
    df = pd.read_csv(csv_path)

    asd_mask = (
        ((df['ODC'] == 1) | (df['TV'] == 1) | (df['ODE'] == 1)) &
        (df[OTHER_COLS].sum(axis=1) == 0)
    )
    non_asd_mask = df['Disease_Risk'] == 0

    asd_df     = df[asd_mask].copy()
    non_asd_df = df[non_asd_mask].copy()

    asd_df['label']     = 1
    non_asd_df['label'] = 0

    result = pd.concat([asd_df, non_asd_df], ignore_index=True)[['ID', 'label']]

    image_dir = Path(image_dir)
    result['image_path'] = result['ID'].apply(lambda x: image_dir / f'{x}.png')

    # 이미지 파일 존재 여부 확인
    exists = result['image_path'].apply(lambda p: p.exists())
    missing = (~exists).sum()
    if missing > 0:
        print(f'  [경고] 이미지 파일 없음: {missing}건 제외')
    result = result[exists].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# 데이터 균형 조정
# ---------------------------------------------------------------------------
def balance_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Non-ASD 샘플 수를 ASD 샘플 수에 맞게 랜덤 다운샘플링한다.
    BALANCE_DATASET=True 일 때만 호출된다.
    """
    asd_df     = df[df['label'] == 1]
    non_asd_df = df[df['label'] == 0].sample(n=len(asd_df), random_state=seed)
    balanced   = pd.concat([asd_df, non_asd_df], ignore_index=True)
    balanced   = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


# ---------------------------------------------------------------------------
# 단일 이미지 추론
# ---------------------------------------------------------------------------
def infer_single_image(models: list, image_path: Path) -> float:
    """
    이미지 1장에 대해 앙상블 모델로 ASD 확률을 반환한다.
    전처리는 main.py의 preprocess()를 그대로 사용한다.
    """
    img_tensor = preprocess(image_path)
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    with torch.no_grad():
        outputs = []
        for model in models:
            output = model(img_tensor)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        avg_output = torch.mean(outputs, dim=0)
        avg_output = torch.softmax(avg_output, dim=1)
        asd_prob = avg_output[0][1].item()

    return asd_prob


# ---------------------------------------------------------------------------
# 평가 지표 계산 및 출력
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'threshold':   threshold,
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'auc_roc':     roc_auc_score(y_true, y_prob),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def print_metrics(split_name: str, metrics: dict):
    print(f'\n{"="*50}')
    print(f'  [{split_name.upper()}] 평가 결과  (threshold={metrics["threshold"]:.2f})')
    print(f'{"="*50}')
    print(f'  Accuracy    : {metrics["accuracy"]:.4f}')
    print(f'  Sensitivity : {metrics["sensitivity"]:.4f}  (Recall / TPR)')
    print(f'  Specificity : {metrics["specificity"]:.4f}  (TNR)')
    print(f'  Precision   : {metrics["precision"]:.4f}')
    print(f'  F1 Score    : {metrics["f1"]:.4f}')
    print(f'  AUC-ROC     : {metrics["auc_roc"]:.4f}')
    print(f'  Confusion Matrix:')
    print(f'    TP={metrics["tp"]}  FP={metrics["fp"]}')
    print(f'    FN={metrics["fn"]}  TN={metrics["tn"]}')


# ---------------------------------------------------------------------------
# split별 평가
# ---------------------------------------------------------------------------
def evaluate_split(split_name: str, models: list, threshold: float, output_dir: Path):
    csv_path, image_dir = SPLITS[split_name]
    print(f'\n[{split_name}] 레이블 파싱 중...')
    df = parse_labels(csv_path, image_dir)

    if BALANCE_DATASET:
        df = balance_dataset(df)
        print(f'  [균형 조정] Non-ASD를 ASD 수에 맞게 다운샘플링')

    asd_n     = (df['label'] == 1).sum()
    non_asd_n = (df['label'] == 0).sum()
    print(f'  ASD={asd_n}, Non-ASD={non_asd_n}, 총={len(df)}')

    # 추론
    probs = []
    errors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'  추론 [{split_name}]'):
        try:
            prob = infer_single_image(models, row['image_path'])
            probs.append(prob)
        except Exception as e:
            print(f'\n  [오류] ID={row["ID"]}: {e}')
            probs.append(np.nan)
            errors.append(row['ID'])

    df['asd_prob']  = probs
    df['pred_label'] = (df['asd_prob'] >= threshold).astype('Int64')

    # 오류 샘플 제외 후 평가
    valid = df.dropna(subset=['asd_prob'])
    if len(valid) < len(df):
        print(f'  [경고] 추론 실패 {len(df)-len(valid)}건 제외 후 평가')

    y_true = valid['label'].values
    y_prob = valid['asd_prob'].values

    metrics = compute_metrics(y_true, y_prob, threshold)
    print_metrics(split_name, metrics)

    # 상세 결과 CSV 저장
    output_path = output_dir / f'result_{split_name}.csv'
    df[['ID', 'label', 'asd_prob', 'pred_label']].to_csv(output_path, index=False)
    print(f'  상세 결과 저장: {output_path}')

    return metrics, y_true, y_prob


# ---------------------------------------------------------------------------
# 전체 요약
# ---------------------------------------------------------------------------
def print_summary(all_metrics: dict):
    print(f'\n{"="*50}')
    print('  전체 요약')
    print(f'{"="*50}')
    header = f'  {"Split":<12} {"Acc":>6} {"Sens":>6} {"Spec":>6} {"F1":>6} {"AUC":>6}'
    print(header)
    print(f'  {"-"*48}')
    split_names = [n for n in all_metrics if n != 'overall']
    for name in split_names:
        m = all_metrics[name]
        print(f'  {name:<12} {m["accuracy"]:>6.4f} {m["sensitivity"]:>6.4f} '
              f'{m["specificity"]:>6.4f} {m["f1"]:>6.4f} {m["auc_roc"]:>6.4f}')
    if 'overall' in all_metrics:
        m = all_metrics['overall']
        print(f'  {"-"*48}')
        print(f'  {"overall":<12} {m["accuracy"]:>6.4f} {m["sensitivity"]:>6.4f} '
              f'{m["specificity"]:>6.4f} {m["f1"]:>6.4f} {m["auc_roc"]:>6.4f}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='RFMiD ASD 분류 테스트')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='ASD 판정 확률 임계값 (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='결과 저장 디렉토리 (default: test_results)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 모델 로드 (10개 fold 앙상블)
    model_paths = [f'models/Normal_ASD_fold_{i}.pth' for i in range(10)]
    print('모델 로딩 중...')
    models = load_models(model_paths)
    print(f'모델 {len(models)}개 로드 완료')

    # 전체 split 평가
    all_metrics = {}
    all_y_true, all_y_prob = [], []
    for split_name in ['training', 'validation', 'test']:
        metrics, y_true, y_prob = evaluate_split(split_name, models, args.threshold, output_dir)
        all_metrics[split_name] = metrics
        all_y_true.append(y_true)
        all_y_prob.append(y_prob)

    # overall 지표 계산
    overall_y_true = np.concatenate(all_y_true)
    overall_y_prob = np.concatenate(all_y_prob)
    all_metrics['overall'] = compute_metrics(overall_y_true, overall_y_prob, args.threshold)
    print_metrics('overall', all_metrics['overall'])

    print_summary(all_metrics)

    # 요약 CSV 저장
    summary_df = pd.DataFrame(all_metrics).T.reset_index().rename(columns={'index': 'split'})
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'\n요약 저장: {summary_path}')


if __name__ == '__main__':
    main()
