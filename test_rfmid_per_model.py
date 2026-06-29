"""
RFMiD 데이터셋을 이용한 ASD / Non-ASD 분류 테스트 - 모델별 개별 평가 스크립트

레이블 정의:
  ASD     : ODC==1 OR TV==1 OR ODE==1, 나머지 disease 컬럼은 전부 0
  Non-ASD : Disease_Risk == 0

평가 방식: 10개 fold 모델을 각각 단독으로 평가 (앙상블 아님)
데이터   : Training / Validation / Test set 전체 합산 (908건)
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
    roc_auc_score, confusion_matrix
)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from main import preprocess, load_model


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
ASD_COLS   = ['ODC', 'TV', 'ODE']
OTHER_COLS = [c for c in DISEASE_COLS if c not in ASD_COLS]

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


def load_all_data() -> pd.DataFrame:
    """3개 split을 합산하여 단일 DataFrame으로 반환한다."""
    frames = []
    for split_name, (csv_path, image_dir) in SPLITS.items():
        df = parse_labels(csv_path, image_dir)
        df['split'] = split_name
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# 전처리 캐싱 (모든 이미지를 1회만 전처리)
# ---------------------------------------------------------------------------
def preprocess_all(df: pd.DataFrame) -> list:
    """
    전체 이미지를 전처리하여 텐서 리스트로 반환한다.
    모델과 무관하게 결과가 동일하므로 1회만 수행한다.
    """
    tensors = []
    errors  = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='전처리'):
        try:
            tensor = preprocess(row['image_path'])
            tensors.append(tensor)
        except Exception as e:
            print(f'\n  [오류] ID={row["ID"]}: {e}')
            tensors.append(None)
            errors.append(row['ID'])

    if errors:
        print(f'  [경고] 전처리 실패 {len(errors)}건: {errors}')
    return tensors


# ---------------------------------------------------------------------------
# 단일 모델 추론
# ---------------------------------------------------------------------------
def infer_with_model(model, tensors: list) -> list:
    """
    단일 모델로 전체 텐서를 추론하여 ASD 확률 리스트를 반환한다.
    None 텐서(전처리 실패)는 np.nan으로 채운다.
    """
    probs = []
    model.eval()
    with torch.no_grad():
        for tensor in tensors:
            if tensor is None:
                probs.append(np.nan)
                continue
            img_tensor = tensor.unsqueeze(0).to('cuda')
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()
            probs.append(prob)
    return probs


# ---------------------------------------------------------------------------
# 평가 지표 계산
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision_score(y_true, y_pred, zero_division=0),
        'f1':          f1_score(y_true, y_pred, zero_division=0),
        'auc_roc':     roc_auc_score(y_true, y_prob),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------
def print_per_model_summary(all_metrics: list, total: int, asd_n: int, non_asd_n: int):
    print(f'\n{"="*68}')
    print(f'  모델별 평가 결과  (전체 {total}건: ASD={asd_n}, Non-ASD={non_asd_n})')
    print(f'{"="*68}')
    header = f'  {"Model":<14} {"Acc":>7} {"Sens":>7} {"Spec":>7} {"F1":>7} {"AUC":>7}'
    print(header)
    print(f'  {"-"*62}')

    for entry in all_metrics:
        m = entry['metrics']
        print(f'  {entry["model_name"]:<14} {m["accuracy"]:>7.4f} {m["sensitivity"]:>7.4f} '
              f'{m["specificity"]:>7.4f} {m["f1"]:>7.4f} {m["auc_roc"]:>7.4f}')

    # mean ± std
    keys = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc_roc']
    values = {k: [e['metrics'][k] for e in all_metrics] for k in keys}
    print(f'  {"-"*62}')
    means  = {k: np.mean(values[k]) for k in keys}
    stds   = {k: np.std(values[k])  for k in keys}
    mean_str = f'  {"mean":>14}'
    std_str  = f'  {"std":>14}'
    for k in keys:
        mean_str += f' {means[k]:>7.4f}'
        std_str  += f' {stds[k]:>7.4f}'
    print(mean_str)
    print(std_str)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='RFMiD ASD 분류 모델별 개별 테스트')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='ASD 판정 확률 임계값 (default: 0.5)')
    parser.add_argument('--output_dir', type=str, default='test_results_per_model',
                        help='결과 저장 디렉토리 (default: test_results_per_model)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. 전체 데이터 로딩 (3개 split 합산)
    print('데이터 로딩 중...')
    df = load_all_data()

    if BALANCE_DATASET:
        df = balance_dataset(df)
        print(f'  [균형 조정] Non-ASD를 ASD 수에 맞게 다운샘플링')

    asd_n     = (df['label'] == 1).sum()
    non_asd_n = (df['label'] == 0).sum()
    print(f'전체 {len(df)}건 로드 완료  (ASD={asd_n}, Non-ASD={non_asd_n})')

    # 2. 전처리 1회 수행 (모델과 무관)
    tensors = preprocess_all(df)

    # 전처리 실패 샘플 마스킹
    valid_mask = [t is not None for t in tensors]
    valid_idx  = [i for i, v in enumerate(valid_mask) if v]
    if sum(valid_mask) < len(df):
        print(f'  [경고] 전처리 실패 {len(df)-sum(valid_mask)}건 제외')

    y_true_all = df['label'].values

    # 3. 모델별 단독 추론 및 평가
    model_paths = [f'models/Normal_ASD_fold_{i}.pth' for i in range(10)]
    all_metrics = []
    result_df   = df[['ID', 'label', 'split']].copy()

    for i, model_path in enumerate(model_paths):
        model_name = f'fold_{i}'
        print(f'\n[{model_name}] 모델 로딩 및 추론 중...')

        model = load_model(model_path).to('cuda').eval()
        probs = infer_with_model(model, tensors)
        del model
        torch.cuda.empty_cache()

        result_df[f'prob_{model_name}'] = probs
        result_df[f'pred_{model_name}'] = (np.array(probs) >= args.threshold).astype(int)

        # 유효한 샘플만 평가
        y_prob_arr = np.array(probs)
        valid      = ~np.isnan(y_prob_arr)
        metrics    = compute_metrics(y_true_all[valid], y_prob_arr[valid], args.threshold)

        m = metrics
        print(f'  Accuracy={m["accuracy"]:.4f}  Sensitivity={m["sensitivity"]:.4f}  '
              f'Specificity={m["specificity"]:.4f}  F1={m["f1"]:.4f}  AUC={m["auc_roc"]:.4f}')
        print(f'  TP={m["tp"]}  FP={m["fp"]}  FN={m["fn"]}  TN={m["tn"]}')

        all_metrics.append({'model_name': model_name, 'metrics': metrics})

    # 4. 결과 출력
    print_per_model_summary(all_metrics, len(df), int(asd_n), int(non_asd_n))

    # 5. 결과 저장
    result_path = output_dir / 'result_per_model.csv'
    result_df.to_csv(result_path, index=False)
    print(f'\n상세 결과 저장: {result_path}')

    summary_records = [
        {'model': e['model_name'], **e['metrics']} for e in all_metrics
    ]
    summary_df   = pd.DataFrame(summary_records)
    summary_path = output_dir / 'summary_per_model.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'요약 저장: {summary_path}')


if __name__ == '__main__':
    main()
