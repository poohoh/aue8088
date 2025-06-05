#!/usr/bin/env python3
"""
5-Fold Cross Validation 데이터 분할 스크립트
KAIST-RGBT 데이터셋을 5-fold로 나누어 train/val 파일들을 생성합니다.
RGB(visible)와 LWIR(thermal) 이미지 쌍을 고려하여 분할합니다.
"""

import os
import random
from pathlib import Path
import argparse
from tqdm import tqdm


def create_kfold_splits(data_file, output_dir, k=5, seed=42):
    """
    K-fold cross validation을 위한 데이터 분할
    KAIST 데이터셋의 RGB(visible)+LWIR(thermal) 이미지 쌍을 고려하여 분할
    
    Args:
        data_file: 원본 데이터 파일 경로 (train-all-04.txt)
        output_dir: 출력 디렉토리
        k: fold 수 (default: 5)
        seed: 랜덤 시드
    """
    
    # 랜덤 시드 설정
    random.seed(seed)
    
    # 데이터 파일 읽기
    with open(data_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"전체 데이터 샘플 수: {len(lines)}")
    
    # KAIST 데이터셋은 각 라인에 {} 플레이스홀더가 있음 (visible/lwir로 대체됨)
    # 각 라인은 하나의 이미지 쌍(visible + lwir)을 나타냄
    
    # 데이터 섞기
    random.shuffle(lines)
    
    # 각 fold의 크기 계산
    fold_size = len(lines) // k
    remainder = len(lines) % k
    
    print(f"{k}-fold 분할:")
    print(f"기본 fold 크기: {fold_size}")
    print(f"나머지: {remainder}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 각 fold에 대해 train/val 분할 생성
    for fold in tqdm(range(k), desc="Creating folds"):
        print(f"\nFold {fold + 1} 생성 중...")
        
        # validation 인덱스 계산
        start_idx = fold * fold_size
        # 나머지를 앞쪽 fold들에 분배
        if fold < remainder:
            start_idx += fold
            val_size = fold_size + 1
        else:
            start_idx += remainder
            val_size = fold_size
        
        end_idx = start_idx + val_size
        
        # validation과 training 데이터 분리
        val_data = lines[start_idx:end_idx]
        train_data = lines[:start_idx] + lines[end_idx:]
        
        print(f"  Training 샘플: {len(train_data)}")
        print(f"  Validation 샘플: {len(val_data)}")
        
        # train/val 파일 생성
        create_fold_files(output_path, fold + 1, train_data, val_data)
        
        print(f"  Fold {fold + 1} 파일들이 생성되었습니다.")
    
    # 분할 검증
    verify_splits(output_path, k, len(lines))
    
    # 전체 통계 출력
    print(f"\n=== 분할 완료 ===")
    print(f"총 {k}개의 fold 생성")
    print(f"각 fold별 파일들이 {output_path}에 저장되었습니다.")
    
    # 사용 예시를 위한 YAML 파일 생성
    create_fold_yamls(output_path, k)


def create_fold_files(output_path, fold_num, train_data, val_data):
    """
    각 fold의 train/val 파일들을 생성
    
    Args:
        output_path: 출력 디렉토리 Path 객체
        fold_num: fold 번호
        train_data: 훈련 데이터 리스트 ({}플레이스홀더 포함)
        val_data: 검증 데이터 리스트 ({}플레이스홀더 포함)
    """
    
    # 파일 생성
    train_file = output_path / f"train_fold{fold_num}.txt"
    val_file = output_path / f"val_fold{fold_num}.txt"
    
    # Train 파일 생성
    with open(train_file, 'w') as f:
        for line in train_data:
            f.write(line + '\n')
    
    # Val 파일 생성
    with open(val_file, 'w') as f:
        for line in val_data:
            f.write(line + '\n')
    
    print(f"    Generated: {train_file.name}, {val_file.name}")


def create_test_file(output_path):
    """
    테스트 데이터 파일을 출력 디렉토리에 복사
    """
    
    test_file = Path("datasets/kaist-rgbt/test-all-20.txt")
    
    if not test_file.exists():
        print(f"Warning: 테스트 파일을 찾을 수 없습니다: {test_file}")
        return
    
    # 테스트 데이터 읽기
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Test 파일 생성 ({}플레이스홀더 그대로 유지)
    test_output_file = output_path / "test.txt"
    with open(test_output_file, 'w') as f:
        for line in test_lines:
            f.write(line + '\n')
    
    print(f"  테스트 파일 생성: {test_output_file.name}")


def create_fold_yamls(output_dir, k=5):
    """각 fold별 YAML 설정 파일 생성 - 원본 KAIST 데이터셋 구조에 맞게"""
    
    # 경로 계산을 위한 절대 경로 처리
    output_path = Path(output_dir).resolve()
    yaml_dir = output_path / "yaml_configs"
    yaml_dir.mkdir(exist_ok=True)
    
    # 프로젝트 루트 경로 계산 (현재 작업 디렉토리)
    project_root = Path.cwd()
    
    # YAML에서 datasets/kaist-rgbt까지의 상대 경로 계산
    try:
        # datasets/kaist-rgbt의 절대 경로
        kaist_data_path = project_root / "datasets/kaist-rgbt"
        # YAML 파일 위치에서 datasets/kaist-rgbt로의 상대 경로
        rel_path_to_kaist = os.path.relpath(kaist_data_path, yaml_dir)
        # YAML 파일 위치에서 분할된 파일들로의 상대 경로
        rel_path_to_data = os.path.relpath(output_path, yaml_dir)
    except ValueError:
        # 다른 드라이브에 있는 경우 절대 경로 사용
        rel_path_to_kaist = str(project_root / "datasets/kaist-rgbt")
        rel_path_to_data = str(output_path)
    
    # 테스트 파일 복사 ({}플레이스홀더 그대로 유지)
    create_test_file(output_path)
    
    template = """# KAIST-RGBT {fold_num}-fold Cross Validation
# Auto-generated YAML file for fold {fold_num}
# Generated on: {timestamp}

# Path to dataset root (relative to this YAML file)
path: {kaist_path}

# Training data ({{}} placeholder will be replaced with visible/lwir by YOLOv5)
train:
  - {data_path}/train_fold{fold_num}.txt

# Validation data ({{}} placeholder will be replaced with visible/lwir by YOLOv5)
val:
  - {data_path}/val_fold{fold_num}.txt

# Test data ({{}} placeholder will be replaced with visible/lwir by YOLOv5)
test:
  - {data_path}/test.txt

# Classes for KAIST multispectral pedestrian detection
names:
  0: person
  1: cyclist
  2: people
  3: person?

# Number of classes
nc: 4
"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for fold in range(k):
        fold_num = fold + 1
        
        yaml_content = template.format(
            fold_num=fold_num,
            timestamp=timestamp,
            kaist_path=rel_path_to_kaist,
            data_path=rel_path_to_data
        )
        
        yaml_file = yaml_dir / f"kaist-rgbt-fold{fold_num}.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"YAML 설정 파일 생성: {yaml_file}")
    
    # 사용 가이드 출력
    print(f"\n=== 사용 가이드 ===")
    print(f"생성된 YAML 파일들을 사용하여 학습하려면:")
    for fold in range(1, k + 1):
        yaml_path = yaml_dir / f"kaist-rgbt-fold{fold}.yaml"
        try:
            relative_yaml_path = yaml_path.relative_to(project_root)
            print(f"  Fold {fold}: python train.py --data {relative_yaml_path}")
        except ValueError:
            print(f"  Fold {fold}: python train.py --data {yaml_path}")
    
    print(f"\n또는 데이터 디렉토리로 복사하여 사용:")
    print(f"  cp {yaml_dir}/*.yaml data/")
    print(f"  python train.py --data data/kaist-rgbt-fold1.yaml")


def verify_splits(output_path, k, total_samples):
    """
    분할 결과를 검증하여 데이터 중복이나 누락이 없는지 확인
    
    Args:
        output_path: 출력 디렉토리 Path 객체
        k: fold 수
        total_samples: 전체 샘플 수
    """
    
    print(f"\n=== 분할 검증 ===")
    
    all_val_samples = set()
    total_train_samples = 0
    total_val_samples = 0
    
    for fold in range(1, k + 1):
        # fold 파일들로 검증
        train_file = output_path / f"train_fold{fold}.txt"
        val_file = output_path / f"val_fold{fold}.txt"
        
        with open(train_file, 'r') as f:
            train_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        with open(val_file, 'r') as f:
            val_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 중복 검사를 위해 validation 샘플들을 저장
        for line in val_lines:
            if line in all_val_samples:
                print(f"WARNING: 중복된 validation 샘플 발견: {line}")
            all_val_samples.add(line)
        
        total_train_samples += len(train_lines)
        total_val_samples += len(val_lines)
        
        print(f"  Fold {fold}: Train={len(train_lines)}, Val={len(val_lines)}")
    
    # K-fold에서 예상되는 총 훈련 샘플 수 계산
    expected_train_samples = total_samples * (k - 1)
    avg_train_per_fold = total_train_samples // k
    
    print(f"\n총합:")
    print(f"  Training 샘플: {total_train_samples} (각 샘플이 {k-1}개 fold에서 훈련됨)")
    print(f"  Training 평균/fold: {avg_train_per_fold}")
    print(f"  Validation 샘플: {total_val_samples}")
    print(f"  원본 샘플 수: {total_samples}")
    
    # 검증 로직
    if total_val_samples == total_samples:
        print("✅ 검증 통과: 모든 샘플이 정확히 한 번씩 validation에 포함됨")
    else:
        print(f"❌ 검증 실패: validation 샘플 수 불일치 ({total_val_samples} != {total_samples})")
    
    if len(all_val_samples) == total_samples:
        print("✅ 검증 통과: validation 샘플 중복 없음")
    else:
        print(f"❌ 검증 실패: validation 샘플 중복 또는 누락 ({len(all_val_samples)} != {total_samples})")
    
    if total_train_samples == expected_train_samples:
        print("✅ 검증 통과: 총 훈련 샘플 수가 예상값과 일치함")
    else:
        print(f"❌ 검증 실패: 총 훈련 샘플 수 불일치 ({total_train_samples} != {expected_train_samples})")


def main():
    parser = argparse.ArgumentParser(description='KAIST-RGBT 데이터셋 5-fold 분할')
    parser.add_argument('--data_file', type=str, 
                       default='datasets/kaist-rgbt/train-all-04.txt',
                       help='원본 데이터 파일 경로')
    parser.add_argument('--output_dir', type=str,
                       default='datasets/kaist-rgbt/kfold_splits',
                       help='출력 디렉토리')
    parser.add_argument('--k', type=int, default=5,
                       help='Fold 수 (기본값: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
    args = parser.parse_args()
    
    # 데이터 파일 존재 확인
    if not os.path.exists(args.data_file):
        print(f"오류: 데이터 파일을 찾을 수 없습니다: {args.data_file}")
        return
    
    create_kfold_splits(args.data_file, args.output_dir, args.k, args.seed)


if __name__ == "__main__":
    main()
