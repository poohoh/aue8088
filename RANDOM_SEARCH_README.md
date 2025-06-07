# YOLOv5 Random Search for Hyperparameter Optimization

이 스크립트들은 YOLOv5 모델의 하이퍼파라미터 최적화를 위한 랜덤 서치를 수행합니다.

## 파일 구조

- `random_search.sh`: 메인 랜덤 서치 실행 스크립트
- `analyze_random_search.py`: 결과 분석 및 시각화 스크립트
- `RANDOM_SEARCH_README.md`: 사용법 설명서 (이 파일)

## 사용 방법

### 1. 랜덤 서치 실행

```bash
# 실행 권한 부여 (최초 1회만)
chmod +x random_search.sh

# 랜덤 서치 실행
./random_search.sh
```

### 2. 설정 변경

`random_search.sh` 파일에서 다음 설정들을 수정할 수 있습니다:

```bash
# 시행 횟수
NUM_TRIALS=20

# 기본 하이퍼파라미터 파일
BASE_HYP_FILE="data/hyps/hyp.scratch-med.yaml"

# GPU 설정
CUDA_VISIBLE_DEVICES=5

# 데이터셋 설정
--data datasets/kaist-rgbt/kfold_splits/yaml_configs/kaist-rgbt-fold1.yaml
--cfg models/yolov5s_kaist-rgbt.yaml
```

### 3. 하이퍼파라미터 범위

현재 최적화되는 하이퍼파라미터들과 그 범위:

| 파라미터 | 범위 | 설명 |
|----------|------|------|
| lr0 | 0.001 ~ 0.02 | 초기 학습률 |
| lrf | 0.01 ~ 0.2 | 최종 학습률 비율 |
| momentum | 0.8 ~ 0.957 | SGD 모멘텀 |
| weight_decay | 0.0001 ~ 0.001 | 가중치 감쇠 |
| box | 0.02 ~ 0.1 | 박스 손실 가중치 |
| cls | 0.1 ~ 0.9 | 클래스 손실 가중치 |
| obj | 0.3 ~ 1.5 | 객체 손실 가중치 |
| hsv_h | 0.005 ~ 0.03 | HSV 색조 증강 |
| hsv_s | 0.3 ~ 1.0 | HSV 채도 증강 |
| hsv_v | 0.2 ~ 0.8 | HSV 명도 증강 |
| translate | 0.05 ~ 0.2 | 이미지 평행이동 |
| scale | 0.5 ~ 1.4 | 이미지 스케일링 |
| fliplr | 0.0 ~ 0.8 | 좌우 반전 확률 |
| mosaic | 0.5 ~ 1.0 | 모자이크 증강 확률 |
| mixup | 0.0 ~ 0.3 | Mixup 증강 확률 |
| copy_paste | 0.0 ~ 0.2 | Copy-paste 증강 확률 |

### 4. 결과 분석

랜덤 서치 완료 후 결과를 분석하려면:

```bash
# 필요한 패키지 설치 (최초 1회만)
pip install pandas matplotlib seaborn

# 결과 분석 실행
python analyze_random_search.py --results runs/random_search_YYYYMMDD_HHMMSS/random_search_results.csv
```

## 출력 결과

### 1. 랜덤 서치 결과
- `runs/random_search_YYYYMMDD_HHMMSS/`: 타임스탬프가 포함된 결과 디렉토리
- `random_search_results.csv`: 모든 trial의 하이퍼파라미터와 성능 결과
- `hyp-trial-X.yaml`: 각 trial에서 사용된 하이퍼파라미터 파일
- `random_trial_X/`: 각 trial의 학습 결과 (가중치, 로그, 플롯 등)

### 2. 분석 결과
- `analysis_plots/`: 시각화 결과 디렉토리
  - `performance_distribution.png`: 성능 분포 히스토그램
  - `correlation_heatmap.png`: 하이퍼파라미터 상관관계 히트맵
  - `hyperparameter_scatter.png`: 주요 하이퍼파라미터 vs 성능 산점도
  - `trial_progression.png`: Trial별 성능 변화

### 3. 콘솔 출력
- 각 trial의 하이퍼파라미터 설정값
- 실시간 학습 진행 상황
- 최종 성능 통계 및 최고 성능 trial 정보

## 커스터마이징

### 1. 하이퍼파라미터 범위 수정

`random_search.sh`에서 원하는 하이퍼파라미터의 범위를 수정할 수 있습니다:

```bash
# 예: Learning rate 범위를 0.005 ~ 0.05로 변경
LR0=$(LC_NUMERIC="C" printf "%.6f" $(echo "scale=6; 0.005 + ($RANDOM % 45000) / 1000000" | bc))
```

### 2. 새로운 하이퍼파라미터 추가

1. 랜덤 값 생성 섹션에 새 파라미터 추가
2. sed 명령어로 파일 수정 추가
3. CSV 헤더와 기록 부분에 추가

### 3. 학습 설정 변경

```bash
# 예: batch size, epochs, image size 등 변경
--batch-size 64 \
--epochs 200 \
--imgsz 1024 \
```

## 주의사항

1. **GPU 메모리**: batch size와 image size 설정에 주의
2. **디스크 공간**: 각 trial마다 모델 가중치와 로그가 저장됨
3. **실행 시간**: trial 수와 epochs에 따라 실행 시간이 길어질 수 있음
4. **중단 후 재시작**: 스크립트 중단 시 마지막 trial부터 재시작하려면 스크립트 수정 필요

## 예제

```bash
# 10번의 짧은 trial로 빠른 테스트
# NUM_TRIALS=10, epochs=50으로 설정 후
./random_search.sh

# 결과 분석
python analyze_random_search.py --results runs/random_search_*/random_search_results.csv
```

이 스크립트를 통해 최적의 하이퍼파라미터 조합을 효율적으로 찾을 수 있습니다.
