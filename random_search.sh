#!/bin/bash

# Random Search Script for YOLOv5 Hyperparameter Optimization
# Based on hyperparameter files in data/hyps/

# 1. 총 몇 번의 랜덤 서치를 수행할지 결정
NUM_TRIALS=20

# 2. 기본 하이퍼파라미터 파일 지정
BASE_HYP_FILE="data/hyps/hyp.scratch-low.yaml"

# 3. 결과 저장을 위한 디렉토리 생성
RESULTS_DIR="runs/random_search_low"
mkdir -p $RESULTS_DIR

# 4. 랜덤 서치 결과를 기록할 CSV 파일 생성
RESULTS_CSV="$RESULTS_DIR/random_search_results.csv"
echo "trial_id,lr0,lrf,momentum,weight_decay,box,cls,obj,hsv_h,hsv_s,hsv_v,translate,scale,fliplr,mosaic,mixup,copy_paste,mAP50,mAP50-95" > $RESULTS_CSV

echo "Starting Random Search with $NUM_TRIALS trials"
echo "Results will be saved in: $RESULTS_DIR"

for i in $(seq 1 $NUM_TRIALS)
do
    echo "===========================================" 
    echo "--- Starting Trial $i of $NUM_TRIALS ---"
    echo "==========================================="

    # 5. 임시 하이퍼파라미터 파일 생성
    # 임시 하이퍼파라미터 파일을 위한 폴더 생성
    TEMP_HYP_DIR="data/hyps/random_trials"
    mkdir -p $TEMP_HYP_DIR
    
    # 시작 시간을 기반으로 한 고유 식별자 생성
    TRIAL_TIMESTAMP=$(date +%Y%m%d_%H%M%S_$(shuf -i 1000-9999 -n 1))
    TEMP_HYP_FILE="$TEMP_HYP_DIR/hyp-random-trial-$TRIAL_TIMESTAMP.yaml"
    cp $BASE_HYP_FILE $TEMP_HYP_FILE

    # 6. 랜덤 값 생성 (다양한 하이퍼파라미터들)
    # 더 나은 랜덤 시드 설정 (프로세스 ID와 현재 시간 기반)
    RANDOM_SEED=$(($(date +%s%N | cut -b1-10) + $$ + $i))
    RANDOM=$RANDOM_SEED
    # Learning rate parameters
    LR0=$(python3 -c "import random; random.seed($RANDOM_SEED + 1); print(f'{random.uniform(0.005, 0.02):.6f}')")
    LRF=$(python3 -c "import random; random.seed($RANDOM_SEED + 2); print(f'{random.uniform(0.005, 0.05):.3f}')")
    
    # Optimizer parameters
    MOMENTUM=$(python3 -c "import random; random.seed($RANDOM_SEED + 3); print(f'{random.uniform(0.90, 0.97):.3f}')")
    WEIGHT_DECAY=$(python3 -c "import random; random.seed($RANDOM_SEED + 4); print(f'{random.uniform(0.0001, 0.001):.6f}')")
    
    # Loss weights (adjusted for low-augmentation baseline)
    BOX=$(python3 -c "import random; random.seed($RANDOM_SEED + 5); print(f'{random.uniform(0.02, 0.1):.3f}')")
    CLS=$(python3 -c "import random; random.seed($RANDOM_SEED + 6); print(f'{random.uniform(0.35, 0.6):.2f}')")  # low baseline: 0.5
    OBJ=$(python3 -c "import random; random.seed($RANDOM_SEED + 7); print(f'{random.uniform(0.8, 1.2):.2f}')")  # low baseline: 1.0
    
    # Color augmentation
    HSV_H=$(python3 -c "import random; random.seed($RANDOM_SEED + 8); print(f'{random.uniform(0.005, 0.02):.3f}')")
    HSV_S=$(python3 -c "import random; random.seed($RANDOM_SEED + 9); print(f'{random.uniform(0.3, 0.8):.2f}')")
    HSV_V=$(python3 -c "import random; random.seed($RANDOM_SEED + 10); print(f'{random.uniform(0.2, 0.6):.2f}')")
    
    # Geometric augmentation (adjusted for low-augmentation baseline)
    TRANSLATE=$(python3 -c "import random; random.seed($RANDOM_SEED + 11); print(f'{random.uniform(0.05, 0.15):.2f}')")  # low baseline: 0.1
    SCALE=$(python3 -c "import random; random.seed($RANDOM_SEED + 12); print(f'{random.uniform(0.3, 0.8):.2f}')")       # low baseline: 0.5
    FLIPLR=$(python3 -c "import random; random.seed($RANDOM_SEED + 13); print(f'{random.uniform(0.2, 0.8):.2f}')")      # low baseline: 0.5
    
    # Advanced augmentation (adjusted for low-augmentation baseline)
    MOSAIC=$(python3 -c "import random; random.seed($RANDOM_SEED + 14); print(f'{random.uniform(0.5, 1.0):.2f}')")      # low baseline: 1.0
    MIXUP=$(python3 -c "import random; random.seed($RANDOM_SEED + 15); print(f'{random.uniform(0.0, 0.2):.2f}')")       # low baseline: 0.0
    COPY_PASTE=$(python3 -c "import random; random.seed($RANDOM_SEED + 16); print(f'{random.uniform(0.0, 0.2):.2f}')")  # low baseline: 0.1

    # 7. sed 명령어로 임시 파일의 값을 변경
    sed -i "s/lr0: .*/lr0: $LR0/" $TEMP_HYP_FILE
    sed -i "s/lrf: .*/lrf: $LRF/" $TEMP_HYP_FILE
    sed -i "s/momentum: .*/momentum: $MOMENTUM/" $TEMP_HYP_FILE
    sed -i "s/weight_decay: .*/weight_decay: $WEIGHT_DECAY/" $TEMP_HYP_FILE
    sed -i "s/box: .*/box: $BOX/" $TEMP_HYP_FILE
    sed -i "s/cls: .*/cls: $CLS/" $TEMP_HYP_FILE
    sed -i "s/obj: .*/obj: $OBJ/" $TEMP_HYP_FILE
    sed -i "s/hsv_h: .*/hsv_h: $HSV_H/" $TEMP_HYP_FILE
    sed -i "s/hsv_s: .*/hsv_s: $HSV_S/" $TEMP_HYP_FILE
    sed -i "s/hsv_v: .*/hsv_v: $HSV_V/" $TEMP_HYP_FILE
    sed -i "s/translate: .*/translate: $TRANSLATE/" $TEMP_HYP_FILE
    sed -i "s/scale: .*/scale: $SCALE/" $TEMP_HYP_FILE
    sed -i "s/fliplr: .*/fliplr: $FLIPLR/" $TEMP_HYP_FILE
    sed -i "s/mosaic: .*/mosaic: $MOSAIC/" $TEMP_HYP_FILE
    sed -i "s/mixup: .*/mixup: $MIXUP/" $TEMP_HYP_FILE
    sed -i "s/copy_paste: .*/copy_paste: $COPY_PASTE/" $TEMP_HYP_FILE

    echo "Hyperparameters for Trial $i:"
    echo "  lr0=$LR0, lrf=$LRF, momentum=$MOMENTUM, weight_decay=$WEIGHT_DECAY"
    echo "  box=$BOX, cls=$CLS, obj=$OBJ"
    echo "  hsv_h=$HSV_H, hsv_s=$HSV_S, hsv_v=$HSV_V"
    echo "  translate=$TRANSLATE, scale=$SCALE, fliplr=$FLIPLR"
    echo "  mosaic=$MOSAIC, mixup=$MIXUP, copy_paste=$COPY_PASTE"

    # 8. 생성된 하이퍼파라미터 파일을 결과 디렉토리에 백업
    # 결과 디렉토리에도 하이퍼파라미터 파일들을 위한 폴더 생성
    BACKUP_HYP_DIR="$RESULTS_DIR/hyperparameters"
    mkdir -p $BACKUP_HYP_DIR
    cp $TEMP_HYP_FILE "$BACKUP_HYP_DIR/hyp-trial-$TRIAL_TIMESTAMP.yaml"

    # 9. YOLOv5 학습 실행
    TRIAL_NAME="random_trial_$TRIAL_TIMESTAMP"
    echo "Starting training for trial $i (ID: $TRIAL_TIMESTAMP)..."
    
    CUDA_VISIBLE_DEVICES=5 python train.py \
        --data datasets/kaist-rgbt/kfold_splits/yaml_configs/kaist-rgbt-fold1.yaml \
        --cfg models/yolov5s_kaist-rgbt.yaml \
        --weights yolov5s.pt \
        --hyp $TEMP_HYP_FILE \
        --batch-size 16 \
        --epochs 40 \
        --imgsz 640 \
        --name "$TRIAL_NAME" \
        --rgbt \
        --cos-lr \
        --patience 15 \
        --project "$RESULTS_DIR"

    # 10. 결과 추출 및 CSV에 기록
    RESULT_PATH="$RESULTS_DIR/$TRIAL_NAME"
    if [ -f "$RESULT_PATH/results.csv" ]; then
        # results.csv의 마지막 줄에서 mAP 값들 추출 (일반적으로 마지막 epoch의 결과)
        LAST_LINE=$(tail -n 1 "$RESULT_PATH/results.csv")
        MAP50=$(echo $LAST_LINE | cut -d',' -f7)        # mAP@0.5
        MAP50_95=$(echo $LAST_LINE | cut -d',' -f8)     # mAP@0.5:0.95
        
        # CSV에 결과 기록 (trial ID로 timestamp 사용)
        echo "$TRIAL_TIMESTAMP,$LR0,$LRF,$MOMENTUM,$WEIGHT_DECAY,$BOX,$CLS,$OBJ,$HSV_H,$HSV_S,$HSV_V,$TRANSLATE,$SCALE,$FLIPLR,$MOSAIC,$MIXUP,$COPY_PASTE,$MAP50,$MAP50_95" >> $RESULTS_CSV
        
        echo "Trial $i (ID: $TRIAL_TIMESTAMP) completed - mAP@0.5: $MAP50, mAP@0.5:0.95: $MAP50_95"
    else
        echo "Warning: Results file not found for trial $i (ID: $TRIAL_TIMESTAMP)"
        echo "$TRIAL_TIMESTAMP,$LR0,$LRF,$MOMENTUM,$WEIGHT_DECAY,$BOX,$CLS,$OBJ,$HSV_H,$HSV_S,$HSV_V,$TRANSLATE,$SCALE,$FLIPLR,$MOSAIC,$MIXUP,$COPY_PASTE,N/A,N/A" >> $RESULTS_CSV
    fi

    # 11. 임시 하이퍼파라미터 파일 정리
    rm $TEMP_HYP_FILE

    echo "--- Finished Trial $i (ID: $TRIAL_TIMESTAMP) ---"
    echo ""
done

echo "=========================================="
echo "Random Search Completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Summary CSV: $RESULTS_CSV"
echo "=========================================="

# 12. 최고 성능 trial 찾기
echo "Finding best performing trial..."
python3 << EOF
import pandas as pd
import os

results_file = '$RESULTS_CSV'
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    # mAP@0.5:0.95 기준으로 최고 성능 찾기
    df_valid = df[df['mAP50-95'] != 'N/A']
    if not df_valid.empty:
        df_valid['mAP50-95'] = pd.to_numeric(df_valid['mAP50-95'])
        best_trial = df_valid.loc[df_valid['mAP50-95'].idxmax()]
        print(f"Best Trial ID: {best_trial['trial_id']}")
        print(f"Best mAP@0.5:0.95: {best_trial['mAP50-95']:.4f}")
        print(f"Best mAP@0.5: {best_trial['mAP50']:.4f}")
        print("Best Hyperparameters:")
        print(f"  lr0: {best_trial['lr0']}")
        print(f"  lrf: {best_trial['lrf']}")
        print(f"  momentum: {best_trial['momentum']}")
        print(f"  weight_decay: {best_trial['weight_decay']}")
        print(f"  mixup: {best_trial['mixup']}")
        print(f"  mosaic: {best_trial['mosaic']}")
    else:
        print("No valid results found.")
else:
    print("Results file not found.")
EOF
