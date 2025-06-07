#!/bin/bash

# Quick test to observe learning curves with different epochs
# This helps determine optimal epoch count for random search

BASE_HYP_FILE="data/hyps/hyp.scratch-med.yaml"
TEST_DIR="runs/epoch_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $TEST_DIR

echo "Testing different epoch counts to determine optimal length for random search..."

# Test with 3 different hyperparameter settings
for test_id in 1 2 3; do
    echo "=== Test $test_id/3 ==="
    
    # Create test hyperparameter file
    TEMP_HYP_FILE="data/hyps/hyp-test-$test_id.yaml"
    cp $BASE_HYP_FILE $TEMP_HYP_FILE
    
    # Slightly different settings for each test
    if [ $test_id -eq 1 ]; then
        # High LR, low augmentation
        sed -i "s/lr0: .*/lr0: 0.015/" $TEMP_HYP_FILE
        sed -i "s/mixup: .*/mixup: 0.0/" $TEMP_HYP_FILE
        sed -i "s/mosaic: .*/mosaic: 0.8/" $TEMP_HYP_FILE
    elif [ $test_id -eq 2 ]; then
        # Medium LR, medium augmentation  
        sed -i "s/lr0: .*/lr0: 0.01/" $TEMP_HYP_FILE
        sed -i "s/mixup: .*/mixup: 0.1/" $TEMP_HYP_FILE
        sed -i "s/mosaic: .*/mosaic: 1.0/" $TEMP_HYP_FILE
    else
        # Low LR, high augmentation
        sed -i "s/lr0: .*/lr0: 0.005/" $TEMP_HYP_FILE
        sed -i "s/mixup: .*/mixup: 0.2/" $TEMP_HYP_FILE
        sed -i "s/mosaic: .*/mosaic: 1.0/" $TEMP_HYP_FILE
        sed -i "s/hsv_s: .*/hsv_s: 0.9/" $TEMP_HYP_FILE
    fi
    
    # Run training for 100 epochs to see full curve
    CUDA_VISIBLE_DEVICES=4 python train.py \
        --data datasets/kaist-rgbt/kfold_splits/yaml_configs/kaist-rgbt-fold1.yaml \
        --cfg models/yolov5s_kaist-rgbt.yaml \
        --weights yolov5s.pt \
        --hyp $TEMP_HYP_FILE \
        --batch-size 32 \
        --epochs 100 \
        --imgsz 640 \
        --name "epoch_test_$test_id" \
        --rgbt \
        --cos-lr \
        --patience 25 \
        --project "$TEST_DIR"
    
    rm $TEMP_HYP_FILE
done

echo "========================================"
echo "Epoch test completed!"
echo "Check the results.csv files in $TEST_DIR"
echo "Look for when mAP@0.5:0.95 stabilizes"
echo "========================================"

# Analyze the results
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt
import os

test_dir = "$TEST_DIR"
if os.path.exists(test_dir):
    plt.figure(figsize=(15, 5))
    
    for test_id in [1, 2, 3]:
        results_path = f"{test_dir}/epoch_test_{test_id}/results.csv"
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            plt.subplot(1, 3, test_id)
            
            # Remove any non-numeric values
            df_clean = df[pd.to_numeric(df.iloc[:, 7], errors='coerce').notna()]
            
            if len(df_clean) > 0:
                epochs = df_clean.iloc[:, 0]  # epoch column
                map_50_95 = pd.to_numeric(df_clean.iloc[:, 7])  # mAP@0.5:0.95
                
                plt.plot(epochs, map_50_95, 'b-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('mAP@0.5:0.95')
                plt.title(f'Test {test_id} Learning Curve')
                plt.grid(True, alpha=0.3)
                
                # Find when 90% of final performance is reached
                final_map = map_50_95.iloc[-1]
                target_map = final_map * 0.9
                
                reaching_epochs = epochs[map_50_95 >= target_map]
                if len(reaching_epochs) > 0:
                    early_epoch = reaching_epochs.iloc[0]
                    plt.axvline(x=early_epoch, color='r', linestyle='--', 
                              label=f'90% at epoch {early_epoch}')
                    plt.legend()
                
                print(f"Test {test_id}: Final mAP@0.5:0.95 = {final_map:.4f}")
                if len(reaching_epochs) > 0:
                    print(f"  90% of final performance reached at epoch {early_epoch}")
    
    plt.tight_layout()
    plt.savefig(f'{test_dir}/learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nLearning curves saved to: {test_dir}/learning_curves.png")
    print("Based on these curves, choose appropriate epoch count for random search.")
EOF
