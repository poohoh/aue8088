#!/usr/bin/env python3

import sys
import torch
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.torch_utils import select_device
from utils.test_time_augmentation import EnhancedTTA

# 기본 설정
device = select_device('4')
weights = 'second/s_bat32_epo300_aughigh/weights/last.pt'
data_path = 'data/kaist-rgbt.yaml'
batch_size = 1  # 배치 크기를 1로 줄임
imgsz = 640
task = 'val'

print("=== TTA 디버깅 ===")

try:
    # 모델 로드
    model = attempt_load(weights, device=device)
    model = model.half()
    model.eval()
    print("Model loaded successfully")
    
    # 데이터로더 생성
    data = check_dataset(data_path)
    dataloader = create_dataloader(
        data[task], imgsz, batch_size, 32, False,
        pad=0.5, rect=False, workers=0,
        prefix="debug: ",
        rgbt_input=True
    )[0]
    print("Dataloader created successfully")
    
    # TTA 객체 생성 (빠른 모드)
    tta = EnhancedTTA(device=device, fast_mode=True)
    print(f"TTA config - scales: {tta.scales}, flips: {tta.flips}, rotations: {tta.rotations}")
    
    # 첫 번째 배치로 TTA 테스트
    print("Testing TTA transformations...")
    with torch.no_grad():
        for i, (imgs, targets, paths, shapes, indices) in enumerate(dataloader):
            print(f"Processing batch {i}...")
            
            # 입력 전처리
            imgs = [x.to(device).half() / 255.0 for x in imgs]
            img_single = [x[0:1] for x in imgs]  # 배치에서 첫 번째 이미지만
            print(f"Input shapes: {[x.shape for x in img_single]}")
            
            # TTA 변환 테스트
            test_count = 0
            for scale in tta.scales:
                for flip in tta.flips:
                    for rotation in tta.rotations:
                        for cj in tta.color_jitters:
                            test_count += 1
                            print(f"  Test {test_count}: scale={scale}, flip={flip}, rot={rotation}")
                            
                            try:
                                # 변환 적용
                                aug_img, transform_info = tta._apply_to_streams(
                                    [x.clone() for x in img_single],
                                    scale, flip, rotation, cj
                                )
                                print(f"    Transform successful: {[x.shape for x in aug_img]}")
                                
                                # 모델 추론
                                pred = model(aug_img)
                                print(f"    Inference successful: {pred[0].shape}")
                                
                            except Exception as e:
                                print(f"    ERROR: {e}")
                                import traceback
                                traceback.print_exc()
                                break
                            
                            if test_count >= 4:  # 처음 4개만 테스트
                                break
                        if test_count >= 4:
                            break
                    if test_count >= 4:
                        break
                if test_count >= 4:
                    break
            
            print(f"TTA test completed for batch {i}")
            break  # 첫 번째 배치만 테스트
            
    print("TTA debugging completed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
