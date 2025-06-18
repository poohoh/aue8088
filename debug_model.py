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

# 기본 설정
device = select_device('4')
weights = 'second/s_bat32_epo300_aughigh/weights/last.pt'
data_path = 'data/kaist-rgbt.yaml'
batch_size = 2
imgsz = 640
task = 'val'

print("=== 모델 및 추론 디버깅 ===")

# 모델 로드
print("Loading RGBT model...")
try:
    model = attempt_load(weights, device=device)
    print(f"Model loaded: {type(model)}")
    print(f"Model stride: {model.stride}")
    
    # Half precision
    model = model.half()
    print("Model converted to half precision")
    
    # 데이터로더 생성
    data = check_dataset(data_path)
    dataloader = create_dataloader(
        data[task], imgsz, batch_size, 32, False,
        pad=0.5, rect=False, workers=0,
        prefix="debug: ",
        rgbt_input=True
    )[0]
    
    # 첫 번째 배치로 추론 테스트
    print("Testing model inference...")
    model.eval()
    with torch.no_grad():
        for i, (imgs, targets, paths, shapes, indices) in enumerate(dataloader):
            print(f"Processing batch {i}...")
            
            # 입력 전처리
            if isinstance(imgs, list):
                imgs = [x.to(device).half() / 255.0 for x in imgs]
                print(f"Input streams: {[x.shape for x in imgs]}")
            else:
                imgs = imgs.to(device).half() / 255.0
                print(f"Input tensor: {imgs.shape}")
            
            # 모델 추론
            try:
                pred = model(imgs)
                print(f"Inference successful! Output shape: {pred[0].shape if isinstance(pred, tuple) else pred.shape}")
            except Exception as e:
                print(f"Inference failed: {e}")
                import traceback
                traceback.print_exc()
                break
            
            if i >= 2:  # Test only first 3 batches
                break
                
    print("Model inference test completed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
