#!/usr/bin/env python3

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from utils.torch_utils import select_device

# 기본 설정
device = select_device('4')
data_path = 'data/kaist-rgbt.yaml'
batch_size = 2
imgsz = 640
task = 'val'

print("=== 데이터로더 디버깅 ===")

# 데이터셋 체크
data = check_dataset(data_path)
print(f"Data: {data}")

# 데이터로더 생성 (RGBT 모드)
print("Creating RGBT dataloader...")
try:
    dataloader = create_dataloader(
        data[task], imgsz, batch_size, 32, False,
        pad=0.5, rect=False, workers=0,  # workers=0 for debugging
        prefix="debug: ",
        rgbt_input=True
    )[0]
    print(f"Dataloader created successfully. Length: {len(dataloader)}")
    
    # 첫 번째 배치 테스트
    print("Testing first batch...")
    for i, (imgs, targets, paths, shapes, indices) in enumerate(dataloader):
        print(f"Batch {i}:")
        if isinstance(imgs, list):
            print(f"  - Multi-stream input: {len(imgs)} streams")
            for j, img in enumerate(imgs):
                print(f"    Stream {j}: {img.shape} {img.dtype}")
        else:
            print(f"  - Single tensor: {imgs.shape} {imgs.dtype}")
        print(f"  - Targets: {targets.shape if targets is not None else 'None'}")
        print(f"  - Paths: {len(paths)}")
        
        if i >= 2:  # Test only first 3 batches
            break
            
except Exception as e:
    print(f"Error creating dataloader: {e}")
    import traceback
    traceback.print_exc()
