#!/usr/bin/env python3
"""
RGB-T 모델 테스트 (4채널 입력)
"""

import torch
import numpy as np
from pathlib import Path
import sys

# YOLOv5 root directory 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def test_rgbt_model(weight_path):
    """RGB-T 모델 테스트 (4채널 입력)"""
    print(f"테스트할 weight 파일: {weight_path}")
    
    try:
        # 1. PyTorch로 직접 모델 로딩
        print("\n1. RGB-T 모델 로딩 중...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model = checkpoint['model'].float().eval().to(device)
        
        print(f"✓ 모델 로딩 성공")
        print(f"✓ 클래스: {model.names}")
        print(f"✓ 클래스 수: {model.nc}")
        
        # 2. RGB-T 더미 데이터 생성 (4채널)
        print(f"\n2. RGB-T 더미 데이터 생성...")
        batch_size = 1
        channels = 4  # RGB(3) + Thermal(1)
        height, width = 640, 640
        
        # RGB-T 이미지 데이터 생성
        rgbt_img = torch.randn(batch_size, channels, height, width).to(device)
        print(f"✓ RGB-T 이미지 생성: {rgbt_img.shape}")
        print(f"  - RGB 채널: {rgbt_img[:, :3, :, :].shape}")
        print(f"  - Thermal 채널: {rgbt_img[:, 3:, :, :].shape}")
        
        # 3. 추론 실행
        print(f"\n3. RGB-T 추론 테스트...")
        model.eval()
        with torch.no_grad():
            pred = model(rgbt_img)
        
        print(f"✓ RGB-T 추론 성공!")
        
        if isinstance(pred, (list, tuple)):
            print(f"✓ 출력 개수: {len(pred)}")
            for i, p in enumerate(pred):
                if isinstance(p, torch.Tensor):
                    print(f"  - 출력 {i}: {p.shape}")
                    if i == 0:  # 첫 번째 출력의 상세 정보
                        print(f"    범위: [{p.min().item():.4f}, {p.max().item():.4f}]")
        elif isinstance(pred, torch.Tensor):
            print(f"✓ 출력 크기: {pred.shape}")
            print(f"✓ 출력 범위: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        # 4. 분리된 RGB, Thermal 테스트
        print(f"\n4. 분리된 입력 테스트...")
        try:
            rgb_only = torch.randn(batch_size, 3, height, width).to(device)
            thermal_only = torch.randn(batch_size, 1, height, width).to(device)
            
            print(f"RGB 단독 입력 테스트...")
            with torch.no_grad():
                pred_rgb = model(rgb_only)
            print(f"✗ RGB 단독으로는 작동하지 않아야 함 (하지만 작동함)")
            
        except Exception as e:
            print(f"✓ RGB 단독 입력 실패 (예상됨): {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_attempt_load(weight_path):
    """attempt_load로 RGB-T 테스트"""
    print(f"\n5. attempt_load로 RGB-T 테스트...")
    
    try:
        from models.experimental import attempt_load
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model = attempt_load(weight_path, device=device, inplace=True, fuse=True)
        print(f"✓ attempt_load 성공")
        
        # RGB-T 추론 테스트
        rgbt_img = torch.randn(1, 4, 640, 640).to(device)  # 4채널
        with torch.no_grad():
            pred = model(rgbt_img)
        
        print(f"✓ attempt_load로 RGB-T 추론 성공!")
        if isinstance(pred, (list, tuple)):
            for i, p in enumerate(pred):
                if isinstance(p, torch.Tensor):
                    print(f"  - 출력 {i}: {p.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ attempt_load 오류: {e}")
        return False

def main():
    weight_file = "second/s_bat32_epo300_low_mosaic2/weights/best.pt"
    
    print("=" * 60)
    print("YOLOv5 RGB-T 모델 테스트")
    print("=" * 60)
    
    weight_path = Path(weight_file)
    if weight_path.exists():
        print(f"\n{'='*50}")
        success1 = test_rgbt_model(weight_path)
        success2 = test_with_attempt_load(weight_path)
        
        if success1 or success2:
            print(f"\n✓ 최종 결과: RGB-T 모델이 정상적으로 동작합니다!")
        else:
            print(f"\n✗ 최종 결과: 모든 시도 실패")
        print(f"{'='*50}")
    else:
        print(f"\n✗ 파일이 존재하지 않습니다: {weight_file}")

if __name__ == "__main__":
    main()
