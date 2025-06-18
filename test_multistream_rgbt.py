#!/usr/bin/env python3
"""
RGB-T Multi-Stream 모델 테스트 (분리된 입력)
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

def test_multistream_rgbt_model(weight_path):
    """Multi-Stream RGB-T 모델 테스트"""
    print(f"테스트할 weight 파일: {weight_path}")
    
    try:
        # 1. 모델 로딩
        print("\n1. Multi-Stream RGB-T 모델 로딩 중...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model = checkpoint['model'].float().eval().to(device)
        
        print(f"✓ 모델 로딩 성공")
        print(f"✓ 클래스: {model.names}")
        print(f"✓ 클래스 수: {model.nc}")
        
        # 2. Multi-Stream 입력 데이터 생성
        print(f"\n2. Multi-Stream 입력 데이터 생성...")
        batch_size = 1
        height, width = 640, 640
        
        # RGB 스트림 (3채널)
        rgb_stream = torch.randn(batch_size, 3, height, width).to(device)
        
        # Thermal 스트림 (1채널)  
        thermal_stream = torch.randn(batch_size, 1, height, width).to(device)
        
        # 입력을 리스트로 구성
        multi_stream_input = [rgb_stream, thermal_stream]
        
        print(f"✓ RGB 스트림: {rgb_stream.shape}")
        print(f"✓ Thermal 스트림: {thermal_stream.shape}")
        print(f"✓ 입력 스트림 수: {len(multi_stream_input)}")
        
        # 3. Multi-Stream 추론 실행
        print(f"\n3. Multi-Stream 추론 테스트...")
        model.eval()
        with torch.no_grad():
            pred = model(multi_stream_input)
        
        print(f"✓ Multi-Stream 추론 성공!")
        
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
        
        # 4. 다양한 입력 크기 테스트
        print(f"\n4. 다양한 입력 크기 테스트...")
        try:
            # 다른 크기로 테스트
            sizes = [320, 416, 512]
            for size in sizes:
                rgb_test = torch.randn(1, 3, size, size).to(device)
                thermal_test = torch.randn(1, 1, size, size).to(device)
                test_input = [rgb_test, thermal_test]
                
                with torch.no_grad():
                    pred_test = model(test_input)
                
                print(f"  ✓ 크기 {size}x{size}: 성공")
                
        except Exception as e:
            print(f"  ✗ 다양한 크기 테스트 오류: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_stream_error(weight_path):
    """단일 스트림 입력 오류 확인"""
    print(f"\n5. 단일 스트림 입력 오류 확인...")
    
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model = checkpoint['model'].float().eval().to(device)
        
        # 단일 텐서 입력 (오류 발생 예상)
        single_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            pred = model(single_input)
            
        print(f"✗ 예상과 다름: 단일 입력이 성공함")
        return False
        
    except Exception as e:
        print(f"✓ 예상대로 단일 입력 실패: {str(e)[:50]}...")
        return True

def main():
    weight_file = "second/s_bat32_epo300_low_mosaic2/weights/best.pt"
    
    print("=" * 60)
    print("YOLOv5 Multi-Stream RGB-T 모델 테스트")
    print("=" * 60)
    
    weight_path = Path(weight_file)
    if weight_path.exists():
        print(f"\n{'='*50}")
        success1 = test_multistream_rgbt_model(weight_path)
        success2 = test_single_stream_error(weight_path)
        
        if success1:
            print(f"\n✓ 최종 결과: Multi-Stream RGB-T 모델이 정상적으로 동작합니다!")
            print(f"✓ 중요: RGB와 Thermal을 분리된 스트림으로 입력해야 함")
        else:
            print(f"\n✗ 최종 결과: Multi-Stream 테스트 실패")
        print(f"{'='*50}")
    else:
        print(f"\n✗ 파일이 존재하지 않습니다: {weight_file}")

if __name__ == "__main__":
    main()
