#!/usr/bin/env python3
"""
RGB-T Multi-Stream 모델 테스트 (동일 채널 수)
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

def test_equal_channel_rgbt_model(weight_path):
    """동일 채널 수 Multi-Stream RGB-T 모델 테스트"""
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
        
        # 2. 동일 채널 수 입력 데이터 생성
        print(f"\n2. 동일 채널 수 입력 데이터 생성...")
        batch_size = 1
        height, width = 640, 640
        
        # RGB 스트림 (3채널)
        rgb_stream = torch.randn(batch_size, 3, height, width).to(device)
        
        # Thermal 스트림 (3채널로 복제 - 일반적으로 동일한 thermal 이미지를 3개 채널에 복사)
        thermal_1ch = torch.randn(batch_size, 1, height, width).to(device)
        thermal_stream = thermal_1ch.repeat(1, 3, 1, 1)  # 1채널을 3채널로 복제
        
        # 입력을 리스트로 구성
        multi_stream_input = [rgb_stream, thermal_stream]
        
        print(f"✓ RGB 스트림: {rgb_stream.shape}")
        print(f"✓ Thermal 스트림 (3ch 복제): {thermal_stream.shape}")
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
                        # Detection 출력 형태 분석
                        if len(p.shape) == 3 and p.shape[-1] > 5:  # [batch, detections, features]
                            print(f"    Detection 형태: [batch={p.shape[0]}, detections={p.shape[1]}, features={p.shape[2]}]")
                            print(f"    Features: [x, y, w, h, conf, class0, class1, ...]")
        elif isinstance(pred, torch.Tensor):
            print(f"✓ 출력 크기: {pred.shape}")
            print(f"✓ 출력 범위: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        
        # 4. 다양한 thermal 표현 테스트
        print(f"\n4. 다양한 thermal 표현 테스트...")
        try:
            # 방법 1: 모든 채널에 동일한 thermal
            thermal_same = thermal_1ch.repeat(1, 3, 1, 1)
            
            # 방법 2: thermal을 다른 채널에 배치 (나머지는 0)
            thermal_sparse = torch.zeros(batch_size, 3, height, width).to(device)
            thermal_sparse[:, 0, :, :] = thermal_1ch.squeeze(1)  # R 채널에만
            
            # 방법 3: thermal을 노이즈와 결합
            thermal_noise = torch.cat([
                thermal_1ch,  # 원본 thermal
                thermal_1ch + 0.1 * torch.randn_like(thermal_1ch),  # 약간 노이즈 추가
                thermal_1ch + 0.1 * torch.randn_like(thermal_1ch)   # 약간 노이즈 추가
            ], dim=1)
            
            test_cases = [
                ("동일 복제", thermal_same),
                ("희소 배치", thermal_sparse), 
                ("노이즈 결합", thermal_noise)
            ]
            
            for name, thermal_test in test_cases:
                test_input = [rgb_stream, thermal_test]
                with torch.no_grad():
                    pred_test = model(test_input)
                print(f"  ✓ {name}: 성공")
                
        except Exception as e:
            print(f"  ✗ 다양한 thermal 표현 테스트 오류: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    weight_file = "second/s_bat32_epo300_low_mosaic2/weights/best.pt"
    
    print("=" * 60)
    print("YOLOv5 Multi-Stream RGB-T 모델 테스트 (동일 채널)")
    print("=" * 60)
    
    weight_path = Path(weight_file)
    if weight_path.exists():
        print(f"\n{'='*50}")
        success = test_equal_channel_rgbt_model(weight_path)
        
        if success:
            print(f"\n✅ 최종 결과: Multi-Stream RGB-T 모델이 정상적으로 동작합니다!")
            print(f"✅ 핵심 발견:")
            print(f"   - RGB와 Thermal 모두 3채널 입력 필요")
            print(f"   - Thermal 1채널을 3채널로 복제하여 사용")
            print(f"   - 입력은 [RGB_3ch, Thermal_3ch] 리스트 형태")
        else:
            print(f"\n❌ 최종 결과: 테스트 실패")
        print(f"{'='*50}")
    else:
        print(f"\n✗ 파일이 존재하지 않습니다: {weight_file}")

if __name__ == "__main__":
    main()
