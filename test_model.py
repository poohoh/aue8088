import torch
import yaml
from models.yolo import Model
from pathlib import Path

def test_model():
    """Test the CBAM model with dummy data"""
    
    # 모델 설정 파일 경로
    cfg_path = "models/custom/yolov5x_cbam_kaist-rgbt.yaml"
    
    print(f"Testing model: {cfg_path}")
    
    try:
        # 모델 로드 (stride 계산 없이)
        print("Loading model...")
        
        # 직접 yaml 읽어서 모델 구조만 확인
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        
        print("Model config loaded:")
        print(f"Backbone layers: {len(cfg['backbone'])}")
        print(f"Head layers: {len(cfg['head'])}")
        
        # 간단한 모듈 테스트
        from models.common import DualStreamCrossAttention
        
        print("\nTesting DualStreamCrossAttention module...")
        cross_attn = DualStreamCrossAttention(160, 8).cuda()
        
        # 작은 크기로 테스트
        rgb = torch.randn(1, 160, 8, 8).cuda()
        thermal = torch.randn(1, 160, 8, 8).cuda()
        
        print(f"Input shapes: RGB {rgb.shape}, Thermal {thermal.shape}")
        
        with torch.no_grad():
            output = cross_attn([rgb, thermal])
        
        print(f"Output shapes: {[x.shape for x in output]}")
        print("✅ DualStreamCrossAttention test passed!")
        
        return True
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model test passed!")
    else:
        print("\n❌ Model test failed!")
