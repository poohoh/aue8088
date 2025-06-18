import torch
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

def run_test():
    # --- 설정 ---
    weights = 'yolov5s.pt'  # 사용하는 가중치 파일 (예: yolov5s.pt)
    img_size = 640
    device = select_device('0') # 사용할 GPU 번호
    
    # --- 모델 로드 ---
    print(f"Loading model from {weights}...")
    model = attempt_load(weights, device=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    print(f"Model loaded on {device}. Image size: {imgsz}, Stride: {stride}")

    # --- 샘플 이미지 준비 ---
    # bus.jpg가 없다면, 아무 이미지나 data/images/ 폴더에 넣고 경로를 수정하세요.
    im_path = 'data/images/bus.jpg' 
    print(f"Loading sample image from {im_path}...")
    im = Image.open(im_path).convert('RGB')
    
    # 이미지 전처리 (letterbox와 유사하게)
    im = np.array(im)
    shape = im.shape[:2]
    r = imgsz / max(shape)
    unpad_shape = (int(round(shape[0] * r)), int(round(shape[1] * r)))
    im = torch.from_numpy(im).to(device)
    im = im.permute(2, 0, 1) # HWC to CHW
    im = F.resize(im, size=unpad_shape)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0) # 배치 차원 추가
    print(f"Image preprocessed. Tensor shape: {im.shape}")

    # --- 테스트 실행 ---
    model.eval()

    # 1. FP32 (정밀 모드) 테스트
    print("\n--- Running in FP32 mode ---")
    with torch.no_grad():
        pred_fp32 = model(im)[0]
        print(f"FP32 Output shape: {pred_fp32.shape}")
        print(f"FP32 Output stats: mean={pred_fp32.mean():.4f}, std={pred_fp32.std():.4f}")
        if torch.isnan(pred_fp32).any() or torch.isinf(pred_fp32).any():
            print("❌ FP32 output contains NaN or Inf!")
        else:
            print("✅ FP32 output is stable.")

    # 2. AMP (FP16 모드) 테스트
    print("\n--- Running in AMP (FP16) mode ---")
    with torch.no_grad(), torch.cuda.amp.autocast():
        pred_amp = model(im)[0]
        print(f"AMP Output shape: {pred_amp.shape}")
        print(f"AMP Output stats: mean={pred_amp.mean():.4f}, std={pred_amp.std():.4f}")
        if torch.isnan(pred_amp).any():
            print("❌ AMP output contains NaN!")
        elif torch.isinf(pred_amp).any():
            print("❌ AMP output contains Inf!")
        else:
            print("✅ AMP output is stable.")

    # 3. 비교
    print("\n--- Comparing outputs ---")
    try:
        # AMP 출력을 FP32로 변환하여 비교
        diff = torch.abs(pred_fp32 - pred_amp.float()).max()
        print(f"Max absolute difference between FP32 and AMP outputs: {diff.item():.6f}")
        if torch.allclose(pred_fp32, pred_amp.float(), atol=0.1):
             print("✅✅✅ AMP Check PASSED manually!")
        else:
             print("❌❌❌ AMP Check FAILED manually!")
    except Exception as e:
        print(f"Error during comparison: {e}")


if __name__ == '__main__':
    run_test()