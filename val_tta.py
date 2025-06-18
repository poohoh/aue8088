# val_tta.py

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# 경로 설정 (프로젝트 구조에 맞게 조정 필요)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.dataloaders import create_dataloader
from utils.general import (
    LOGGER, TQDM_BAR_FORMAT, check_dataset, check_img_size, check_yaml,
    colorstr, increment_path, print_args, scale_boxes, xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.torch_utils import select_device, smart_inference_mode
# TTA 핵심 함수 import
from utils.test_time_augmentation import run_tta


# <<< 이식된 함수 1: val.py의 save_one_json 함수
def save_one_json(predn, jdict, path, index, class_map):
    """
    Saves one JSON detection result with image ID, category ID, bounding box, and score.

    Example: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    """
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        if p[4] < 0.1:  # Lower threshold to include more predictions
            continue
        jdict.append(
            {
                "image_name": image_id,
                "image_id": int(index),
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


@smart_inference_mode()
def run_tta_validation(
    data,
    weights=None,
    batch_size=32,
    imgsz=640,
    conf_thres=0.001,
    iou_thres=0.40,
    task="val",
    device="",
    workers=8,
    single_cls=False,
    verbose=False,
    save_txt=False,
    save_json=False, # <<< 옵션 추가
    project="runs/val",
    name="exp_tta",
    exist_ok=False,
    half=True,
    rgbt=False,          # <<< 새 인자
    fast_tta=False,      # <<< 빠른 TTA 모드
):
    """TTA를 사용하여 검증을 실행하고 JSON 결과를 저장합니다."""
    device = select_device(device, batch_size=batch_size)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 모델 로드
    if rgbt:
        # RGBT 모델의 경우 attempt_load 사용 (다중 스트림 지원) (3채널, 3채널의 lwir, rgb 데이터가 2개 스트림으로)
        model = attempt_load(weights, device=device)  # 이미 .to(device) 수행
        model.eval()  # ← 추가
        if half:  # half-precision
            model.half()  # 반드시 .to(device) 이후에 호출
        stride = max(model.stride) if isinstance(model.stride, (list, tuple)) else int(model.stride.max()) if hasattr(model.stride, 'max') else model.stride
    else:
        # 일반 모델의 경우 DetectMultiBackend 사용
        model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
        model.eval()  # 추론 모드 고정
        stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    
    # names 확보 (커스텀 두-스트림 가중치에 .names 속성이 없을 때 대비)
    names = model.names if hasattr(model, "names") else data.get("names", [])
    
    # 데이터로더 생성
    data = check_dataset(data)
    dataloader = create_dataloader(
        data[task], imgsz, batch_size, stride, single_cls,
        pad=0.5, rect=False, workers=workers,
        prefix=colorstr(f"{task}: "),
        rgbt_input=rgbt              # <<< 두-스트림 로드
    )[0]

    # TTA 실행 (반환값 pred는 '원본 이미지' 좌표계 기준)
    predictions, targets_list, paths, shapes, indices = run_tta(
        model=model, dataloader=dataloader, device=device,
        conf_thres=conf_thres, iou_thres=iou_thres,
        verbose=verbose, rgbt=rgbt, fast_mode=fast_tta      # <<< 전달
    )

    # 평가 루프 변수 초기화
    stats, ap, ap_class = [], [], []
    nc = int(data["nc"])
    names = model.names
    jdict, class_map = [], list(range(1000))
    
    # 기본 메트릭 값 초기화
    mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    
    for i, pred_original in enumerate(tqdm(predictions, desc="Evaluating predictions")):
        labels = targets_list[i][:, 1:].to(device)
        
        pred_letterbox = pred_original.clone()
        
        # ========================= 핵심 수정 부분 ② =========================
        # scale_boxes 호출 시 dataloader가 제공한 ratio_pad 정보를 명시적으로 전달하여 정밀도 향상
        if pred_letterbox.shape[0] > 0:
            pred_letterbox[:, :4] = scale_boxes(
                shapes[i][0],               # from_shape: (h0, w0)
                pred_letterbox[:, :4],      # boxes
                (imgsz, imgsz),             # to_shape
                ratio_pad=shapes[i][1]      # ratio_pad: (ratio, (dw, dh))
            )
        # =================================================================

        if save_json:
            save_one_json(pred_original, jdict, Path(paths[i]), indices[i], class_map)

        # mAP 계산을 위한 통계 (이하 동일)
        correct = torch.zeros(pred_letterbox.shape[0], 10, dtype=torch.bool, device=device)
        if labels.shape[0]:
            from utils.metrics import box_iou
            iou = box_iou(labels[:, 1:], pred_letterbox[:, :4])
            correct_class = labels[:, 0:1] == pred_letterbox[:, 5]
            for j in range(10):
                x = torch.where((iou >= (0.5 + 0.05 * j)) & correct_class)
                if x[0].shape[0]:
                    matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                    if x[0].shape[0] > 1:
                        matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), j] = True
        
        stats.append((correct.cpu(), pred_letterbox[:, 4].cpu(), pred_letterbox[:, 5].cpu(), labels[:, 0].cpu()))

    # 최종 메트릭 계산 및 출력 (기존과 동일)
    stats = [torch.cat(x, 0).numpy() for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=nc)
        
        LOGGER.info(("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"))
        LOGGER.info(("%22s" + "%11i" * 2 + "%11.3g" * 4) % ("all", len(paths), nt.sum(), mp, mr, map50, map))

    # <<< 추가: JSON 파일 저장 및 kaisteval 실행 로직
    if save_json and len(jdict):
        w = Path(weights).stem if weights is not None else 'model'
        pred_json = str(save_dir / f"{w}_predictions.json")
        LOGGER.info(f"\nSaving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f, indent=2)
        
        try:
            # KAIST 데이터셋 평가 스크립트가 있다면 실행
            ann_file = ROOT / 'utils/eval/KAIST_val-A_annotation.json'
            if not ann_file.exists():
                LOGGER.warning(f"KAIST annotation file not found at {ann_file}, skipping kaisteval.")
            else:
                LOGGER.info(f"Running kaisteval on {pred_json}...")
                os.system(f"python3 {ROOT / 'utils/eval/kaisteval.py'} --annFile {ann_file} --rstFile {pred_json}")
        except Exception as e:
            LOGGER.warning(f"kaisteval unable to run: {e}")


    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    return (mp, mr, map50, map), None, None


def parse_opt():
    """`val_tta.py`를 직접 실행하기 위한 Argument Parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp_tta', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--rgbt', action='store_true', help='two-stream RGB-T input')
    parser.add_argument('--fast-tta', action='store_true', help='use fast TTA (fewer augmentations)')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    run_tta_validation(**vars(opt))