# utils/test_time_augmentation.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List
import torchvision.transforms.functional as TF
from tqdm import tqdm

from utils.general import non_max_suppression, scale_boxes
from ensemble_boxes import weighted_boxes_fusion

class EnhancedTTA:
    # __init__ 및 _apply_transform 함수는 이전과 동일
    def __init__(self, device='cuda', conf_thres=0.001, iou_thres=0.6, max_det=1000, fast_mode=False):
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        
        if fast_mode:
            # 빠른 모드: 최소한의 증강만 사용 (총 4가지 조합)
            self.scales = [1.0]  # 스케일 변화 없음
            self.flips = ['none', 'horizontal']  # 플립만 사용
            self.rotations = [0]  # 회전 없음
            self.color_jitters = [
                {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0},
                {'brightness': 0.9, 'contrast': 1.0, 'saturation': 1.0},
            ]
        else:
            # 기본 모드: 전체 증강 사용 (총 54가지 조합)
            self.scales = [0.8, 1.0, 1.2]
            self.flips = ['none', 'horizontal']
            self.rotations = [-10, 0, 10]
            self.color_jitters = [
                {'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0},
                {'brightness': 0.8, 'contrast': 1.0, 'saturation': 1.0},
                {'brightness': 1.2, 'contrast': 1.0, 'saturation': 1.0},
            ]
        
        self.fusion_conf_thr = 0.05

    def _apply_to_streams(self, img_list, scale, flip, rot, cj):
        """모든 스트림에 동일한 변환을 적용"""
        out, info = [], None
        for s, im in enumerate(img_list):
            # thermal(1-채널)에는 색상 jitter 생략
            cj_use = cj if (im.shape[1] == 3) else {'brightness':1.0,'contrast':1.0,'saturation':1.0}
            im_t, info = self._apply_transform(im, scale, flip, rot, cj_use)
            out.append(im_t)
        return out, info      # info 는 스트림 공통

    def _apply_transform(self, img, scale, flip_type, rotation, color_jitter):
        h, w = img.shape[2:]
        original_dtype = img.dtype  # 원본 데이터 타입 저장
        transform_info = {
            'scale': scale, 'flip_type': flip_type, 'rotation': rotation,
            'crop_offset_x': 0, 'crop_offset_y': 0, 'original_shape': (h, w)
        }
        
        # 색상 변환은 CPU에서 float32로 수행 (half-precision 호환성 확보)
        img_cpu = img.cpu().float()
        img_cpu = TF.adjust_brightness(img_cpu, color_jitter['brightness'])
        img_cpu = TF.adjust_contrast(img_cpu, color_jitter['contrast'])
        img_cpu = TF.adjust_saturation(img_cpu, color_jitter['saturation'])
        img = img_cpu.to(self.device).to(original_dtype)  # 원본 데이터 타입으로 복원
        
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
            if scale > 1.0:
                transform_info['crop_offset_y'] = (new_h - h) // 2
                transform_info['crop_offset_x'] = (new_w - w) // 2
                img = img[:, :, transform_info['crop_offset_y']:transform_info['crop_offset_y'] + h,
                          transform_info['crop_offset_x']:transform_info['crop_offset_x'] + w]
            else:
                pad_h = int(round((h - new_h) / 2.0))
                pad_w = int(round((w - new_w) / 2.0))
                img = F.pad(img, (pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h), value=0.447)
        if flip_type == 'horizontal':
            img = torch.flip(img, dims=[3])
        if rotation != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img_np = img.cpu().numpy().transpose(0, 2, 3, 1)[0]
            img_np = cv2.warpAffine(img_np, M, (w, h), borderValue=(0.447, 0.447, 0.447))
            img = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device).to(original_dtype)  # 원본 데이터 타입으로 복원
            transform_info['rotation_matrix'] = M
        else:
            transform_info['rotation_matrix'] = None
        return img, transform_info

    def _reverse_augmentations(self, pred, info):
        """
        <<< 수정됨 (피드백 ①): 역변환 순서를 Rotation -> Flip -> Scale 로 수정
        """
        if len(pred) == 0:
            return pred
            
        pred = pred.clone()
        h, w = info['original_shape']
        scale = info['scale']
        
        # 1. 회전 역변환 (가장 먼저 수행)
        if info['rotation_matrix'] is not None:
            M_inv = cv2.invertAffineTransform(info['rotation_matrix'])
            M_inv_t = torch.from_numpy(M_inv).float().to(pred.device)
            boxes = pred[:, :4]
            corners = torch.cat([
                boxes[:, :2], boxes[:, 2:3], boxes[:, 1:2],
                boxes[:, 2:], boxes[:, 0:1], boxes[:, 3:4]
            ], dim=1).view(-1, 4, 2)
            corners_hom = torch.cat([corners, torch.ones(corners.shape[0], 4, 1, device=pred.device)], dim=2)
            transformed_corners = torch.matmul(corners_hom, M_inv_t.T.unsqueeze(0))
            x_, y_ = transformed_corners[:, :, 0], transformed_corners[:, :, 1]
            pred[:, 0], pred[:, 1] = torch.min(x_, dim=1)[0], torch.min(y_, dim=1)[0]
            pred[:, 2], pred[:, 3] = torch.max(x_, dim=1)[0], torch.max(y_, dim=1)[0]

        # 2. 플립 역변환
        if info['flip_type'] == 'horizontal':
            pred[:, [0, 2]] = w - pred[:, [2, 0]]
        
        # 3. 스케일 역변환 (가장 마지막에 수행)
        if scale != 1.0:
            if scale > 1.0:
                pred[:, [0, 2]] += info['crop_offset_x']
                pred[:, [1, 3]] += info['crop_offset_y']
                pred[:, :4] /= scale
            else:
                new_h, new_w = int(h * scale), int(w * scale)
                pad_h = int(round((h - new_h) / 2.0))
                pad_w = int(round((w - new_w) / 2.0))
                pred[:, [0, 2]] -= pad_w
                pred[:, [1, 3]] -= pad_h
                pred[:, :4] /= scale
        
        return pred

    # fuse_predictions 함수는 이전과 동일
    def fuse_predictions(self, predictions_list: List[torch.Tensor], original_shape):
        if not predictions_list:
            return torch.zeros((0, 6), device=self.device)
        h, w = original_shape
        all_boxes, all_scores, all_labels = [], [], []
        for pred in predictions_list:
            if len(pred) > 0:
                boxes = pred[:, :4].clone()
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h
                all_boxes.append(boxes)
                all_scores.append(pred[:, 4])
                all_labels.append(pred[:, 5])
        if not all_boxes:
            return torch.zeros((0, 6), device=self.device)
        boxes, scores, labels = weighted_boxes_fusion(
            [b.cpu().float().numpy() for b in all_boxes],
            [s.cpu().float().numpy() for s in all_scores],
            [l.cpu().float().numpy() for l in all_labels],
            weights=None, iou_thr=self.iou_thres, skip_box_thr=self.fusion_conf_thr)
        fused = torch.zeros((len(boxes), 6), device=self.device)
        if len(boxes) > 0:
            fused[:, :4] = torch.from_numpy(boxes).float().to(self.device)
            fused[:, 4] = torch.from_numpy(scores).float().to(self.device)
            fused[:, 5] = torch.from_numpy(labels).float().to(self.device)
            fused[:, [0, 2]] *= w
            fused[:, [1, 3]] *= h
        return fused

def run_tta(model, dataloader, device, conf_thres=0.001, iou_thres=0.6,
            verbose=True, rgbt=False, fast_mode=False):
    tta = EnhancedTTA(device=device, conf_thres=conf_thres, iou_thres=iou_thres, fast_mode=fast_mode)
    model.eval()
    is_half = next(model.parameters()).dtype == torch.float16
    
    all_fused_predictions, all_targets, all_paths, all_shapes, all_indices = [], [], [], [], []
    pbar = tqdm(dataloader, desc='Running Enhanced TTA') if verbose else dataloader
    
    with torch.no_grad():
        for imgs, targets, paths, shapes, indices in pbar:
            multi = isinstance(imgs, list)
            if multi:
                imgs = [x.to(device).float()/255 for x in imgs]
                bs = imgs[0].shape[0]
            else:
                imgs = imgs.to(device).float()/255
                bs = imgs.shape[0]
            
            # <<< 수정됨 (피드백 ③): 불안정한 unique_batch_indices 로직 완전 제거
            
            for i in range(bs):
                if multi:
                    img_single = [x[i:i+1] for x in imgs]   # 두 스트림
                else:
                    img_single = imgs[i:i+1]
                augmented_predictions = []
                first_transform_info = None
                
                for scale in tta.scales:
                    for flip in tta.flips:
                        for rotation in tta.rotations:
                            for cj in tta.color_jitters:
                                if multi:
                                    aug_img, transform_info = tta._apply_to_streams(
                                        [x.clone() for x in img_single],
                                        scale, flip, rotation, cj)
                                else:
                                    aug_img, transform_info = tta._apply_transform(
                                        img_single.clone(), scale, flip, rotation, cj)
                                if first_transform_info is None:
                                    first_transform_info = transform_info
                                
                                if is_half: 
                                    if multi:
                                        aug_img = [x.half() for x in aug_img]
                                    else:
                                        aug_img = aug_img.half()
                                
                                # ========================= 핵심 수정 부분 ① =========================
                                # 모델 호출 시 RGBT(multi-stream) 입력을 위해 리스트를 언패킹하여 전달
                                if multi:
                                    # aug_img는 [rgb_tensor, thermal_tensor] 리스트
                                    out = model(*aug_img, augment=False)[0] 
                                else:
                                    # aug_img는 단일 텐서
                                    out = model(aug_img, augment=False)[0]
                                # =================================================================

                                pred = non_max_suppression(out, tta.conf_thres, tta.iou_thres, max_det=tta.max_det)[0]
                                
                                if pred is not None and len(pred) > 0:
                                    rev_pred = tta._reverse_augmentations(pred, transform_info)
                                    augmented_predictions.append(rev_pred)
                
                if first_transform_info:
                    fused_pred = tta.fuse_predictions(augmented_predictions, first_transform_info['original_shape'])
                else:
                    fused_pred = torch.zeros((0, 6), device=device)

                all_fused_predictions.append(fused_pred)
                # 루프 변수 i는 배치 내 인덱스를 정확히 가리키므로 그대로 사용
                all_targets.append(targets[targets[:, 0] == i])
                all_paths.append(paths[i])
                all_shapes.append(shapes[i])
                all_indices.append(indices[i]) # 데이터로더가 제공하는 원본 인덱스 저장

    return all_fused_predictions, all_targets, all_paths, all_shapes, all_indices