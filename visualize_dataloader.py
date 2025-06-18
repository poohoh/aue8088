#!/usr/bin/env python3
"""
Test YOLOv5 RGB-T dataloader with various augmentations
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
import shutil
import yaml

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import create_dataloader, LoadRGBTImagesAndLabels
from utils.general import (
    LOGGER,
    check_dataset,
    check_yaml,
    colorstr,
    xywhn2xyxy,
    init_seeds,
)
from utils.augmentations import letterbox


def draw_boxes(img, labels, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image"""
    h, w = img.shape[:2]
    
    for i, label in enumerate(labels):
        if len(label) < 5:
            continue
        cls, x_center, y_center, width, height = label[:5]
        if cls < 0:  # ignore label
            continue
        
        # Convert normalized xywh to pixel xyxy
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw class label
        label_text = f"Class {int(cls)}"
        cv2.putText(img, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, thickness)
    
    return img


def save_comparison_image(save_path, original_imgs, augmented_imgs, original_labels, 
                         augmented_labels, aug_name):
    """Save comparison image showing original and augmented images with labels"""
    # Convert tensors to numpy arrays if needed
    if isinstance(augmented_imgs[0], torch.Tensor):
        aug_lwir = augmented_imgs[0].permute(1, 2, 0).numpy() * 255
        aug_vis = augmented_imgs[1].permute(1, 2, 0).numpy() * 255
    else:
        aug_lwir = augmented_imgs[0]
        aug_vis = augmented_imgs[1]
    
    # Convert to uint8
    aug_lwir = aug_lwir.astype(np.uint8)
    aug_vis = aug_vis.astype(np.uint8)
    
    # Original images
    orig_lwir = original_imgs[0].copy()
    orig_vis = original_imgs[1].copy()
    
    # Draw boxes on images BEFORE resizing to maintain label accuracy
    orig_lwir_boxed = draw_boxes(orig_lwir.copy(), original_labels, (0, 255, 0))
    orig_vis_boxed = draw_boxes(orig_vis.copy(), original_labels, (0, 255, 0))
    aug_lwir_boxed = draw_boxes(aug_lwir.copy(), augmented_labels, (0, 255, 255))
    aug_vis_boxed = draw_boxes(aug_vis.copy(), augmented_labels, (0, 255, 255))
    
    # Now resize images to match - use the larger dimensions for better quality
    target_h = max(orig_lwir.shape[0], aug_lwir.shape[0])
    target_w = max(orig_lwir.shape[1], aug_lwir.shape[1])
    
    # Resize all images to target size
    orig_lwir_boxed = cv2.resize(orig_lwir_boxed, (target_w, target_h))
    orig_vis_boxed = cv2.resize(orig_vis_boxed, (target_w, target_h))
    aug_lwir_boxed = cv2.resize(aug_lwir_boxed, (target_w, target_h))
    aug_vis_boxed = cv2.resize(aug_vis_boxed, (target_w, target_h))
    
    # Create comparison grid (2x2)
    # Top row: original LWIR, original Visible
    # Bottom row: augmented LWIR, augmented Visible
    h, w = target_h, target_w
    grid = np.zeros((h*2 + 20, w*2 + 20, 3), dtype=np.uint8)
    
    # Fill with white background
    grid.fill(255)
    
    # Place images
    grid[10:h+10, 10:w+10] = orig_lwir_boxed
    grid[10:h+10, w+20:w*2+20] = orig_vis_boxed
    grid[h+20:h*2+20, 10:w+10] = aug_lwir_boxed
    grid[h+20:h*2+20, w+20:w*2+20] = aug_vis_boxed
    
    # Add labels
    cv2.putText(grid, "Original LWIR", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 2)
    cv2.putText(grid, "Original Visible", (w+20, 25), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 2)
    cv2.putText(grid, f"{aug_name} LWIR", (10, h+35), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 2)
    cv2.putText(grid, f"{aug_name} Visible", (w+20, h+35), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 0), 2)
    
    # Save
    cv2.imwrite(str(save_path), grid)
    LOGGER.info(f"Saved comparison image: {save_path}")


def test_augmentations(dataset, save_dir, num_samples=5):
    """Test different augmentation techniques"""
    samples_found = 0
    sample_indices = []
    
    # Find samples with labels
    LOGGER.info("Finding samples with labels...")
    for i in range(len(dataset)):
        if len(dataset.labels[i]) > 0 and samples_found < num_samples:
            # Check if any label is not ignored (class >= 0)
            valid_labels = dataset.labels[i][dataset.labels[i][:, 0] >= 0]
            if len(valid_labels) > 0:
                sample_indices.append(i)
                samples_found += 1
    
    LOGGER.info(f"Found {samples_found} samples with valid labels")
    
    # Augmentation configurations to test
    augmentations = [
        {"name": "No_Augmentation", "config": {}},
        {"name": "HSV", "config": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4}},
        {"name": "Rotation", "config": {"degrees": 10.0}},
        {"name": "Translation", "config": {"translate": 0.1}},
        {"name": "Scale", "config": {"scale": 0.5}},
        {"name": "Shear", "config": {"shear": 10.0}},
        {"name": "Perspective", "config": {"perspective": 0.0005}},
        {"name": "FlipLR", "config": {"fliplr": 1.0}},
        {"name": "FlipUD", "config": {"flipud": 1.0}},
        {"name": "Mosaic", "config": {"mosaic": 1.0}},
        {"name": "MixUp", "config": {"mixup": 1.0, "mosaic": 1.0}},
    ]
    
    # Process each sample
    for sample_idx, dataset_idx in enumerate(sample_indices):
        LOGGER.info(f"\nProcessing sample {sample_idx + 1}/{samples_found}")
        
        # Get original images and labels
        original_imgs, _, (h, w) = dataset.load_image(dataset_idx)
        original_labels = dataset.labels[dataset_idx].copy()
        
        # Remove extra columns if present (keep only [class, x, y, w, h])
        if original_labels.shape[1] > 5:
            original_labels = original_labels[:, :5]
        
        # Remove ignored labels for visualization
        valid_mask = original_labels[:, 0] >= 0
        original_labels_vis = original_labels[valid_mask]
        
        # Create sample directory
        sample_dir = save_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save original image
        save_comparison_image(
            sample_dir / "00_original.jpg",
            original_imgs,
            original_imgs,
            original_labels_vis,
            original_labels_vis,
            "Original"
        )
        
        # Test each augmentation
        for aug_idx, aug_config in enumerate(augmentations):
            aug_name = aug_config["name"]
            LOGGER.info(f"  Testing {aug_name}...")
            
            # Create a temporary dataset with specific augmentation
            temp_dataset = LoadRGBTImagesAndLabels(
                dataset.path,
                img_size=dataset.img_size,
                batch_size=1,
                augment=True,
                hyp={**dataset.hyp, **aug_config["config"]},
                rect=False,
                cache_images=False,
                single_cls=False,
                stride=dataset.stride,
                pad=0.5,
                prefix="",
                rank=-1,
            )
            
            # Copy necessary attributes
            temp_dataset.indices = dataset.indices
            temp_dataset.ims = dataset.ims
            temp_dataset.im_files = dataset.im_files
            temp_dataset.label_files = dataset.label_files
            temp_dataset.labels = dataset.labels
            temp_dataset.segments = dataset.segments
            temp_dataset.shapes = dataset.shapes
            # Copy optional attributes if they exist
            if hasattr(dataset, 'im_hw0'):
                temp_dataset.im_hw0 = dataset.im_hw0
            if hasattr(dataset, 'im_hw'):
                temp_dataset.im_hw = dataset.im_hw
            
            # Set specific augmentation parameters
            if aug_name == "No_Augmentation":
                temp_dataset.augment = False
                temp_dataset.mosaic = False
            elif aug_name in ["Mosaic", "MixUp"]:
                temp_dataset.mosaic = True
            else:
                temp_dataset.mosaic = False
                # Set all other augmentations to 0 except the one we're testing
                for key in ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", 
                           "scale", "shear", "perspective", "fliplr", "flipud", "mixup"]:
                    temp_dataset.hyp[key] = aug_config["config"].get(key, 0.0)
            
            # Get augmented sample
            try:
                imgs, labels, _, _, _ = temp_dataset[dataset_idx]
                
                # Debug output for detailed analysis
                print(f"\n=== {aug_name} ===")
                print(f"Original image shapes: LWIR={original_imgs[0].shape}, VIS={original_imgs[1].shape}")
                if isinstance(imgs[0], torch.Tensor):
                    print(f"Augmented image shapes: LWIR={imgs[0].shape}, VIS={imgs[1].shape}")
                else:
                    print(f"Augmented image shapes: LWIR={imgs[0].shape}, VIS={imgs[1].shape}")
                print(f"Original labels: {original_labels_vis}")
                
                # Convert labels to numpy for visualization
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.numpy()
                else:
                    labels_np = labels
                
                print(f"Raw augmented labels shape: {labels_np.shape}")
                print(f"Raw augmented labels: {labels_np}")
                
                # Remove batch index if present (shape will be (N, 6) with batch index, (N, 5) without)
                if labels_np.shape[1] == 6:
                    labels_np = labels_np[:, 1:]  # Remove batch index - now shape is (N, 5)
                    print(f"After removing batch index: {labels_np}")
                # If shape is already (N, 5), no need to remove anything
                
                # Filter out ignored labels (class < 0)
                valid_mask = labels_np[:, 0] >= 0
                labels_vis = labels_np[valid_mask]
                print(f"Final augmented labels: {labels_vis}")
                
                if aug_name == "No_Augmentation":
                    print(f"Labels are equal? {np.array_equal(original_labels_vis, labels_vis)}")
                    print(f"Images are equal? LWIR: {np.array_equal(original_imgs[0], imgs[0] if not isinstance(imgs[0], torch.Tensor) else (imgs[0].permute(1,2,0).numpy() * 255).astype(np.uint8))}")
                
                # Save comparison
                save_comparison_image(
                    sample_dir / f"{aug_idx+1:02d}_{aug_name}.jpg",
                    original_imgs,
                    imgs,
                    original_labels_vis,
                    labels_vis,
                    aug_name
                )
                
            except Exception as e:
                LOGGER.warning(f"  Failed to test {aug_name}: {e}")
                continue
    
    LOGGER.info(f"\nAll tests completed. Results saved to {save_dir}")


def main(opt):
    # Initialize
    init_seeds(opt.seed)
    
    # Setup save directory
    save_dir = Path(opt.save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Testing dataloader with configuration from {opt.data}")
    
    # Load dataset configuration
    data_dict = check_dataset(check_yaml(opt.data))
    train_path = data_dict['train']
    
    # Load hyperparameters
    hyp_path = check_yaml(opt.hyp)
    with open(hyp_path, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    
    # Create dataset (not dataloader to have more control)
    LOGGER.info("Creating dataset...")
    dataset = LoadRGBTImagesAndLabels(
        train_path,
        img_size=opt.imgsz,
        batch_size=opt.batch_size,
        augment=True,
        hyp=hyp,
        rect=False,
        cache_images=False,
        single_cls=opt.single_cls,
        stride=32,
        pad=0.5,
        prefix=colorstr("train: "),
        rank=-1,
    )
    
    LOGGER.info(f"Dataset loaded: {len(dataset)} images")
    
    # Test augmentations
    test_augmentations(dataset, save_dir, num_samples=opt.num_samples)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, 
                       default=ROOT / 'datasets/kaist-rgbt/kfold_splits/yaml_configs/kaist-rgbt-fold1.yaml',
                       help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, 
                       default=ROOT / 'data/hyps/hyp.scratch-low.yaml',
                       help='hyperparameters path')
    parser.add_argument('--imgsz', '--img-size', type=int, default=640, 
                       help='train image size')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='batch size')
    parser.add_argument('--single-cls', action='store_true', 
                       help='train as single-class dataset')
    parser.add_argument('--num-samples', type=int, default=5, 
                       help='number of samples to test')
    parser.add_argument('--save-dir', type=str, default='visualize_dataloader', 
                       help='directory to save visualization results')
    parser.add_argument('--seed', type=int, default=0, 
                       help='random seed')
    
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)