import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import sys
import os

# Add YOLOv5 root to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.dataloaders import create_dataloader
from utils.general import colorstr, check_dataset, check_yaml
from utils.plots import colors


def visualize_batch(batch, save_dir, batch_idx, names):
    """Visualize a batch of RGB-T images with bounding boxes"""
    imgs, targets, paths, shapes, indices = batch
    
    # Unpack RGB-T images
    if isinstance(imgs, list):
        imgs_thermal, imgs_rgb = imgs[0], imgs[1]
    else:
        print("Warning: Expected list of images but got single tensor")
        return
    
    batch_size = imgs_thermal.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5*batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(batch_size):
        # Get images
        thermal_img = imgs_thermal[idx].cpu().numpy()
        rgb_img = imgs_rgb[idx].cpu().numpy()
        
        # Convert from CHW to HWC
        thermal_img = thermal_img.transpose(1, 2, 0)
        rgb_img = rgb_img.transpose(1, 2, 0)
        
        # Denormalize (0-1 to 0-255)
        thermal_img = (thermal_img * 255).astype(np.uint8)
        rgb_img = (rgb_img * 255).astype(np.uint8)
        
        # Convert RGB from BGR to RGB for display
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # If thermal is single channel, convert to 3 channels for display
        if thermal_img.shape[2] == 1:
            thermal_img = cv2.cvtColor(thermal_img[:,:,0], cv2.COLOR_GRAY2RGB)
        else:
            thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
        
        # Create combined image (side by side)
        h, w = thermal_img.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = thermal_img
        combined[:, w:] = rgb_img
        
        # Get targets for this image
        img_targets = targets[targets[:, 0] == idx]
        
        # Plot thermal image with boxes
        axes[idx, 0].imshow(thermal_img)
        axes[idx, 0].set_title(f'Thermal - {Path(paths[idx]).stem}')
        axes[idx, 0].axis('off')
        
        # Plot RGB image with boxes
        axes[idx, 1].imshow(rgb_img)
        axes[idx, 1].set_title(f'RGB - {Path(paths[idx]).stem}')
        axes[idx, 1].axis('off')
        
        # Plot combined image
        axes[idx, 2].imshow(combined)
        axes[idx, 2].set_title('Combined (Thermal | RGB)')
        axes[idx, 2].axis('off')
        
        # Draw bounding boxes on all images
        for t in img_targets:
            cls = int(t[1])
            # Convert from normalized xywh to pixel xyxy
            x_center, y_center, width, height = t[2:6]
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            x1 = x_center - width/2
            y1 = y_center - height/2
            
            # Get color for class
            color = colors(cls, True)
            color = [c/255.0 for c in color]  # Normalize to 0-1
            
            # Draw on thermal
            rect1 = patches.Rectangle((x1, y1), width, height, 
                                     linewidth=2, edgecolor=color, 
                                     facecolor='none')
            axes[idx, 0].add_patch(rect1)
            axes[idx, 0].text(x1, y1-5, names[cls], color=color, 
                             fontsize=10, weight='bold')
            
            # Draw on RGB
            rect2 = patches.Rectangle((x1, y1), width, height, 
                                     linewidth=2, edgecolor=color, 
                                     facecolor='none')
            axes[idx, 1].add_patch(rect2)
            axes[idx, 1].text(x1, y1-5, names[cls], color=color, 
                             fontsize=10, weight='bold')
            
            # Draw on combined (adjust x coordinate)
            rect3_thermal = patches.Rectangle((x1, y1), width, height, 
                                            linewidth=2, edgecolor=color, 
                                            facecolor='none')
            rect3_rgb = patches.Rectangle((x1+w, y1), width, height, 
                                        linewidth=2, edgecolor=color, 
                                        facecolor='none')
            axes[idx, 2].add_patch(rect3_thermal)
            axes[idx, 2].add_patch(rect3_rgb)
            axes[idx, 2].text(x1, y1-5, names[cls], color=color, 
                            fontsize=10, weight='bold')
    
    plt.tight_layout()
    save_path = save_dir / f'batch_{batch_idx:04d}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved visualization to {save_path}')


def visualize_augmentations(data_yaml, hyp_yaml, num_batches=5, batch_size=4, 
                          img_size=640, save_dir='augmentation_viz'):
    """Visualize augmented RGB-T images from dataloader"""
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load dataset config
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Load hyperparameters
    with open(hyp_yaml, 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Check dataset
    data = check_dataset(data)
    train_path = data['train']
    names = data['names']
    
    # Create dataloader with augmentation
    dataloader, dataset = create_dataloader(
        train_path,
        img_size,
        batch_size,
        32,  # stride
        single_cls=False,
        hyp=hyp,
        augment=True,  # Enable augmentation
        cache=False,
        rect=False,
        rank=-1,
        workers=0,  # Use single worker for visualization
        image_weights=False,
        quad=False,
        prefix=colorstr('train: '),
        shuffle=True,
        seed=0,
        rgbt_input=True  # Enable RGB-T input
    )
    
    print(f'Dataset contains {len(dataset)} images')
    print(f'Visualizing {num_batches} batches with batch_size={batch_size}')
    print(f'Augmentation settings from {hyp_yaml}:')
    print(f'  - Mosaic: {hyp.get("mosaic", 0)}')
    print(f'  - MixUp: {hyp.get("mixup", 0)}')
    print(f'  - HSV: h={hyp.get("hsv_h", 0)}, s={hyp.get("hsv_s", 0)}, v={hyp.get("hsv_v", 0)}')
    print(f'  - Degrees: {hyp.get("degrees", 0)}')
    print(f'  - Translate: {hyp.get("translate", 0)}')
    print(f'  - Scale: {hyp.get("scale", 0)}')
    print(f'  - Shear: {hyp.get("shear", 0)}')
    print(f'  - Perspective: {hyp.get("perspective", 0)}')
    print(f'  - Flip UD: {hyp.get("flipud", 0)}')
    print(f'  - Flip LR: {hyp.get("fliplr", 0)}')
    
    # Visualize batches
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        print(f'\nProcessing batch {i+1}/{num_batches}')
        visualize_batch(batch, save_dir, i, names)
    
    print(f'\nVisualization complete! Check {save_dir} for results.')


def main():
    parser = argparse.ArgumentParser(description='Visualize RGB-T data augmentation')
    parser.add_argument('--data', type=str, default='data/kaist-rgbt.yaml',
                       help='path to dataset yaml')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml',
                       help='path to hyperparameters yaml')
    parser.add_argument('--num-batches', type=int, default=5,
                       help='number of batches to visualize')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='image size')
    parser.add_argument('--save-dir', type=str, default='augmentation_viz',
                       help='directory to save visualizations')
    args = parser.parse_args()
    
    visualize_augmentations(
        args.data,
        args.hyp,
        args.num_batches,
        args.batch_size,
        args.img_size,
        args.save_dir
    )


if __name__ == '__main__':
    main()