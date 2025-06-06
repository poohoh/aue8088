# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    """
    Check YOLOv5 training batch size and suggest optimal batch size.
    
    Args:
        model: YOLOv5 model
        imgsz (int): image size
        amp (bool): use automatic mixed precision
        
    Returns:
        batch_size (int): optimal batch size
    """
    # Check device
    device = next(model.parameters()).device
    if device.type == 'cpu':
        return 16

    # Check memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # (GiB)
    r = torch.cuda.memory_reserved(device) / gb  # (GiB)
    a = torch.cuda.memory_allocated(device) / gb  # (GiB)
    f = t - (r + a)  # free inside reserved
    LOGGER.info(f'{d} ({properties.name}) {t:.1f}G total, {r:.1f}G reserved, {a:.1f}G allocated, {f:.1f}G free')

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)

        # Fit a solution
        y = [x[2] for x in results if x]  # memory [2]
        p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
        b = int((f * 0.9 - p[1]) / p[0])  # optimal batch size
        if None in results:  # some sizes failed
            i = results.index(None)  # first fail index
            if b >= batch_sizes[i]:  # if optimal size is greater than max successful size
                b = batch_sizes[max(i - 1, 0)]  # select prior safe point
        if b < 1 or b > 1024:  # b outside of safe range
            b = 16

        LOGGER.info(f'{colorstr("AutoBatch: ")}Using batch-size {b} for {d} {t:.1f}G total memory, {f:.1f}G free')
        return b
    except Exception as e:
        LOGGER.warning(f'{colorstr("AutoBatch: ")}CUDA Exception in batch size detection: {e}')
        return 16


def autobatch(model, imgsz=640, fraction=0.9, batch_size=16):
    """
    Automatically estimate best batch size to use `fraction` of available CUDA memory.
    
    Args:
        model: YOLO model to use for batch size estimation
        imgsz (int): image size used for training
        fraction (float): fraction of available CUDA memory to use
        batch_size (int): default batch size to return if unable to estimate
        
    Returns:
        batch_size (int): compute optimal batch size
    """
    # Check device
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    device = next(model.parameters()).device
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.1f}G total, {r:.1f}G reserved, {a:.1f}G allocated, {f:.1f}G free')

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.zeros(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)

        # Fit a solution
        y = [x[2] for x in results if x]  # memory [2]
        p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
        b = int((f * fraction - p[1]) / p[0])  # optimal batch size
        if None in results:  # some sizes failed
            i = results.index(None)  # first fail index
            if b >= batch_sizes[i]:  # if optimal size is greater
                b = batch_sizes[max(i - 1, 0)]  # select prior safe point
        if b < 1 or b > 1024:  # b outside of safe range
            b = batch_size

        LOGGER.info(f'{prefix}Using batch-size {b} for {d} {f:.1f}G available')
        return b
    except Exception as e:
        LOGGER.warning(f'{prefix}Exception: {e}, using default batch-size {batch_size}')
        return batch_size
