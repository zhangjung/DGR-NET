"""Utility functions: random seeds, worker/batch-size helpers, HSV conversion, folder checks, logging."""
import os
import random
import logging
import multiprocessing

import cv2
import numpy as np
import torch
import psutil


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_random_seeds(seed=42):
    """Set random seeds for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")


def get_optimal_workers():
    """Dynamically determine the optimal number of worker processes."""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024 ** 3)

    if memory_gb >= 32:
        workers = min(cpu_count, 4)
    elif memory_gb >= 8:
        workers = min(cpu_count, 4)
    elif memory_gb >= 8:
        workers = min(cpu_count, 2)
    else:
        workers = 1

    logger.info(f"System memory: {memory_gb:.1f}GB, CPU cores: {cpu_count}, using workers: {workers}")
    return workers


def get_optimized_batch_size(device, model_size='b2'):
    """Compute an optimized batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 8

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # Dual encoders use more GPU memory; the simplified version uses even more
    if model_size == 'b2':
        if gpu_memory_gb >= 24:
            batch_size = 32  # Simplified version uses more memory, keep batch size moderate
        elif gpu_memory_gb >= 16:
            batch_size = 32
        elif gpu_memory_gb >= 12:
            batch_size = 8
        elif gpu_memory_gb >= 8:
            batch_size = 6
        else:
            batch_size = 4
    else:
        batch_size = 8

    logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB, dual-encoder simplified mode, batch size: {batch_size}")
    return batch_size


def rgb_to_hsv_tensor(rgb_tensor):
    """Convert an RGB tensor to HSV and return the S and V channels."""
    batch_size, channels, height, width = rgb_tensor.shape
    assert channels == 3, f"Expected 3 channels for RGB, got {channels}"

    rgb_np = rgb_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    sv_list = []

    for i in range(batch_size):
        rgb_img = np.clip(rgb_np[i], 0, 1)
        hsv_img = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        s_channel = hsv_img[:, :, 1] / 255.0
        v_channel = hsv_img[:, :, 2] / 255.0
        sv_img = np.stack([s_channel, v_channel], axis=0)
        sv_list.append(sv_img)

    sv_tensor = torch.from_numpy(np.stack(sv_list, axis=0)).float()
    return sv_tensor.to(rgb_tensor.device)


def check_folder_structure(rgb_root_dir, depth_root_dir):
    """Validate the RGB and depth folder structures."""
    logger.info("=" * 70)
    logger.info("Checking folder structure...")
    logger.info("=" * 70)

    class_to_idx = {'0_real': 0, '1_fake': 1}
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    def find_matching_depth_file(depth_dir, rgb_filename):
        base_filename = os.path.splitext(rgb_filename)[0]
        possible_depth_paths = [
            os.path.join(depth_dir, rgb_filename),
            os.path.join(depth_dir, base_filename + '.png'),
            os.path.join(depth_dir, base_filename + '.jpg'),
            os.path.join(depth_dir, base_filename + '_depth.png'),
        ]
        for possible_path in possible_depth_paths:
            if os.path.exists(possible_path):
                return possible_path
        return None

    rgb_class_folders = []
    logger.info(f"📁 Scanning RGB root: {rgb_root_dir}")
    for dirpath, dirnames, _ in os.walk(rgb_root_dir):
        for dirname in dirnames:
            if dirname in class_to_idx:
                full_path = os.path.join(dirpath, dirname)
                relative_path = os.path.relpath(full_path, rgb_root_dir)
                rgb_class_folders.append((full_path, dirname, relative_path))
                logger.info(f"  ✅ Found RGB class folder: {relative_path}")

    depth_class_folders = []
    logger.info(f"\n📁 Scanning depth root: {depth_root_dir}")
    for dirpath, dirnames, _ in os.walk(depth_root_dir):
        for dirname in dirnames:
            if dirname in class_to_idx:
                full_path = os.path.join(dirpath, dirname)
                relative_path = os.path.relpath(full_path, depth_root_dir)
                depth_class_folders.append((full_path, dirname, relative_path))
                logger.info(f"  ✅ Found depth class folder: {relative_path}")

    logger.info(f"\n📊 Statistics:")
    logger.info(f"  Total RGB class folders: {len(rgb_class_folders)}")
    logger.info(f"  Total depth class folders: {len(depth_class_folders)}")

    matched_pairs = []
    total_pairs = 0

    for rgb_path, class_name, rgb_relative in rgb_class_folders:
        corresponding_depth_path = os.path.join(depth_root_dir, rgb_relative)
        if os.path.exists(corresponding_depth_path):
            matched_pairs.append((rgb_path, corresponding_depth_path, class_name, rgb_relative))

            rgb_files = [f for f in os.listdir(rgb_path)
                         if os.path.splitext(f)[1].lower() in valid_extensions]
            total_pairs += len(rgb_files)

    logger.info(f"\n📋 Final counts:")
    logger.info(f"  Total matched folder pairs: {len(matched_pairs)}")
    logger.info(f"  Estimated image pairs: {total_pairs}")

    if total_pairs == 0:
        logger.error("❌ No valid image pairs found!")
        return False
    else:
        logger.info("✅ Folder structure check passed!")
        return True
