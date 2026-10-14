"""Dataset and paired data augmentation for RGB + depth image pairs."""
import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from utils import logger


def load_image_and_depth_fast(image_info):
    """Efficiently load an RGB image and its corresponding depth map."""
    rgb_path, depth_path, label, transform = image_info

    try:
        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')

        base_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        rgb_tensor = base_transform(rgb_image)
        depth_tensor = base_transform(depth_image)

        if transform:
            try:
                rgb_tensor, depth_tensor = transform(rgb_tensor, depth_tensor)
            except TypeError:
                # Fallback for legacy RGB-only transforms
                rgb_tensor = transform(rgb_tensor)

        return rgb_tensor, depth_tensor, torch.tensor(label, dtype=torch.float32)

    except Exception as e:
        logger.error(f"Error loading images: {rgb_path}, {depth_path}, error={e}")
        default_rgb = torch.zeros(3, 224, 224)
        default_depth = torch.zeros(1, 224, 224)
        return default_rgb, default_depth, torch.tensor(label, dtype=torch.float32)


class HighPerformanceRGBDepthDataset(Dataset):
    """High-performance RGB + depth dataset."""

    def __init__(self, rgb_root_dir, depth_root_dir, transform=None):
        self.rgb_root_dir = rgb_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'0_real': 0, '1_fake': 1}

        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

        logger.info(f"Recursively scanning RGB folder: {rgb_root_dir}")
        logger.info(f"Recursively scanning depth folder: {depth_root_dir}")

        rgb_class_folders = self._find_class_folders(rgb_root_dir)
        depth_class_folders = self._find_class_folders(depth_root_dir)

        logger.info(f"Found RGB class folders: {len(rgb_class_folders)}")
        logger.info(f"Found depth class folders: {len(depth_class_folders)}")

        matched_count = 0
        unmatched_count = 0

        for rgb_class_path, class_name in rgb_class_folders:
            label = self.class_to_idx[class_name]
            relative_path = os.path.relpath(rgb_class_path, rgb_root_dir)
            corresponding_depth_path = os.path.join(depth_root_dir, relative_path)

            if not os.path.exists(corresponding_depth_path):
                logger.warning(f"Corresponding depth folder does not exist: {corresponding_depth_path}")
                continue

            if os.path.basename(corresponding_depth_path) != class_name:
                logger.warning(f"Depth folder class name mismatch: {corresponding_depth_path}")
                continue

            logger.info(f"Processing class folder: {relative_path} (label: {label})")

            rgb_files = []
            for filename in os.listdir(rgb_class_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in valid_extensions:
                    rgb_files.append(filename)
            rgb_files.sort()

            depth_files = []
            for filename in os.listdir(corresponding_depth_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in valid_extensions:
                    depth_files.append(filename)
            depth_files.sort()

            for filename in rgb_files:
                rgb_path = os.path.join(rgb_class_path, filename)
                depth_path = self._find_matching_depth_file(corresponding_depth_path, filename, valid_extensions)

                if depth_path:
                    self.samples.append((rgb_path, depth_path, label))
                    matched_count += 1
                else:
                    unmatched_count += 1

            if unmatched_count > 0 and len(rgb_files) == len(depth_files):
                logger.info(f"  In {relative_path}, attempting index-based pairing for {len(rgb_files)} files")
                for i, rgb_filename in enumerate(rgb_files):
                    rgb_path = os.path.join(rgb_class_path, rgb_filename)
                    depth_filename = depth_files[i]
                    depth_path = os.path.join(corresponding_depth_path, depth_filename)

                    if not any(sample[0] == rgb_path for sample in self.samples):
                        self.samples.append((rgb_path, depth_path, label))
                        matched_count += 1
                        unmatched_count -= 1

            if unmatched_count > 0:
                logger.warning(f"  In {relative_path}, {unmatched_count} RGB files have no matching depth map")

        if unmatched_count > 5:
            logger.warning(f"A total of {unmatched_count} RGB images have no matching depth map")

        if not self.samples:
            raise RuntimeError(f"No matching image pairs found under '{rgb_root_dir}' and '{depth_root_dir}'.")

        logger.info(f"Successfully matched {matched_count} RGB-depth image pairs")

        class_counts = {}
        for _, _, label in self.samples:
            class_name = list(self.class_to_idx.keys())[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name}: {count} pairs")

    def _find_class_folders(self, root_dir):
        """Recursively find all 0_real and 1_fake subfolders."""
        class_folders = []
        for dirpath, dirnames, _ in os.walk(root_dir):
            for dirname in dirnames:
                if dirname in self.class_to_idx:
                    full_path = os.path.join(dirpath, dirname)
                    class_folders.append((full_path, dirname))
        return class_folders

    def _find_matching_depth_file(self, depth_dir, rgb_filename, valid_extensions):
        """Find the depth file matching a given RGB filename."""
        base_filename = os.path.splitext(rgb_filename)[0]

        possible_depth_paths = [
            os.path.join(depth_dir, rgb_filename),
            os.path.join(depth_dir, base_filename + '.png'),
            os.path.join(depth_dir, base_filename + '.jpg'),
            os.path.join(depth_dir, base_filename + '.jpeg'),
            os.path.join(depth_dir, base_filename + '_depth.png'),
            os.path.join(depth_dir, base_filename + '_depth.jpg'),
            os.path.join(depth_dir, base_filename + '_depth.jpeg'),
            os.path.join(depth_dir, base_filename + '.tiff'),
            os.path.join(depth_dir, base_filename + '.tif'),
            os.path.join(depth_dir, base_filename + '_d.png'),
            os.path.join(depth_dir, base_filename + '_d.jpg'),
            os.path.join(depth_dir, 'depth_' + base_filename + '.png'),
            os.path.join(depth_dir, 'depth_' + base_filename + '.jpg'),
        ]

        for possible_path in possible_depth_paths:
            if os.path.exists(possible_path):
                return possible_path

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, depth_path, label = self.samples[idx]
        return load_image_and_depth_fast((rgb_path, depth_path, label, self.transform))


class PairedAugment:
    """
    Paired synchronized geometric augmentation + paired erasing (RGB/depth at the same locations).
    Color jitter is applied to RGB only (the depth map is not affected); no blur or sharpening.
    Inputs and outputs are both [0, 1] tensors:
      - Inputs: rgb_tensor01 [3, H, W], depth_tensor01 [1, H, W] or [3, H, W]
      - Outputs: rgb_out [3, 224, 224], depth_out [1/3, 224, 224]
    """

    def __init__(self, is_training=True,
                 out_size=224,
                 scale=(0.85, 1.0),
                 ratio=(0.9, 1.1),
                 max_rotate=10,
                 # RGB color augmentation parameters
                 brightness=0.15,
                 contrast=0.15,
                 saturation=0.15,
                 hue=0.05,
                 # Erasing parameters
                 erase_p=0.10,
                 erase_scale=(0.02, 0.1),
                 erase_ratio=(0.3, 3.3),
                 erase_value_rgb=0.0,
                 erase_value_depth=0.0):
        self.is_training = is_training
        self.out_size = out_size
        self.scale = scale
        self.ratio = ratio
        self.max_rotate = max_rotate
        self.erase_p = erase_p
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase_value_rgb = erase_value_rgb
        self.erase_value_depth = erase_value_depth

        # Color augmentation parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _sample_geom(self, H, W):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            torch.zeros(3, H, W), scale=self.scale, ratio=self.ratio
        )
        do_flip = random.random() < 0.5
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        return (i, j, h, w), do_flip, angle

    def _apply_geom(self, img, crop_box, do_flip, angle, is_depth=False):
        interp = transforms.InterpolationMode.NEAREST if is_depth else transforms.InterpolationMode.BILINEAR
        img = transforms.functional.resized_crop(img, *crop_box, size=[self.out_size, self.out_size],
                                                 interpolation=interp)
        if do_flip:
            img = transforms.functional.hflip(img)
        img = transforms.functional.rotate(img, angle, interpolation=interp, fill=0)
        return img

    def _paired_erasing(self, rgb, depth):
        if self.erase_p <= 0 or random.random() >= self.erase_p:
            return rgb, depth
        _, H, W = rgb.shape
        # Sample area
        area = H * W
        for _ in range(10):
            target_area = random.uniform(*self.erase_scale) * area
            aspect_ratio = random.uniform(*self.erase_ratio)
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            if h < H and w < W and h > 0 and w > 0:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                rgb[:, top:top + h, left:left + w] = self.erase_value_rgb
                depth[:, top:top + h, left:left + w] = self.erase_value_depth
                break
        return rgb, depth

    def _apply_color_jitter(self, rgb):
        """
        Apply color augmentation to RGB only; the depth map is not affected.
        """
        if random.random() < 0.8:  # 80% chance to adjust brightness
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            rgb = transforms.functional.adjust_brightness(rgb, factor)

        if random.random() < 0.8:  # 80% chance to adjust contrast
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            rgb = transforms.functional.adjust_contrast(rgb, factor)

        if random.random() < 0.8:  # 80% chance to adjust saturation
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            rgb = transforms.functional.adjust_saturation(rgb, factor)

        if random.random() < 0.8:  # 80% chance to adjust hue
            factor = random.uniform(-self.hue, self.hue)
            rgb = transforms.functional.adjust_hue(rgb, factor)

        return rgb

    def __call__(self, rgb_tensor01, depth_tensor01):
        if not self.is_training:
            # Size alignment only (safety fallback; usually already 224 upstream)
            _, H, W = rgb_tensor01.shape
            crop_box = (0, 0, H, W)
            do_flip = False
            angle = 0.0
            rgb = self._apply_geom(rgb_tensor01, crop_box, do_flip, angle, is_depth=False)
            depth = self._apply_geom(depth_tensor01, crop_box, do_flip, angle, is_depth=True)
            return rgb, depth

        _, H, W = rgb_tensor01.shape
        crop_box, do_flip, angle = self._sample_geom(H, W)
        rgb_geo = self._apply_geom(rgb_tensor01, crop_box, do_flip, angle, is_depth=False)
        depth_geo = self._apply_geom(depth_tensor01, crop_box, do_flip, angle, is_depth=True)

        # Color augmentation (RGB only)
        rgb_geo = self._apply_color_jitter(rgb_geo)

        # Paired erasing
        rgb_geo, depth_geo = self._paired_erasing(rgb_geo, depth_geo)
        return rgb_geo, depth_geo


def get_paired_transforms(is_training=True):
    """Return paired synchronized augmentation (geometric) + paired erasing + RGB color jitter."""
    return PairedAugment(is_training=is_training)
