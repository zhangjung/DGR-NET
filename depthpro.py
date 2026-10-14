"""批量遍历退化条件的 rgb 文件夹，模型只加载一次。"""
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

# 填包含各退化条件的总目录；也可以填写多个数据集总目录。
ROOT_FOLDERS = [
    r"F:\ZJ\JPEG_CSAIID",
    # r"F:\ZJ\JPEG_GenImage",
]
MODEL_DIR = r"C:\Users\tdt\.cache\huggingface\hub\models--apple--DepthPro-hf\snapshots\de816c8ce7168afcb231f96d501d72b869d0beda"
OUTPUT_FOLDER_NAME = "rgb_depth"
INCLUDE_CLEAN = True  # False: 不处理 clean/rgb
SKIP_EXISTING = True  # 跳过可读取且尺寸匹配的旧灰度图，须确认来自相同模型及RGB
# False 会重新生成、替换对应输出。原始RGB不会修改。
EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def correct_orientation(image):
    """保留原代码 EXIF 3/6/8 的旋转处理。"""
    try:
        orientation = image.getexif().get(274)
        angle = {3: 180, 6: 270, 8: 90}.get(orientation)
        if angle is not None:
            image = image.rotate(angle, expand=True)
    except (AttributeError, KeyError):
        pass
    return image


def discover_rgb_folders(roots, include_clean=True):
    found = set()
    for value in roots:
        root = Path(value).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"总目录不存在：{root}")
        for directory, dirs, _ in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d.lower() not in
                             {OUTPUT_FOLDER_NAME.lower(), "depth", "_depthpro_logs"}
                             and not (Path(directory) / d).is_symlink())
            current = Path(directory)
            if current.name.lower() == "rgb":
                if include_clean or current.parent.name.lower() != "clean":
                    found.add(current.resolve())
                dirs[:] = []
    return sorted(found, key=str)


def collect_jobs(rgb):
    output = rgb.parent / OUTPUT_FOLDER_NAME
    jobs, targets = [], set()
    for directory, dirs, names in os.walk(rgb, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(directory) / d).is_symlink())
        for name in sorted(names):
            source = Path(directory) / name
            if source.suffix.lower() not in EXTENSIONS or source.is_symlink():
                continue
            relative = source.relative_to(rgb)
            target = output / relative.parent / (source.stem + "_depth.png")
            key = str(target).casefold()
            if key in targets:
                raise ValueError(f"深度输出重名，请检查同目录同主名图片：{target}")
            targets.add(key)
            jobs.append((source, target))
    if not jobs:
        raise ValueError(f"未找到图片：{rgb}")
    return output, jobs


def valid_existing(path, size):
    if not path.is_file():
        return False
    try:
        with Image.open(path) as depth:
            depth.load()
            return depth.format == "PNG" and depth.mode == "L" and depth.size == size
    except (OSError, ValueError):
        return False


def normalize_depth(depth):
    """保留原 log1p -> 逐图 min-max -> uint8，仅增加异常检查。"""
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("深度必须是二维非负有限数值")
    depth = np.log1p(depth)
    span = depth.max() - depth.min()
    if not np.isfinite(span) or span <= 0:
        raise ValueError("深度为常量或范围无效，无法归一化")
    return ((depth - depth.min()) / span * 255).astype("uint8")


def process_image(source, target, processor, model, device, torch):
    with Image.open(source) as original:
        image = correct_orientation(original).convert("RGB")
        if SKIP_EXISTING and valid_existing(target, image.size):
            return "skipped"
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        processed = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)])
        depth = normalize_depth(processed[0]["predicted_depth"].detach().cpu().numpy())
        if depth.shape != (image.height, image.width):
            raise ValueError("深度图和RGB尺寸不一致")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".png.tmp")
    Image.fromarray(depth).save(temporary, format="PNG")
    temporary.replace(target)
    return "generated"


def save_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    if not ROOT_FOLDERS:
        raise ValueError("请填写 ROOT_FOLDERS")
    if OUTPUT_FOLDER_NAME.lower() in {"rgb", "", ".", ".."} or any(c in OUTPUT_FOLDER_NAME for c in "/\\:"):
        raise ValueError("输出目录名必须为不同于rgb的单个目录名")
    folders = discover_rgb_folders(ROOT_FOLDERS, INCLUDE_CLEAN)
    if not folders:
        raise ValueError("未找到名为 rgb 的文件夹，请检查总目录")
    tasks = [(rgb, *collect_jobs(rgb)) for rgb in folders]
    for rgb, output, jobs in tasks:
        print(f"{rgb} -> {output} （{len(jobs)} 张）", flush=True)
    for name in ("config.json", "preprocessor_config.json"):
        if not (Path(MODEL_DIR) / name).is_file():
            raise FileNotFoundError(f"本地模型配置不存在：{Path(MODEL_DIR) / name}")

    import torch
    from tqdm import tqdm
    from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"本地加载模型（仅一次），设备：{device}", flush=True)
    processor = DepthProImageProcessorFast.from_pretrained(
        MODEL_DIR, do_convert_rgb=True, local_files_only=True)
    model = DepthProForDepthEstimation.from_pretrained(
        MODEL_DIR, local_files_only=True).to(device)
    model.eval()
    log_folder = Path(ROOT_FOLDERS[0]).resolve() / "_depthpro_logs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_folder.mkdir(parents=True, exist_ok=False)
    summary = dict(model_dir=MODEL_DIR, device=str(device), skip_existing=SKIP_EXISTING,
                   include_clean=INCLUDE_CLEAN, status="running", folders=[],
                   normalization="log1p -> per-image minmax -> uint8")
    save_json(log_folder / "summary.json", summary)
    with (log_folder / "images.jsonl").open("w", encoding="utf-8") as records:
        for rgb, output, jobs in tasks:
            counts = dict(rgb=str(rgb), depth=str(output), total=len(jobs),
                          generated=0, skipped=0, failed=0)
            summary["folders"].append(counts)
            with tqdm(total=len(jobs), desc=f"{rgb.parent.name} 深度", unit="张", dynamic_ncols=True) as bar:
                for source, target in jobs:
                    error = None
                    try:
                        status = process_image(source, target, processor, model, device, torch)
                    except Exception as exception:
                        status, error = "failed", str(exception)
                        tqdm.write(f"处理失败：{source} | {error}")
                    counts[status] += 1
                    records.write(json.dumps(dict(rgb=str(source), depth=str(target),
                                                   status=status, error=error), ensure_ascii=False) + "\n")
                    records.flush()
                    bar.update(1)
                    bar.set_postfix(生成=counts["generated"], 跳过=counts["skipped"], 失败=counts["failed"])
            save_json(log_folder / "summary.json", summary)
    failed = sum(row["failed"] for row in summary["folders"])
    summary["status"] = "completed" if failed == 0 else "completed_with_errors"
    save_json(log_folder / "summary.json", summary)
    print(f"处理结束，失败 {failed} 张。日志：{log_folder}", flush=True)
    if failed:
        raise SystemExit("存在失败图片，请查看日志、修复后重跑；不要静默丢弃失败样本。")


if __name__ == "__main__":
    main()
