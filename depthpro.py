import os
from PIL import Image, ExifTags
import torch
from tqdm import tqdm
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import numpy as np
# 设备配置最原始的逻辑，但是速度太慢，这个代码要求python>=3.9,transformers要是最新版，且能访问huggingface才可以，满足以上条件，就可以从huggingface上下载depthpro模型进行深度估计
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 深度图估计的代码，使用depthpro模型
# 路径配置
# input_folder = r"F:\605depthpro\real"
# output_folder = r"F:\605depthpro\output_real"
input_folder = r"F:\ZJ\CSAIID\SemCross"
output_folder = r"F:\ZJ\CSAIID\SemCross_depth"
os.makedirs(output_folder, exist_ok=True)

# 初始化模型
image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", do_convert_rgb=True)
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
# image_processor = DepthProImageProcessorFast.from_pretrained("depthpro/", do_convert_rgb=True)
# model = DepthProForDepthEstimation.from_pretrained("depthpro/").to(device)
# image_processor = DepthProImageProcessorFast.from_pretrained("./DepthPro", do_convert_rgb=True)
# model = DepthProForDepthEstimation.from_pretrained("./DepthPro").to(device)
def correct_orientation(image):
    """根据EXIF方向标签自动旋转图像"""
    try:
        # 获取EXIF方向标签（编号274）
        exif = image._getexif()
        if exif:
            orientation = exif.get(274)  # 274对应Orientation标签[4,6](@ref)
            # 执行旋转/翻转操作
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError):
        pass
    return image


def process_image(img_path, rel_path):
    """处理单张图片并保存深度图（含进度条更新）"""
    try:
        image = Image.open(img_path)
        image = correct_orientation(image)
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        post_processed = image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        depth = post_processed[0]["predicted_depth"]
        # 对数归一化
        depth = np.log1p(depth.detach().cpu().numpy())  # 避免零值溢出
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255

        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype("uint8")

        # 创建输出目录
        output_subdir = os.path.join(output_folder, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)

        # 保存深度图
        output_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_depth.png"
        output_path = os.path.join(output_subdir, output_name)
        Image.fromarray(depth).save(output_path, format="PNG")
        return True
    except Exception as e:
        print(f"\n处理失败：{img_path} | 错误：{str(e)}")
        return False


# 收集所有待处理文件路径
supported_exts = ('.tif', '.jpg', '.jpeg', '.png')
file_list = []
for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith(supported_exts):
            input_path = os.path.join(root, f)
            rel_path = os.path.relpath(input_path, input_folder)
            file_list.append((input_path, rel_path))

# 带进度条的批量处理
total_files = len(file_list)
with tqdm(total=total_files, desc="生成深度图", unit="张", dynamic_ncols=True) as pbar:
    for input_path, rel_path in file_list:
        success = process_image(input_path, rel_path)
        pbar.set_postfix_str(f"当前文件：{os.path.basename(input_path)}")
        pbar.update(1)
        if success:
            pbar.set_postfix_str(f"已完成：{os.path.basename(input_path)}", refresh=False)