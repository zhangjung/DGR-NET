import os
from PIL import Image, ExifTags
import torch
from tqdm import tqdm
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
import numpy as np
# Basic device selection. This script requires Python 3.9 or later, the latest
# Transformers release, and access to Hugging Face to download the DepthPro model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Estimate depth maps with the DepthPro model.
# Path configuration.
# input_folder = r"F:\605depthpro\real"
# output_folder = r"F:\605depthpro\output_real"
input_folder = r"F:\ZJ\CSAIID\SemCross"
output_folder = r"F:\ZJ\CSAIID\SemCross_depth"
os.makedirs(output_folder, exist_ok=True)

# Initialize the model.
image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", do_convert_rgb=True)
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
# image_processor = DepthProImageProcessorFast.from_pretrained("depthpro/", do_convert_rgb=True)
# model = DepthProForDepthEstimation.from_pretrained("depthpro/").to(device)
# image_processor = DepthProImageProcessorFast.from_pretrained("./DepthPro", do_convert_rgb=True)
# model = DepthProForDepthEstimation.from_pretrained("./DepthPro").to(device)
def correct_orientation(image):
    """Rotate an image automatically according to its EXIF orientation tag."""
    try:
        # Read the EXIF orientation tag (ID 274).
        exif = image._getexif()
        if exif:
            orientation = exif.get(274)  # 274 is the EXIF Orientation tag.
            # Apply the required rotation.
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
    """Process one image and save its estimated depth map."""
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
        # Apply logarithmic normalization.
        depth = np.log1p(depth.detach().cpu().numpy())  # Avoid numerical issues at zero.
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255

        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth = depth.astype("uint8")

        # Create the output directory.
        output_subdir = os.path.join(output_folder, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)

        # Save the depth map.
        output_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_depth.png"
        output_path = os.path.join(output_subdir, output_name)
        Image.fromarray(depth).save(output_path, format="PNG")
        return True
    except Exception as e:
        print(f"\nFailed to process: {img_path} | Error: {str(e)}")
        return False


# Collect all image paths to process.
supported_exts = ('.tif', '.jpg', '.jpeg', '.png')
file_list = []
for root, dirs, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith(supported_exts):
            input_path = os.path.join(root, f)
            rel_path = os.path.relpath(input_path, input_folder)
            file_list.append((input_path, rel_path))

# Process all images with a progress bar.
total_files = len(file_list)
with tqdm(total=total_files, desc="Generating depth maps", unit="image", dynamic_ncols=True) as pbar:
    for input_path, rel_path in file_list:
        success = process_image(input_path, rel_path)
        pbar.set_postfix_str(f"Current file: {os.path.basename(input_path)}")
        pbar.update(1)
        if success:
            pbar.set_postfix_str(f"Completed: {os.path.basename(input_path)}", refresh=False)
