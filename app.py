# app.py
from flask import Flask, render_template, request, redirect, url_for, send_file, url_for, session, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import os

from mne.viz import plot_epochs_image
from pyasn1_modules.rfc7030 import aa_asymmDecryptKeyID
from unicodedata import category
from werkzeug.utils import secure_filename
import cv2

from diffusers import DiffusionPipeline
# model_id = "yahoo-inc/photo-background-generation"
# pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
# pipeline = pipeline.to('cuda')
import torch
from PIL import Image, ImageOps, ImageFont, ImageDraw
import requests
from io import BytesIO
from transparent_background import Remover
# import jsonify
import uuid
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

import base64
from flask import request, send_file
from rembg import remove

# from PIL import Image
# import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from model_inference import predict_masks, color_palette



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
UPLOAD_FOLDER = 'static/edited'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

remover = Remover(mode='base')


# Home page
# @app.route('/')
# def home():
#     return render_template('home.html')

# Editor with AI
@app.route('/editor-ai', methods=['GET', 'POST'])
def editor_ai():
    image_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_path = filepath

            if 'remove_bg' in request.form:
                img = cv2.imread(filepath)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                result = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(filepath, result)

    return render_template('editor_ai.html', image_path=image_path)

# Basic Editor
@app.route('/editor-basic', methods=['GET', 'POST'])
def editor_basic():
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        edit_type = request.form['edit_type']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Đọc và xử lý ảnh bằng OpenCV
            img_np = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if edit_type == 'gray':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif edit_type == 'invert':
                img = cv2.bitwise_not(img)
            elif edit_type == 'blur':
                img = cv2.GaussianBlur(img, (15, 15), 0)

            # Lưu lại ảnh đã chỉnh sửa
            if edit_type == 'gray':
                cv2.imwrite(filepath, img)
            else:
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            return render_template('editor_basic.html', filename=filename)

    return render_template('editor_basic.html', filename=filename)


# Đặt đường dẫn lưu ảnh đã upload
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        filename = app.config['UPLOAD_FOLDER'] + '/' +file.filename
        file.save(filename)
        print(f"File will be saved to: {filename}")  # In ra đường dẫn nơi ảnh được lưu
        # Sau khi upload, chuyển hướng tới trang chỉnh sửa và truyền đường dẫn hình ảnh
        return redirect(url_for('editor', image_path=filename))


@app.route('/edit_image', methods=['POST', 'GET'])
# @app.route('/editor', methods=['POST', 'GET'])
def editor():
    if request.method == 'POST':
        image_path = request.args.get('image_path', None)
        print(image_path)
        # Đường dẫn thực tế đến thư mục chứa sticker
        sticker_folder = os.path.join(app.static_folder, 'stickers')

        # Lấy danh sách các file sticker (lọc đuôi .png, .jpg, .webp, .svg, ...)
        stickers = [
            'stickers/' + filename
            for filename in os.listdir(sticker_folder)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg'))
        ]

        return render_template('editor.html', stickers=stickers, image_path=image_path)
    else:
        image_path = request.args.get('image_path', None)
        sticker_folder = os.path.join(app.static_folder, 'stickers')

        # Lấy danh sách các file sticker (lọc đuôi .png, .jpg, .webp, .svg, ...)
        stickers = [
            'stickers/' + filename
            for filename in os.listdir(sticker_folder)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg'))
        ]
        return render_template('editor.html', stickers=stickers, image_path=image_path)


def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)



# @app.route('/ai_editor')
#
# def ai_editor():
#     image_path = request.args.get('image_path', None)
#     sticker_folder = os.path.join(app.static_folder, 'stickers')
#
#     # Lấy danh sách các file sticker (lọc đuôi .png, .jpg, .webp, .svg, ...)
#     stickers = [
#         'stickers/' + filename
#         for filename in os.listdir(sticker_folder)
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg'))
#     ]
#     return render_template('ai_editor.html', stickers=stickers,  image_path=image_path)

# from flask import Flask, request, render_template, redirect, url_for
# import os
# from werkzeug.utils import secure_filename
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/ai_editor')
def ai_editor():
    image_path = request.args.get('image_path', None)  # chỉ cần lấy từ URL
    print(f"[ai_editor] Image path: {image_path}")

    # Load sticker nếu cần
    sticker_folder = os.path.join(app.static_folder, 'stickers')
    stickers = [
        'stickers/' + filename
        for filename in os.listdir(sticker_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.svg'))
    ]
    image_path = request.args.get('image_path', None)
    print(image_path)
    # return render_template('ai_editor.html', image_path=image_path)

    return render_template('ai_editor.html', image_path=image_path, stickers=stickers)

@app.route("/remove_background", methods=["POST"])
def remove_background():
    
    # from transparent_background import Remover

    # Nhận ảnh base64 từ frontend
    data = request.get_json()
    img_data = data['image_data']
    header, encoded = img_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)


    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    output = remove(img)
    # return output

    # Trả ảnh về
    buf = BytesIO()
    output.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

from PIL import Image, ImageOps
# import seed
def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

@app.route('/generate_background', methods=['POST'])

def generate_background():
    model_id = "yahoo-inc/photo-background-generation"
    pipe = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id).to("cuda")
    try:
        prompt = request.form.get("prompt")
        file = request.files.get("foreground")
        print(f"[generate_background] Generating image for prompt: {prompt}")
        print(f"[generate_background] File image for prompt: {file}")
        

        if not prompt or not file:
            return {"error": "Missing prompt or foreground"}, 400

        # Load foreground image (after remove background)

        foreground = Image.open(file).convert("RGBA")
        foreground1 = foreground.copy()
        width, height = foreground1.size
        # foreground = resize_with_padding(foreground, (512, 512))
        r, g, b, a = foreground.split()
        img = Image.merge("RGB", (r, g, b))
        # mask = a
        inverted_rgb = ImageOps.invert(img)
        foreground = Image.merge("RGBA", (*inverted_rgb.split(), a))
                # im1 = mask.save("geeks.jpg")
        seed = 13
# a
        generator = torch.Generator(device='cuda').manual_seed(seed)
        # foreground = Image.open(file).convert("RGBA")
        print('Size of this is: ', foreground.size)
        
        mask = foreground
        # Tạo background bằng AI
        bg_width, bg_height = foreground.size
        # im1 = foreground.save("geeks.jpg")
        cond_scale = 1.0
        print(f"Generating background for prompt '{prompt}' with size {bg_width}x{bg_height}")
        # background = pipe(prompt, height=bg_height, width=bg_width).images[0].convert("RGBA")
        with torch.autocast("cuda"):
            background = pipe(
                prompt=prompt,
                image=img,
                mask_image=mask,
                control_image=mask,
                num_images_per_prompt=1,
                generator=generator,
                num_inference_steps=20,
                guess_mode=False,
                width=width,
                height=height,
                controlnet_conditioning_scale=cond_scale
            ).images[0].convert("RGBA")

            #     # --- Thay đổi màu background, ví dụ overlay xanh lam nhẹ ---
        # Đảm bảo foreground cùng kích thước background để alpha_composite hoạt động tốt
        bg_width, bg_height = background.size
        foreground_resized = foreground1.resize(background.size, Image.LANCZOS)
        # foreground_resized = foreground1.copy()
        # foreground_resized.thumbnail((bg_width, bg_height), Image.LANCZOS)

# bbb

        print(foreground_resized.size)

        # Ghép ảnh foreground lên background, giữ nguyên màu và alpha của foreground
        combined = Image.alpha_composite(background, foreground_resized)

        # Nếu bạn muốn trả về kích thước gốc của foreground (không bắt buộc)
        # combined = combined.resize(foreground1.size, Image.LANCZOS)


        # Lưu hoặc trả về
        buf = BytesIO()
        combined.save(buf, format='PNG')
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        print("[generate_background] Error:", e)
        return {"error": str(e)}, 500
def random_color(alpha=0.5):
    return (random.random(), random.random(), random.random(), alpha)


import torch
import torch.nn as nn
import torchvision


def load_model_network():
    model = torchvision.models.efficientnet_b3(pretrained=False, progress=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.25, inplace=True),
        nn.Linear(in_features=1536, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_features=512, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_features=512, out_features=101, bias=True),
    )
    return model
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# sam.to(device)
# mask_generator = SamAutomaticMaskGenerator(sam)

def visualize_panoptic_segmentation(
    original_image_np, segmentation_mask, segments_info, category_names
):
    """
    Visualizes the segmentation mask overlaid on the original image with category labels.

    Args:
        original_image_np (np.ndarray): The original image in NumPy array format.
        segmentation_mask (np.ndarray): The segmentation mask.
        segments_info (list): Information about the segments.
        category_names (list): List of category names corresponding to segment IDs.

    Returns:
        PIL.Image.Image: The overlayed image with segmentation mask and labels.
    """
    # Create a blank image for the segmentation mask
    segmentation_image = np.zeros_like(original_image_np)

    num_classes = len(category_names)
    palette = color_palette(num_classes)

    # Apply colors to the segmentation mask
    for segment in segments_info:
        if segment["label_id"] == 0:
            continue
        color = palette[segment["label_id"]]
        mask = segmentation_mask == segment["id"]
        segmentation_image[mask] = color

    # Overlay the segmentation mask on the original image
    alpha = 0.2  # Transparency for the overlay
    overlay_image = cv2.addWeighted(
        original_image_np, 1 - alpha, segmentation_image, alpha, 0
    )

    # Convert to PIL image for text drawing
    overlay_image_pil = Image.fromarray(overlay_image)
    draw = ImageDraw.Draw(overlay_image_pil)
    # Set up font size
    base_font_size = max(
        20, int(min(original_image_np.shape[0], original_image_np.shape[1]) * 0.015)
    )

    # Optional: Load custom font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", base_font_size)
        # font = ImageFont.load_default()
    except IOError:
        raise RuntimeError(
            "Custom font not found. Please ensure the font file is available."
        )

    # Draw category labels on the image
    for segment in segments_info:
        label_id = segment.get("label_id")
        if label_id is not None and 0 <= label_id < len(category_names):
            category = category_names[label_id]
            mask = (segmentation_mask == segment["id"]).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            # if num_labels > 1:
            #     largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            #     centroid_x = int(centroids[largest_component][0])
            #     centroid_y = int(centroids[largest_component][1])
            #
            #     # Ensure text is within image bounds
            #     text_position = (
            #         max(0, min(centroid_x, original_image_np.shape[1] - 1)),
            #         max(0, min(centroid_y, original_image_np.shape[0] - 1)),
            #     )
            #     draw.text(text_position, category, fill=(0, 0, 0), font=font)
            if num_labels > 1:
                for i in range(1, num_labels):  # Bỏ background (label 0)
                    centroid_x = int(centroids[i][0])
                    centroid_y = int(centroids[i][1])

                    # Ensure text is within image bounds
                    text_position = (
                        max(0, min(centroid_x, original_image_np.shape[1] - 1)),
                        max(0, min(centroid_y, original_image_np.shape[0] - 1)),
                    )
                    draw.text(text_position, category, fill=(0, 0, 0), font=font)

    return overlay_image_pil


# def encode_mask_image(mask_array, color,category):
#     """Chuyển mask numpy array (bool hoặc 0-1) thành ảnh PNG base64 với alpha"""
#     # mask_array dạng bool, chuyển thành RGBA (ví dụ màu xanh + alpha)
#
#     # color = palette[mask_array]
#     # mask = segmentation_mask == segment["id"]
#     # segmentation_image[mask_array] = color
#     h, w = mask_array.shape
#     rgba = np.zeros((h, w, 4), dtype=np.uint8)
#     rgba[..., 0] = color[0]      # R
#     rgba[..., 1] = color[1]      # G
#     rgba[..., 2] = color[2]        # B
#     # alpha_value = 0.3
#     rgba[..., 3] = (mask_array * 255).astype(np.uint8)  # alpha theo mask
#
#     base_font_size = max(
#         20, int(min(w, h) * 0.015)
#     )
#     print(base_font_size)
#
#
#     # print(font)
#
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         mask_array, connectivity=4
#     )
#     pil_img = Image.fromarray(rgba, mode='RGBA')
#     # print(pil_img.size)
#     draw = ImageDraw.Draw(pil_img)
#     font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=20)
#     # if num_labels > 1:
#     #     for i in range(1, num_labels):  # Bỏ background (label 0)
#     # # print(centroids)
#     #         centroid_x = int(centroids[i][0])
#     #         centroid_y = int(centroids[i][1])
#     #         print(centroid_x, centroid_y)
#     # # Ensure text is within image bounds
#     #         text_position = (
#     #             max(0, min(centroid_x, w - 1)),
#     #             max(0, min(centroid_y, h - 1)),
#     #         )
#     #         print(text_position)
#     #         print(category)
#     #
#     #         draw.text(text_position, category, fill=(0, 0, 0), font=font)
#     if num_labels > 1:
#         largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         centroid_x = int(centroids[largest_component][0])
#         centroid_y = int(centroids[largest_component][1])
#
#         # Ensure text is within image bounds
#         text_position = (
#             max(0, min(centroid_x, w - 1)),
#             max(0, min(centroid_y, h - 1)),
#         )
#         draw.text(text_position, category, fill=(0, 0, 0), font=font)
#
#     print('Done')
#     # pil_img = Image.fromarray(rgba)
#     buffered = io.BytesIO()
#     pil_img.save(buffered, format="PNG")
#     encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
#     return f"data:image/png;base64,{encoded}"


# def encode_mask_image(mask_array, color,category):
#     """Chuyển mask numpy array (bool hoặc 0-1) thành ảnh PNG base64 với alpha"""
#     # mask_array dạng bool, chuyển thành RGBA (ví dụ màu xanh + alpha)
#
#     # color = palette[mask_array]
#     # mask = segmentation_mask == segment["id"]
#     # segmentation_image[mask_array] = color
#     h, w = mask_array.shape
#     rgba = np.zeros((h, w, 4), dtype=np.uint8)
#     rgba[..., 0] = color[0]      # R
#     rgba[..., 1] = color[1]      # G
#     rgba[..., 2] = color[2]        # B
#     # alpha_value = 0.3
#     rgba[..., 3] = (mask_array * 255).astype(np.uint8)  # alpha theo mask
#
#     base_font_size = max(
#         20, int(min(w, h) * 0.015)
#     )
#     print(base_font_size)
#
#
#     # print(font)
#
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         mask_array, connectivity=4
#     )
#     pil_img = Image.fromarray(rgba, mode='RGBA')
#
#     # print(pil_img.size)
#     draw = ImageDraw.Draw(pil_img)
#     font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=20)
#     # if num_labels > 1:
#     #     for i in range(1, num_labels):  # Bỏ background (label 0)
#     # # print(centroids)
#     #         centroid_x = int(centroids[i][0])
#     #         centroid_y = int(centroids[i][1])
#     #         print(centroid_x, centroid_y)
#     # # Ensure text is within image bounds
#     #         text_position = (
#     #             max(0, min(centroid_x, w - 1)),
#     #             max(0, min(centroid_y, h - 1)),
#     #         )
#     #         print(text_position)
#     #         print(category)
#     #
#     #         draw.text(text_position, category, fill=(0, 0, 0), font=font)
#     if num_labels > 1:
#         largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         centroid_x = int(centroids[largest_component][0])
#         centroid_y = int(centroids[largest_component][1])
#
#         # Ensure text is within image bounds
#         text_position = (
#             max(0, min(centroid_x, w - 1)),
#             max(0, min(centroid_y, h - 1)),
#         )
#         draw.text(text_position, category, fill=(0, 0, 0), font=font)
#
#     print('Done')
#     # pil_img = Image.fromarray(rgba)
#     buffered = io.BytesIO()
#     pil_img.save(buffered, format="PNG")
#     encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return encoded_img



def encode_mask_image(mask_array, color, category):
    """
    Chuyển mask numpy array (bool hoặc 0-1) thành ảnh PNG base64 có alpha,
    đồng thời crop sát vùng mask để bounding box nhỏ lại.
    Vẽ chữ category tại centroid lớn nhất của mask.
    """
    import numpy as np
    import io
    import base64
    from PIL import Image, ImageDraw, ImageFont
    import cv2

    h, w = mask_array.shape

    # Tạo ảnh RGBA từ mask và màu
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (mask_array * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgba, mode='RGBA')

    # Lấy bounding box vùng mask (phần alpha > 0)
    bbox = pil_img.getbbox()
    if bbox is None:
        # Mask rỗng, trả về None hoặc ảnh trong suốt nhỏ
        return None

    # Crop ảnh sát bounding box
    pil_img_cropped = pil_img.crop(bbox)

    # Chuyển ảnh crop về numpy để tìm centroid
    cropped_alpha = np.array(pil_img_cropped)[..., 3]
    cropped_mask = (cropped_alpha > 0).astype(np.uint8)

    # Tìm connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cropped_mask, connectivity=4
    )

    draw = ImageDraw.Draw(pil_img_cropped)

    # Chọn font size phù hợp dựa trên kích thước crop
    font_size = max(12, int(min(pil_img_cropped.width, pil_img_cropped.height) * 0.15))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except:
        font = ImageFont.load_default()

    if num_labels > 1:
        # centroid lớn nhất (bỏ background label 0)
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        centroid_x = int(centroids[largest_component][0])
        centroid_y = int(centroids[largest_component][1])

        # Giới hạn vị trí centroid trong ảnh crop
        centroid_x = max(0, min(centroid_x, pil_img_cropped.width - 1))
        centroid_y = max(0, min(centroid_y, pil_img_cropped.height - 1))

        # Vẽ chữ category
        draw.text((centroid_x, centroid_y), category, fill=(0, 0, 0, 255), font=font)

    # Encode lại thành base64 PNG
    buffered = io.BytesIO()
    pil_img_cropped.save(buffered, format='PNG')
    encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    # return encoded_img
    return  encoded_img, bbox  # (left, upper, right, lower)


# def encode_mask_image(mask_array, color, category):
#     """Chuyển mask numpy array (bool hoặc 0-1) thành ảnh PNG base64 với alpha và crop sát vùng mask"""
#
#     h, w = mask_array.shape
#     rgba = np.zeros((h, w, 4), dtype=np.uint8)
#     rgba[..., 0] = color[0]      # R
#     rgba[..., 1] = color[1]      # G
#     rgba[..., 2] = color[2]      # B
#     rgba[..., 3] = (mask_array * 255).astype(np.uint8)  # alpha theo mask
#
#     # Chuyển sang PIL Image để crop vùng mask sát nhất
#     pil_img = Image.fromarray(rgba, mode='RGBA')
#
#     # Tìm bounding box vùng mask (non-zero alpha)
#     bbox = pil_img.getbbox()
#     if bbox is None:
#         # Mask rỗng
#         return None
#
#     # Crop ảnh theo bounding box
#     pil_img_cropped = pil_img.crop(bbox)
#
#     # Lấy mask crop lại (dùng alpha channel để tìm centroid)
#     cropped_mask = np.array(pil_img_cropped)[..., 3] > 0
#
#     # Tính connected components trên mask crop để lấy centroid lớn nhất
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
#         cropped_mask.astype(np.uint8), connectivity=4
#     )
#
#     draw = ImageDraw.Draw(pil_img_cropped)
#     font_size = max(20, int(min(pil_img_cropped.width, pil_img_cropped.height) * 0.15))
#     font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
#
#     if num_labels > 1:
#         largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         centroid_x = int(centroids[largest_component][0])
#         centroid_y = int(centroids[largest_component][1])
#
#         # Đảm bảo centroid nằm trong crop bounds
#         centroid_x = max(0, min(centroid_x, pil_img_cropped.width - 1))
#         centroid_y = max(0, min(centroid_y, pil_img_cropped.height - 1))
#
#         draw.text((centroid_x, centroid_y), category, fill=(0, 0, 0, 255), font=font)
#
#     buffered = io.BytesIO()
#     pil_img_cropped.save(buffered, format="PNG")
#     encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     return encoded_img




# import torchvision.transforms as T
# @app.route('/segment_image', methods=['POST'])
# def segment_image():
#     try:
#         data = request.get_json()
#         image_data_url = data['image_data']
#         # Remove prefix 'data:image/png;base64,'
#         header, encoded = image_data_url.split(",", 1)
#         image_bytes = base64.b64decode(encoded)
#
#         # Đọc ảnh từ bytes
#         img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         img_pil.save("output_image_1.png")
#
#
#         # img_np = np.array(img_pil)  # RGB
#
#
#         original_image_np, segmentation_mask, segments_info, id2label = predict_masks(img_pil)
#         print('segmentation_mask: ', segmentation_mask.shape)
#         instance_images = []
#         num_classes = len(id2label)
#         palette = color_palette(num_classes)
#
#         for segment in segments_info:
#             if segment["label_id"] == 0:
#                 continue
#             color = palette[segment["label_id"]]
#             mask = segmentation_mask == segment["id"]
#             mask = mask.astype(np.uint8)
#             print(mask.shape)
#             label_id = segment.get("label_id")
#             category = id2label[label_id]
#             print(category)
#
#         # for mask in masks:
#         #     idx +=1
#         #     print(idx)
#         #     # print("Keys:", mask.keys())
#         #     # label_id = mask.get("label_id") or mask.get("class_id") or mask.get("category_id")
#         #     # print('Label class:', label_id)
#         #     # if mask.get("label_id", 0) == 0:
#         #     #     continue  # bỏ class 0 nếu có key 'label_id'
#         #     # print(mask)
#         #     m = mask.astype(np.uint8)  # mask nhị phân 0/1
#         #
#         #     # Tạo ảnh mask RGBA base64
#             encoded_mask = encode_mask_image(mask, color, category)
#             instance_images.append(encoded_mask)
#
#         return jsonify({'instances': instance_images})
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

def extract_object_with_mask(image, mask_array):
    """
    Cắt vùng object từ ảnh gốc theo mask, áp dụng alpha để xóa nền.
    Trả về ảnh PNG base64 có thể hiển thị và di chuyển độc lập.
    """
    image = np.array(image)
    h, w = mask_array.shape
    # bbox = image.getbbox()
    y_indices, x_indices = np.where(mask_array == 1)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None, None, None

    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)

    # Crop ảnh gốc và mask tương ứng
    cropped_img = image[min_y:max_y+1, min_x:max_x+1]
    cropped_mask = mask_array[min_y:max_y+1, min_x:max_x+1]
    # print(cropped_img.size)

    # Chuyển ảnh sang RGBA để thêm alpha
    if cropped_img.shape[2] == 3:
        rgba = np.concatenate([cropped_img, np.ones((*cropped_img.shape[:2], 1), dtype=np.uint8)*255], axis=2)
    else:
        rgba = cropped_img.copy()  # đã có alpha

    # Áp dụng alpha theo mask
    # rgba[..., 3] = (cropped_mask * 255).astype(np.uint8)
    rgba[..., 3] = np.where(cropped_mask, 255, 0).astype(np.uint8)


    pil_img = Image.fromarray(rgba, mode='RGBA')
    print(pil_img.size)

    # pil_img.save('005.jpg')

    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # buffered = io.BytesIO()
    # pil_img_cropped.save(buffered, format='PNG')
    # encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print('Done: ')
    return encoded_img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1

import torchvision.transforms as T
from scipy.ndimage import label as connected_components
@app.route('/segment_image', methods=['POST'])
def segment_image():
    try:
        data = request.get_json()
        image_data_url = data['image_data']
        # Remove prefix 'data:image/png;base64,'
        header, encoded = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Đọc ảnh từ bytes
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_pil.save("output_image_1.png")


        # img_np = np.array(img_pil)  # RGB


        original_image_np, segmentation_mask, segments_info, id2label = predict_masks(img_pil)
        print('segmentation_mask: ', segmentation_mask.shape)
        instance_images = []
        num_classes = len(id2label)
        palette = color_palette(num_classes)

        for segment in segments_info:
            if segment["label_id"] == 0:
                continue
            color = palette[segment["label_id"]]
            mask = segmentation_mask == segment["id"]
            label_id = segment.get("label_id")
            category = id2label[label_id]
            labeled_mask, num_instances = connected_components(mask)
            for instance_id in range(1, num_instances + 1):
                instance_mask = (labeled_mask == instance_id).astype(np.uint8)
                y_indices, x_indices = np.where(instance_mask == 1)
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue

                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                width = x_max - x_min + 1
                height = y_max - y_min + 1
                centroid_x = int(np.mean(x_indices - x_min))  # relative trong bbox
                centroid_y = int(np.mean(y_indices - y_min))

                # encoded_mask, bbox = encode_mask_image(instance_mask, color, category)
                encoded_img, x, y, w, h = extract_object_with_mask(img_pil, instance_mask)
                # encoded_mask.save('002.jpg')
                print(x, y, w, h )


                instance_images.append({
                    "id": f"{label_id}_{instance_id}",
                    "label": category,
                    "image_base64": encoded_img,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "bbox": [int(x_min), int(y_min), int(width), int(height)],
                    "centroid": [centroid_x, centroid_y]
                })

            # instance_images.append({
            #     'image_base64': encoded_mask,
            #     'x': centroid_x,
            #     'y': centroid_y,
            #     'category': category
            # })

            # instance_images.append(encoded_mask)

        return jsonify({'instances': instance_images})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load SAM model once khi server khởi động
# sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
# mask_generator = SamAutomaticMaskGenerator(sam)

# def generate_mask_overlay(image_rgb: np.ndarray, masks: list):
#     """
#     Vẽ các mask và bounding box lên ảnh RGB gốc, trả về ảnh overlay (numpy uint8)
#     """
#     plt.figure(figsize=(12, 12))
#     fig, ax = plt.subplots(1)
#     ax.imshow(image_rgb)

#     for mask in masks:
#         m = mask["segmentation"]  # bool mask
#         bbox = mask["bbox"]       # [x, y, w, h]
#         color = random_color(alpha=0.5)

#         rgba_mask = np.zeros((*m.shape, 4))
#         rgba_mask[m] = color

#         ax.imshow(rgba_mask)

#         x, y, w, h = bbox
#         rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color[:3], facecolor='none')
#         ax.add_patch(rect)

#     plt.axis('off')
#     plt.tight_layout(pad=0)

#     # Lưu ảnh từ figure vào buffer rồi đọc lại để trả về
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#     buf.seek(0)
#     img = Image.open(buf).convert('RGBA')
#     return img

# @app.route('/segment_image', methods=['POST'])
# def segment_image_api():
#     try:
#         data = request.get_json()
#         image_data_url = data['image_data']
#         # data = request.get_json(force=True)
#         # image_data_url = data.get('image_data')
#         # print('Here: ', image_data_url)
#         if not image_data_url:
#             return jsonify({'error': 'No image_data found in request'}), 400

#         # Tách header base64
#         header, encoded = image_data_url.split(',', 1)
#         img_bytes = base64.b64decode(encoded)
#         image_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
#         print(image_pil.size)

#         # Chuyển PIL sang numpy RGB
#         image_rgb = np.array(image_pil)

#         # Tạo mask với SAM
#         masks = mask_generator.generate(image_rgb)

#         # Tạo ảnh overlay mask + bbox
#         result_img = generate_mask_overlay(image_rgb, masks)
#         print(result_img.size)
#         # Encode ảnh kết quả sang base64
#         buffered = io.BytesIO()
#         result_img.save(buffered, format='PNG')
#         encoded_result = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         result_data_url = f'data:image/png;base64,{encoded_result}'
#         print(result_data_url[:30])

#         return jsonify({'segmented_image_base64': result_data_url})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
