import io
import os
import cv2
import time
import uuid
import atexit
import paddle
import numpy as np
import datetime
import json
import requests
import statistics
import traceback
from ocr import ocr, ocr_text
from flask_cors import CORS
from loguru import logger
from s3 import getObject, putObject
from shared_state import shared_state
from cryptor import FolderCryptor
from parse_file_director import ParseOCRDirector
from flask import Flask, request, jsonify, send_file
from algo.enhancement.enhancement import enhancement
from algo.images_detect.images_detect import plag_pdfImages
from algo.remove_qrcode.identify_qrcode import identify_qrcode
from algo.handwriting.remove_handwriting import remove_handwriting
from algo.pdf.pdf2images import extract_images_from_pdf, doc2imgList, pdf2images, analyze_image
from algo.table_detect.table_detect import image2tables
from algo.header_footer.remove_footer_header import rm_footer_header, get_rm_footer_header, get_header_footer

# 模型文件夹管理：自动解密，进程退出时加密
MODEL_FOLDER = "models"
MODEL_KEY = os.environ.get("MODEL_KEY", "").encode()
_cryptor = FolderCryptor(MODEL_KEY) if MODEL_KEY else None
if _cryptor is not None:
    _cryptor.decrypt_folder(MODEL_FOLDER)


def _encrypt_models():
    if _cryptor is not None:
        _cryptor.encrypt_folder(MODEL_FOLDER)


atexit.register(_encrypt_models)
from algo.redmask_remove.remove_redmask import infer_start, remove_red_seal

from config import structure, OCR, table_OCR, handwriting_model, paddleOCR, uv_model, wired_predictor, cls_model, text_det_predictor, ORIENT_CLASSES, orientation_model, textline_orientation_model

sendImage_url = os.environ.get("SEND_SERVER", "http://127.0.0.1:5700")
DEBUG_SAVE_DIR = os.environ.get("DEBUG_SAVE_DIR", "")  # 调试开关，设置路径后保存前端图和OCR裁剪图

# 版面分析缓存：避免同一页重复调用 PP-DocLayoutV2
_layout_cache = {}


def _get_cached_layout(img_input):
    """获取缓存的版面分析结果。支持文件路径（缓存）和 numpy 数组（直接计算）。"""
    if isinstance(img_input, str):
        if img_input not in _layout_cache:
            with shared_state.structure_lock:
                _layout_cache[img_input] = structure.predict(img_input, layout_nms=True)
        return _layout_cache[img_input]
    else:
        with shared_state.structure_lock:
            return structure.predict(img_input, layout_nms=True)


def _clear_layout_cache():
    _layout_cache.clear()

paddle.set_flags({'FLAGS_eager_delete_tensor_gb': 0.0})
paddle.set_flags({'FLAGS_fast_eager_deletion_mode': True})
paddle.set_flags({'FLAGS_memory_fraction_of_eager_deletion': 1})
paddle.set_flags({"FLAGS_allocator_strategy": "naive_best_fit"})
try:
    os.mkdir("/tmp")
except FileExistsError:
    pass
app = Flask(__name__)
app.json.sort_keys = False
CORS(app, supports_credentials=True)
host = os.environ.get("hostname", "http://127.0.0.1:5000")
logger.info("日志输出中------")
logger.info(f"hostname:{host}")



def _download_http(url, filepath):
    """通过 HTTP 直接下载文件。"""
    import urllib.request
    urllib.request.urlretrieve(url, filepath)
def pdf_download_file(file_ref) -> str:
    """下载 PDF。支持 S3 key 或完整 HTTP URL。"""
    filepath = f'/tmp/{uuid.uuid4()}.pdf'
    if file_ref.startswith('http://') or file_ref.startswith('https://'):
        _download_http(file_ref, filepath)
    else:
        data = getObject(file_ref)
        with open(filepath, 'wb') as f:
            f.write(data)
    return filepath


def download_file(file_ref) -> str:
    """下载文件。支持 S3 key 或完整 HTTP URL。"""
    filename = file_ref.split('/')[-1].split('?')[0]
    filepath = f'/tmp/{filename}'
    if os.path.exists(filepath):
        return filepath
    filepath = f'/tmp/{uuid.uuid4()}.png'
    if file_ref.startswith('http://') or file_ref.startswith('https://'):
        _download_http(file_ref, filepath)
    else:
        data = getObject(file_ref)
        with open(filepath, 'wb') as f:
            f.write(data)
    return filepath



def file_bytes(file_path):
    return_data = io.BytesIO()
    with open(file_path, 'rb') as fo:
        return_data.write(fo.read())
    return_data.seek(0)
    os.remove(file_path)
    return return_data


@app.route('/', methods=['GET', 'HEAD'])
def hello():
    with open('welcome.html', 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content


@app.route('/object/<filename>', methods=['GET', 'DELETE'])
def manage_object(filename: str):
    if request.method == 'DELETE':
        try:
            os.remove(f'/tmp/{filename}')
            return jsonify({
                "code": 200,
                "status": "Success",
                "msg": "File deleted successfully."
            })
        except FileNotFoundError:
            return jsonify({
                "code": 404,
                "status": "Not Found",
                "msg": "File not found.",
                "data": {"error": "File not found."}
            })
        except Exception as e:
            return jsonify({
                "code": 500,
                "status": "Internal Server Error",
                "msg": "An error occurred while deleting the file.",
                "data": {"error": str(e)}
            })

    if not os.path.exists(f'/tmp/{filename}'):
        return jsonify({
            "code": 404,
            "status": "Not Found",
            "msg": "File not found.",
            "data": {"error": "File not found."}
        })
    ext = filename.split('.')[-1].lower()
    if 'jpg' in ext or 'jpeg' in ext:
        new_filename = f'{uuid.uuid4()}.jpg'
        img = cv2.imread(f'/tmp/{filename}')
        cv2.imwrite(f'/tmp/{new_filename}', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return send_file(file_bytes(f'/tmp/{new_filename}'), mimetype='image/jpeg', download_name=filename)
    return send_file(f'/tmp/{filename}')


@app.route('/image_similarity', methods=['POST'])
def similarity_images():
    originPdf = request.files['originPdf']
    targetPdf = request.files['targetPdf']
    pdf_path1 = f'/tmp/{uuid.uuid4()}.pdf'
    pdf_path2 = f'/tmp/{uuid.uuid4()}.pdf'
    originPdf.save(pdf_path1)
    targetPdf.save(pdf_path2)
    try:
        similarity = plag_pdfImages(pdf_path1, pdf_path2)
    except:
        similarity = []
    os.remove(pdf_path1)
    os.remove(pdf_path2)
    return jsonify({
        "code": 0,
        "result": similarity
    })


@app.route('/pdf-to-image', methods=['POST'])
def pdf_to_image():
    files = request.files['pdf']
    pdf_path = f'/tmp/{uuid.uuid4()}.pdf'
    files.save(pdf_path)
    images = doc2imgList(pdf_path)
    os.remove(pdf_path)
    images = list(map(lambda x: x.split('/')[-1], images))
    images = list(map(lambda x: f"{host}/object/{x}", images))
    return jsonify({
        "code": 0,
        "data": {"images": images}
    })


@app.route('/textPdf-image-ocr', methods=['POST'])
def textPdf_image_ocr():
    files = request.files['pdf']
    pdf_path = f'/tmp/{uuid.uuid4()}.pdf'
    files.save(pdf_path)
    result = extract_images_from_pdf(pdf_path)
    os.remove(pdf_path)
    return jsonify({
        "code": 0,
        "data": {"result": result}
    })


@app.route('/remove_directory', methods=['POST'])
def process_json():
    json_data = request.get_json()

    if not json_data:
        return []
    ocr_parser = ParseOCRDirector()
    try:
        result = ocr_parser.read_and_process_json(json_data)
    except Exception as e:
        logger.info(str(e))
        return jsonify(json_data)
    return jsonify(result)


def remove_directory(office_result):
    filter_data = []
    for page_data in office_result:
        page = page_data["page"]
        for line in page_data["lines"]:
            for char_data in line["chars"]:
                filter_data.append({
                    "char": char_data["char"],
                    "x": char_data["bbox"]["x"],
                    "y": char_data["bbox"]["y"],
                    "width": char_data["bbox"]["w"],
                    "height": char_data["bbox"]["h"],
                    "page": page
                })
    ocr_parser = ParseOCRDirector()
    result = ocr_parser.read_and_process_json(filter_data)
    filtered_office_result = []
    for page_result in office_result:
        page = page_result['page']
        filtered_lines = []
        for line in page_result['lines']:
            keep_line = True
            for collection in result['collections']:
                start_paragraph = collection['start_paragraph']
                end_paragraph = collection['end_paragraph']
                top_y = start_paragraph['y'] - 50
                if top_y < 0:
                    top_y = 0
                if page == start_paragraph['page'] and page == end_paragraph['page']:
                    if (0 <= line['bbox']['x'] <= end_paragraph['x'] and
                            top_y <= line['bbox']['y'] <= end_paragraph['y']):
                        keep_line = False
                        break
            if keep_line:
                filtered_lines.append(line)
        for new_idx, line in enumerate(filtered_lines):
            line["line"] = new_idx
        filtered_page_result = {
            "image": page_result["image"],
            "tables": page_result["tables"],
            "lines": filtered_lines,
            "red_masks": page_result["red_masks"],
            "page": page
        }
        filtered_office_result.append(filtered_page_result)
    return filtered_office_result


def download_and_convert_pdf_to_images(fileUrl, fileType, fileExt, pageRange):
    pdf_path = pdf_download_file(fileUrl)
    images, embed_images = pdf2images(pdf_path, pageRange=pageRange)
    images_new = []
    if embed_images and fileExt != "tiff":
        for image in images:
            is_scan, _ , _  = analyze_image(image)
            pdf_type = "扫描件" if is_scan else "拍照件"
            if pdf_type == "拍照件":
                img_array = cv2.imread(image)
                image_data = uv_images(img_array)
                output_path = f"/tmp/{uuid.uuid4()}.png"
                cv2.imwrite(output_path, image_data)
                os.remove(image)
                images_new.append(output_path)
            else:
                images_new.append(image)
    else:
        images_new = images
    if fileType == 'textPdf':
        header_result, footer_result = get_header_footer(pdf_path, pageRange=pageRange)
    else:
        header_result = []
        footer_result = []
    os.remove(pdf_path)
    return images_new, footer_result, header_result


def remove_qrcode_process(images):
    """images: list of file paths or numpy arrays. Returns same type as input."""
    qrcode_result = []
    for img in images:
        processed_img = identify_qrcode(img)
        if isinstance(img, str):
            img_name = f"/tmp/{uuid.uuid4()}.png"
            cv2.imwrite(img_name, processed_img)
            qrcode_result.append(img_name)
            time.sleep(0.02)
            os.remove(img)
        else:
            qrcode_result.append(processed_img)
    return qrcode_result


def remove_redSeal_process(images, ignoreSeals):
    """去红章。输入输出同类型：文件路径列表 或 numpy 数组列表。"""
    redmask_images = []
    remove_Redmask = []
    is_file_mode = images and isinstance(images[0], str)

    for img_input in images:
        img = cv2.imread(img_input) if is_file_mode else img_input
        h, w = img.shape[:2]
        remove_redmask_result = infer_start(img_input, ignoreSeals, img_array=img)

        if remove_redmask_result:
            regions = [[r['bbox']['x'], r['bbox']['y'], r['bbox']['w'], r['bbox']['h']] for r in remove_redmask_result]
            img = remove_red_seal(img, regions)

        if is_file_mode:
            out_path = f"/tmp/{uuid.uuid4()}.png"
            cv2.imwrite(out_path, img)
            redmask_images.append(out_path)
            os.remove(img_input)
        else:
            redmask_images.append(img)
        remove_Redmask.append(remove_redmask_result)

    return redmask_images, remove_Redmask


def remove_num2_footer_header(images, ignorePageHeaderAndFooter):
    """页眉页脚去除。输入输出同类型：文件路径列表 或 numpy 数组列表。"""
    if ignorePageHeaderAndFooter:
        is_file_mode = images and isinstance(images[0], str)
        result = []
        for img in images:
            layout = _get_cached_layout(img)
            processed = rm_footer_header(img, structure, boxes=layout[0]['boxes'])
            result.append(processed)
        return result
    return images


def get_num2_footer_header(images, ignorePageHeaderAndFooter):
    images_list = []
    if ignorePageHeaderAndFooter:
        for img1 in images:
            img = img1 if not isinstance(img1, str) else cv2.imread(img1, cv2.IMREAD_COLOR)
            img_res = get_rm_footer_header(img, structure)
            images_list.append(img_res)
            if isinstance(img1, str): os.remove(img1)
        return images_list
    else:
        return images_list


def remove_footer_header(images, ignorePageHeaderAndFooter):
    header_list = [0] * len(images)
    footer_list = [1] * len(images)

    return images, header_list, footer_list


def slope_process(images):
    slope_images = []
    # memory = cp.get_default_memory_pool()
    # pinned_memory = cp.get_default_pinned_memory_pool()
    for image_path in images:
        # img_name = f"/tmp/{uuid.uuid4()}.png"
        image_ori = f"/tmp/{uuid.uuid4()}.png"
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        texts, scores = ocr_text(image_path, OCR)
        if texts is None or not texts:
            rotated = img
        else:
            # 计算置信度平均值
            scores = [s for s in scores if isinstance(s, (int, float))]
            average_score = sum(scores) / len(scores) if scores else 0
            # 根据平均值决定是否旋转
            if average_score <= 0.5:
                rotated = cv2.rotate(img, cv2.ROTATE_180)
            else:
                rotated = img
        cv2.imwrite(image_ori, rotated)
        # img_slope = slope_correct(image_ori)
        # cv2.imwrite(img_name, img_slope)
        slope_images.append(image_ori)
        os.remove(image_path)
        # os.remove(image_ori)
    # memory.free_all_blocks()
    # pinned_memory.free_all_blocks()
    return slope_images


def enhancement_process(images):
    def enhance_and_save(img_path):
        img = cv2.imread(img_path)
        if img is None:
            return img_path
        img_name = f"/tmp/{uuid.uuid4()}.png"
        enhanced_img_data = enhancement(img)
        cv2.imwrite(img_name, enhanced_img_data)
        try:
            os.remove(img_path)
        except:
            pass
        return img_name

    enhance_images = [enhance_and_save(img_data) for img_data in images]
    return enhance_images


def watermark_removal_process(images):
    """多页投票去水印，单页自动降级为单页二值化。输入输出均为 numpy 数组列表。"""
    from algo.enhancement.enhancement import enhancement_multi_by_size, enhancement
    # 兼容文件路径和 numpy 数组
    imgs = [img if not isinstance(img, str) else cv2.imread(img) for img in images]
    if len(imgs) <= 1:
        result = [enhancement(img) for img in imgs]
    else:
        result = enhancement_multi_by_size(imgs)
    # 清理旧的临时文件（如果有）
    for old in images:
        if isinstance(old, str):
            try: os.remove(old)
            except: pass
    return result  # 返回 numpy 数组列表


def _preprocess_orientation(img):
    h, w = img.shape[:2]
    if h < w:
        new_h, new_w = 256, int(w * 256 / h)
    else:
        new_w, new_h = 256, int(h * 256 / w)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    sy, sx = (new_h - 224) // 2, (new_w - 224) // 2
    img = img[sy:sy + 224, sx:sx + 224]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return np.transpose(img, (2, 0, 1))


def predict_orientation(img):
    inp = _preprocess_orientation(img)
    batch = np.stack([inp, inp, inp], axis=0).astype(np.float32)
    out = orientation_model.run(None, {'x': batch})[0]
    preds = np.argmax(out, axis=1)
    return int(np.argmax(np.bincount(preds)))


ROTATE_MAP = {
    0: None,
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}

ROTATE_DESC = {0: "不旋转", 1: "逆时针90°", 2: "旋转180°", 3: "顺时针90°"}


def _detect_page_orientation_textline(image_path):
    """
    方向检测-方法 textline（默认）：
    1. 文本框几何分析 → 确定水平(0°/180°)还是垂直(90°/270°)，忽略非标准方向框
    2. PP-LCNet_x1_0_textline_ori → 区分 0° 和 180°
    """
    t_start = time.time()

    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path  # numpy 数组
    if img is None:
        return None

    t0 = time.time()
    det_result = list(text_det_predictor(image_path if isinstance(image_path, str) else img))
    t_ocr = time.time() - t0
    if not det_result:
        return None
    boxes = det_result[0].get('dt_polys', [])
    if len(boxes) == 0:
        return None

    # ====== Step 1: 文本框几何分析 ======
    t1 = time.time()
    top_n = min(len(boxes), 100)
    horizontal_count = 0    # 主轴 ≈ 0°（水平文字）
    vertical_count = 0      # 主轴 ≈ 90°（垂直文字）
    box_crops = []           # (crop_bgr, is_horizontal, area) 用于 textline_ori

    for idx in range(top_n):
        poly = np.array(boxes[idx], dtype=np.float32)
        if poly.shape[0] != 4:
            continue

        # 判断四条边是否接近标准方向（偏差 < 5°）
        pts_shifted = np.roll(poly, -1, axis=0)
        edges = pts_shifted - poly
        edge_angles = np.abs(np.arctan2(edges[:, 1], edges[:, 0]) * 180 / np.pi) % 90
        if not np.all(np.minimum(edge_angles, 90 - edge_angles) < 5):
            continue  # 非标准方向文本框，忽略

        # 找出主轴方向（较长的那组对边）
        edge_lens = np.linalg.norm(edges, axis=1)
        pair_03 = edge_lens[0] + edge_lens[2]
        pair_12 = edge_lens[1] + edge_lens[3]
        dominant_idx = 0 if pair_03 > pair_12 else 1
        dom_angle = np.arctan2(edges[dominant_idx, 1], edges[dominant_idx, 0]) * 180 / np.pi
        dom_angle = dom_angle % 180  # [0, 180)

        if min(dom_angle, 180 - dom_angle) < 10:
            horizontal_count += 1
            is_horizontal = True
        elif abs(dom_angle - 90) < 10:
            vertical_count += 1
            is_horizontal = False
        else:
            continue  # 理论上不会到这里（已过滤非标准方向）

        # 裁剪文本框
        x_min = max(0, int(poly[:, 0].min()) - 2)
        y_min = max(0, int(poly[:, 1].min()) - 2)
        x_max = min(img.shape[1], int(poly[:, 0].max()) + 2)
        y_max = min(img.shape[0], int(poly[:, 1].max()) + 2)
        crop = img[y_min:y_max, x_min:x_max]
        if crop.size > 0:
            area = crop.shape[0] * crop.shape[1]
            box_crops.append((crop, is_horizontal, area))

    total_standard = horizontal_count + vertical_count
    if total_standard == 0:
        return None

    # 多数投票：水平还是垂直
    if horizontal_count >= vertical_count:
        base_rotation = 0      # 0° 或 180°
        target_boxes = [(c, a) for c, h, a in box_crops if h]
    else:
        base_rotation = 90     # 90° 或 270°
        target_boxes = [(c, a) for c, h, a in box_crops if not h]

    t_geom = time.time() - t1
    logger.info(
        f"[orient textline] geometry: horizontal={horizontal_count} vertical={vertical_count} "
        f"→ base_rotation={base_rotation}°  boxes={len(target_boxes)}  ocr={t_ocr:.2f}s  geom={t_geom:.2f}s"
    )

    # ====== Step 2: PP-LCNet_x1_0_textline_ori 区分 0° vs 180° ======
    t2 = time.time()
    if not target_boxes:
        return {0: 0, 90: 1}.get(base_rotation, 0)  # 降级：几何结果

    # 按面积降序取前 20 个
    target_boxes.sort(key=lambda x: -x[1])
    target_boxes = target_boxes[:10]

    textline_results = []

    for crop, _area in target_boxes:
        # 垂直文本框先旋转 90° 使其变为水平
        if base_rotation == 90:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # PaddleX predictor
        pred = list(textline_orientation_model.predict(crop))
        result = int(pred[0]['class_ids'][0].item())  # 0 → 0_degree, 1 → 180_degree
        textline_results.append(result)

    if not textline_results:
        return {0: 0, 90: 1}.get(base_rotation, 0)

    # 多数投票
    textline_vote = statistics.mode(textline_results)
    textline_rotation = 0 if textline_vote == 0 else 180

    # 最终角度
    final_angle = (base_rotation + textline_rotation) % 360
    # 映射到返回值：0→不用转, 1→逆90°, 2→转180°, 3→顺90°
    ANGLE_TO_ID = {0: 0, 90: 1, 180: 2, 270: 3}
    result_id = ANGLE_TO_ID[final_angle]

    vote_0 = textline_results.count(0)
    vote_180 = textline_results.count(1)
    t_onnx = time.time() - t2
    t_total = time.time() - t_start
    logger.info(
        f"[orient textline] base={base_rotation}° vote=0°={vote_0}/180°={vote_180} "
        f"→ final={final_angle}° (id={result_id})  onnx={t_onnx:.2f}s  total={t_total:.2f}s"
    )
    return result_id


def _detect_page_orientation_char(image_path):
    """
    方向检测-方法 char：OCR 字符框裁剪 + rapid_orientation 4 分类取众数。
    1. OCR 获取字符级包围盒 (text_word_region)
    2. 每个字符裁剪 → pad square → 224×224 → ImageNet normalize
    3. rapid_orientation 3x 投票 → 4 分类 (0/90/180/270)
    4. 所有字符结果取众数
    """
    t_start = time.time()

    img = cv2.imread(image_path)
    if img is None:
        return None

    t0 = time.time()
    det_result = paddleOCR.predict(image_path)
    t_ocr = time.time() - t0
    if not det_result:
        return None

    text_word_region = det_result[0].get('text_word_region', [])
    if not text_word_region:
        return None

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    orientations = []
    total_chars = 0
    skipped = 0

    t1 = time.time()

    for line_boxes in text_word_region:
        for box in line_boxes:
            total_chars += 1
            pts = np.array(box)
            x_min = int(pts[:, 0].min())
            y_min = int(pts[:, 1].min())
            x_max = int(pts[:, 0].max())
            y_max = int(pts[:, 1].max())

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.shape[1], x_max)
            y_max = min(img.shape[0], y_max)

            if x_max <= x_min or y_max <= y_min:
                skipped += 1
                continue

            char_crop = img[y_min:y_max, x_min:x_max]
            h, w = char_crop.shape[:2]
            if h < 3 or w < 3:
                skipped += 1
                continue

            side = max(h, w)
            padded = np.zeros((side, side, 3), dtype=np.uint8)
            y_off = (side - h) // 2
            x_off = (side - w) // 2
            padded[y_off:y_off + h, x_off:x_off + w] = char_crop

            block = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            block = block.astype(np.float32) / 255.0
            block = (block - mean) / std
            block = np.transpose(block, (2, 0, 1))

            batch = np.stack([block, block, block], axis=0).astype(np.float32)
            out = orientation_model.run(None, {'x': batch})[0]
            preds = np.argmax(out, axis=1)
            cls_id = int(np.argmax(np.bincount(preds)))
            orientations.append(cls_id)

    if not orientations:
        logger.warning(f"[orient char] all {total_chars} chars skipped")
        return None

    result_id = statistics.mode(orientations)
    vote_0 = orientations.count(0)
    vote_90 = orientations.count(1)
    vote_180 = orientations.count(2)
    vote_270 = orientations.count(3)

    t_onnx = time.time() - t1
    t_total = time.time() - t_start
    logger.info(
        f"[orient char] total={total_chars} valid={len(orientations)} skipped={skipped} "
        f"0°={vote_0} 90°={vote_90} 180°={vote_180} 270°={vote_270} "
        f"→ result={result_id} ({result_id * 90}°)  ocr={t_ocr:.2f}s  onnx={t_onnx:.2f}s  total={t_total:.2f}s"
    )
    return result_id


def _detect_page_orientation(image_path, orient_method="textline"):
    """方向检测入口，根据 orient_method 分发到不同的检测逻辑"""
    if orient_method == "char":
        return _detect_page_orientation_char(image_path)
    else:
        return _detect_page_orientation_textline(image_path)


def orientation_process(images, orient_method="textline"):
    """方向矫正。输入输出同类型。"""
    import uuid as _uuid
    orient_images = []
    is_file_mode = images and isinstance(images[0], str)
    for img_input in images:
        if is_file_mode:
            angle = _detect_page_orientation(img_input, orient_method)
            img = cv2.imread(img_input)
        else:
            tmp_path = f"/tmp/{_uuid.uuid4()}.png"
            cv2.imwrite(tmp_path, img_input)
            angle = _detect_page_orientation(tmp_path, orient_method)
            img = img_input
            try: os.remove(tmp_path)
            except: pass
        if angle is None or angle == 0:
            orient_images.append(img_input)
            continue
        angle_deg = int(angle) * 90
        rotate_code = ROTATE_MAP[angle]
        img = cv2.rotate(img, rotate_code)
        if is_file_mode:
            cv2.imwrite(img_input, img)
            orient_images.append(img_input)
        else:
            orient_images.append(img)
    return orient_images

def _unrotate_bbox(x, y, w, h, rotation, crop_w, crop_h):
    """Transform bbox from rotated crop coordinates back to original crop coordinates."""
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return (y, crop_h - x - w, h, w)
    elif rotation == cv2.ROTATE_180:
        return (crop_w - x - w, crop_h - y - h, w, h)
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (crop_w - y - h, x, h, w)
    return (x, y, w, h)


def table_detection(image, boxes=None):
    if boxes is None:
        with shared_state.structure_lock:
            output = structure.predict(image)
        boxes = output[0]['boxes']
    res = {'boxes': boxes}
    figures = []
    tables_info = []  # 存储多个表格的信息，包括裁剪后的图和偏移量

    for box in res['boxes']:
        if box['label'] == 'table':
            x1, y1, x2, y2 = map(int, box['coordinate'])

            # 使用有框表格模型对边框进行补全
            table_img_roi = image[y1:y2, x1:x2]
            table_type = "unknown"
            if table_img_roi.size > 0:
                try:
                    with shared_state.gpu_lock:
                        cls_output = cls_model.predict(table_img_roi)
                    # 增加对 cls_output 的非空校验
                    if cls_output:
                        for res_cls in cls_output:
                            if 'scores' in res_cls and 'label_names' in res_cls:
                                scores = res_cls['scores']
                                label_names = res_cls['label_names']
                                max_idx = np.argmax(scores)
                                table_type = label_names[max_idx]
                                break
                except Exception as e:
                    logger.error(f"Table classification failed: {e}")

                # 如果检测为无框表格（wireless），则不处理，当作没检测到
                if table_type == 'wireless_table':
                    continue

                try:
                    with shared_state.gpu_lock:
                        wired_output = wired_predictor.predict(table_img_roi)

                    # 核心修复：检查 wired_output 是否为 None 或空，防止 PreconditionNotMetError
                    if wired_output is not None:
                        for wired_res in wired_output:
                            w_boxes = []
                            if "boxes" in wired_res:
                                w_boxes = wired_res["boxes"]
                            elif "bbox" in wired_res:
                                w_boxes = wired_res["bbox"]

                            for w_box in w_boxes:
                                if isinstance(w_box, dict):
                                    wx1, wy1, wx2, wy2 = w_box['coordinate']
                                elif len(w_box) == 6:
                                    _, _, wx1, wy1, wx2, wy2 = w_box
                                else:
                                    wx1, wy1, wx2, wy2 = w_box[:4]

                                # 将坐标映射回原图并绘制黑色边框
                                cv2.rectangle(image, (int(wx1 + x1), int(wy1 + y1)),
                                              (int(wx2 + x1), int(wy2 + y1)), (0, 0, 0), 2)
                except Exception as e:
                    logger.error(f"Wired predictor failed: {e}")

                # 重新获取可能已被绘制边框的区域，并在边缘多留10像素的边距
                h_img, w_img = image.shape[:2]
                y1_crop = max(0, y1 - 10)
                y2_crop = min(h_img, y2 + 10)
                x1_crop = max(0, x1 - 10)
                x2_crop = min(w_img, x2 + 10)

                cropped_table = image[y1_crop:y2_crop, x1_crop:x2_crop].copy()

                # 存储该表格的裁剪图以及相对于原图的偏移量（考虑了10像素边距）
                tables_info.append({
                    'image': cropped_table,
                    'x_offset': x1_crop,
                    'y_offset': y1_crop,
                    'bbox': [x1, y1, x2, y2], # 原始检测框
                    'type': table_type
                })

        elif box['label'] == 'image':
            x1, y1, x2, y2 = map(int, box['coordinate'])
            # 收集图片区域坐标
            figures.append([x1, y1, x2, y2])

    paddle.device.set_device('gpu')
    paddle.device.cuda.empty_cache()

    return tables_info, figures


def uv_images(img_data):
    ori_result = uv_model.predict(img_data)
    res_ori = next(ori_result)
    img_array = res_ori['doctr_img']
    paddle.device.set_device('gpu')
    paddle.device.cuda.empty_cache()
    return img_array


def resize_image(image_path):
    max_side = 4000
    png_compression = 0
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    print(f"原图尺寸: {w} x {h}，最大边限制: {max_side}")
    if max(w, h) <= max_side:
        print("图片尺寸符合要求，无需缩放")
        return image_path
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    print(f"缩放至: {new_w} x {new_h} (scale={scale:.4f})")
    # cv2.INTER_LANCZOS4 或 cv2.INTER_AREA 对于缩小效果较好
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    out_path = f"/tmp/{uuid.uuid4()}.png"
    cv2.imwrite(out_path, resized,[cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    os.remove(image_path)
    return out_path


def removeHandwriting(images):
    """去手写。输入输出同类型：文件路径列表 或 numpy 数组列表。"""
    first = images[0]
    if isinstance(first, str):
        first_img = cv2.imread(first)
    else:
        first_img = first
    with shared_state.handwriting_lock:
        first_out = remove_handwriting(handwriting_model, first_img)
    if isinstance(first, str):
        new_first = f"/tmp/{uuid.uuid4()}.png"
        cv2.imwrite(new_first, first_out)
        os.remove(first)
        return [new_first] + images[1:]
    return [first_out] + images[1:]


def _append_table_chars(chars, ch, x1, y1, cw, ch_h, rotation, orig_w, orig_h, x_offset, y_offset):
    text = ch.get("char", "")
    if not text:
        return

    score = ch.get("score")
    count = len(text)
    for idx, single_char in enumerate(text):
        x0 = x1 + cw * idx / count
        w0 = cw / count
        if rotation:
            x_crop, y_crop, w_crop, h_crop = _unrotate_bbox(
                x0 - 20, y1 - 30, w0, ch_h, rotation, orig_w, orig_h
            )
            chars.append({
                "bbox": {
                    "x": int(round(x_crop + x_offset)),
                    "y": int(round(y_crop + y_offset)),
                    "w": max(1, int(round(w_crop))),
                    "h": max(1, int(round(h_crop)))
                },
                "char": single_char,
                "score": score
            })
        else:
            chars.append({
                "bbox": {
                    "x": int(round(x0 - 20 + x_offset)),
                    "y": int(round(y1 - 30 + y_offset)),
                    "w": max(1, int(round(w0))),
                    "h": ch_h
                },
                "char": single_char,
                "score": score
            })


def ocr_proces(images, ignorePageHeaderAndFooter):
    ocr_result = []
    for img_input in images:
        start_time = time.time()
        table_result = []
        page_result = []
        # 表格识别
        if isinstance(img_input, str):
            img_pro = cv2.imread(img_input, cv2.IMREAD_COLOR)
            if img_pro is None:
                logger.error(f"无法读取图片: {img_input}")
                continue
        else:
            img_pro = img_input

        img_res_copy = img_pro.copy()
        # table_detection 现在返回 tables_info (列表，包含多个表格的裁剪图和偏移量) 和 figures
        layout = _get_cached_layout(img_input)
        tables_info, figures = table_detection(img_res_copy, boxes=layout[0]['boxes'])

        # 检查每个表格裁剪图的朝向（用 textline 方法），非 0° 则旋转
        for t_info in tables_info:
            table_img = t_info['image']
            if table_img is not None and table_img.size > 0:
                orientation = _detect_page_orientation_textline(table_img)
                if orientation is not None and orientation != 0:
                    rotate_code = ROTATE_MAP[orientation]
                    orig_h, orig_w = table_img.shape[:2]
                    t_info['image'] = cv2.rotate(table_img, rotate_code)
                    t_info['rotation'] = rotate_code
                    t_info['orig_w'] = orig_w
                    t_info['orig_h'] = orig_h

        clear_img = img_pro.copy()
        # 涂白图片区域
        for fig in figures:
            x1, y1, x2, y2 = fig
            cv2.rectangle(clear_img, (x1, y1), (x2, y2), (255, 255, 255), -1)

        all_tables_data = [] # 存储每个表格的 image2tables 结果
        total_shape_count = 0

        # 处理检测到的多个表格
        for t_info in tables_info:
            table_img = t_info['image']
            if table_img is not None and table_img.size > 0:
                tables = image2tables(table_img)
            else:
                tables = []

            shape_count = sum(len(cells or []) for cells in tables)
            total_shape_count += shape_count

            if shape_count <= 50:
                all_tables_data.append({
                    'tables': tables,
                    'info': t_info
                })
                # 涂白表格区域以便后续OCR文字
                x_offset = t_info['x_offset']
                y_offset = t_info['y_offset']
                rotation = t_info.get('rotation')
                orig_w = t_info.get('orig_w', 0)
                orig_h = t_info.get('orig_h', 0)
                for table_cells in tables:
                    for shape in table_cells:
                        x, y, w, h = shape
                        if rotation:
                            x_orig, y_orig, w_orig, h_orig = _unrotate_bbox(x, y, w, h, rotation, orig_w, orig_h)
                            cv2.rectangle(clear_img, (x_orig + x_offset, y_orig + y_offset),
                                          (x_orig + x_offset + w_orig, y_orig + y_offset + h_orig), (255, 255, 255), -1)
                        else:
                            cv2.rectangle(clear_img, (x + x_offset, y + y_offset),
                                          (x + x_offset + w, y + y_offset + h), (255, 255, 255), -1)
            else:
                # 如果单元格过多，不作为表格处理，也不涂白
                pass

        w = clear_img.shape[1]
        h = clear_img.shape[0]
        cv2.rectangle(clear_img, (0, 0), (int(w * 0.03), h), (255, 255, 255), -1)
        cv2.rectangle(clear_img, (int(w * 0.97), 0), (w, h), (255, 255, 255), -1)
        clear_img_path = f'/tmp/{uuid.uuid4()}.png'
        cv2.imwrite(clear_img_path, clear_img)

        if total_shape_count > 50:
            page_result += ocr(clear_img_path, OCR, doc_type="doc",
                                     ignorePageHeaderAndFooter=ignorePageHeaderAndFooter)
        else:
            page_result += ocr(clear_img_path, paddleOCR, doc_type="doc",
                               ignorePageHeaderAndFooter=ignorePageHeaderAndFooter)
        os.remove(clear_img_path)

        # 对每个表格区域内的单元格进行识别
        for table_data in all_tables_data:
            tables = table_data['tables']
            t_info = table_data['info']
            img_res = t_info['image']
            x_offset = t_info['x_offset']
            y_offset = t_info['y_offset']
            rotation = t_info.get('rotation')
            orig_w = t_info.get('orig_w', 0)
            orig_h = t_info.get('orig_h', 0)

            for table_cells in tables:
                table_ocr_result = []
                for shape in table_cells:
                    x, y, w, h = list(map(int, shape))
                    # 使用裁剪后的 img_res 进行单元格识别
                    im = img_res[y:y + h, x:x + w]
                    if im is None or im.size == 0:
                        continue
                    im = cv2.copyMakeBorder(
                        im,
                        top=30, bottom=30, left=20, right=0,
                        borderType=cv2.BORDER_CONSTANT,
                        value=255
                    )
                    cell_result = ocr(im, table_OCR, x, y)
                    line_result = ""
                    chars = []
                    for i, line in enumerate(cell_result):
                        line_result += line["text"]
                        for ch in line["chars"]:
                            x1, y1, cw, ch_h = ch["bbox"]["x"], ch["bbox"]["y"], ch["bbox"]["w"], ch["bbox"]["h"]
                            _append_table_chars(chars, ch, x1, y1, cw, ch_h, rotation, orig_w, orig_h, x_offset, y_offset)
                    if rotation:
                        x_orig, y_orig, w_orig, h_orig = _unrotate_bbox(x, y, w, h, rotation, orig_w, orig_h)
                        cell_bbox = {"x": x_orig + x_offset, "y": y_orig + y_offset, "w": w_orig, "h": h_orig}
                    else:
                        cell_bbox = {"x": x + x_offset, "y": y + y_offset, "w": w, "h": h}
                    table_ocr_result.append({
                        "bbox": cell_bbox,
                        "chars": chars,
                        "row": "",
                        "col": "",
                        "text": line_result
                    })
                table_ocr_result = sorted(table_ocr_result, key=lambda item: item['bbox']['x'] + item['bbox']['y'] * 36)
                table_result.append(table_ocr_result)

        ocr_result.append({
            "page_result": page_result,
            "table_result": table_result
        })
        logger.info(f"Processing image spend: {time.time() - start_time}s")
    return ocr_result

def send_images(images, taskId):
    result_name = []
    for img_path in images:
        # 兼容 numpy 数组和文件路径
        img = img_path if not isinstance(img_path, str) else cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_name1 = f"/tmp/{uuid.uuid4()}.png"
        path = img_name1.split('/')[-1]
        object_name = f"{taskId}/images/{path}"
        cv2.imwrite(img_name1, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]) # type: ignore
        putObject(object_name, img_name1)
        result_name.append(object_name)
        os.remove(img_name1)
    return result_name


@app.route('/local_ocr', methods=['POST'])
def local_ocr():
    """本地文件上传 OCR 接口，绕过 S3。支持 office/pdf/image。"""
    if 'file' not in request.files:
        return jsonify({"code": 1, "msg": "缺少 file 字段"}), 400

    uploaded = request.files['file']
    if uploaded.filename == '':
        return jsonify({"code": 1, "msg": "未选择文件"}), 400

    ext = os.path.splitext(uploaded.filename)[1].lower()
    file_type_map = {
        '.docx': 'office', '.doc': 'office',
        '.pdf': 'imagePdf',
        '.png': 'image', '.jpg': 'image', '.jpeg': 'image',
        '.tiff': 'image', '.tif': 'image',
    }
    fileType = file_type_map.get(ext)
    if not fileType:
        return jsonify({"code": 1, "msg": f"不支持的文件类型: {ext}"}), 400

    local_path = f'/tmp/{uuid.uuid4()}{ext}'
    uploaded.save(local_path)

    # 可选参数
    forceImage = request.form.get('forceImage', 'true').lower() == 'true'
    ignoreSeals = request.form.get('ignoreSeals', 'true').lower() == 'true'
    ignorePageHeaderAndFooter = request.form.get('ignorePageHeaderAndFooter', 'false').lower() == 'true'
    ignoreContentTable = request.form.get('ignoreContentTable', 'false').lower() == 'true'
    detectOrientation = request.form.get('detectOrientation', 'true').lower() == 'true'
    removeWatermark = request.form.get('removeWatermark', 'false').lower() == 'true'
    pageRangeStr = request.form.get('pageRange', '')

    pageRange = None
    if pageRangeStr:
        try:
            parts = pageRangeStr.split(',')
            pageRange = [[int(parts[0]), int(parts[1])]]
        except:
            pass

    start = datetime.datetime.now()
    taskId = str(uuid.uuid4())[:8]
    app.logger.info(f"[local] {taskId} 开始, file={uploaded.filename}, ext={ext}, type={fileType}")

    try:
        # 1. 文件转图（本地版，绕过 S3）
        if fileType == 'image':
            images = [local_path]
        elif fileType == 'office':
            images = doc2imgList(local_path)
        else:  # pdf
            images, _ = pdf2images(local_path, pageRange=pageRange)

        # 2. OCR 管道
        if forceImage:
            start_time = time.time()
            img_result = images
            # ★ 转为内存管道：一次性读入 numpy 数组，后续全程内存传递
            img_result = [cv2.imread(p) for p in img_result]
            for p in images:
                try: os.remove(p)
                except: pass

            if fileType == 'imagePdf':
                img_result = removeHandwriting(img_result)
            if removeWatermark:
                img_result = remove_qrcode_process(img_result)
                img_result = watermark_removal_process(img_result)
            if detectOrientation:
                img_result = orientation_process(img_result)
            img_result, redmask_result = remove_redSeal_process(img_result, ignoreSeals)
            img_result, _, _ = remove_footer_header(img_result, ignorePageHeaderAndFooter)
            img_result = remove_num2_footer_header(img_result, ignorePageHeaderAndFooter)
            # --- debug: 保存传给OCR的图片 ---
            if DEBUG_SAVE_DIR:
                debug_pages = os.path.join(DEBUG_SAVE_DIR, taskId, "pages")
                os.makedirs(debug_pages, exist_ok=True)
                for di, dp in enumerate(img_result):
                    d_img = dp if not isinstance(dp, str) else cv2.imread(dp)
                    d_path = os.path.join(debug_pages, f"page{di+1:02d}.png")
                    cv2.imwrite(d_path, d_img)
                logger.info(f"[debug] 保存{len(img_result)}张页面图到 {debug_pages}")
            # --- end debug ---

            # 将 numpy 数组保存为临时文件，供 OCR 和上传使用
            temp_img_files = []
            for img_np in img_result:
                tpath = f"/tmp/{uuid.uuid4()}.png"
                cv2.imwrite(tpath, img_np)
                temp_img_files.append(tpath)

            # 保存宽高
            width_list = [img.shape[1] for img in img_result]
            height_list = [img.shape[0] for img in img_result]

            # OCR 使用本地临时文件
            result = ocr_proces(temp_img_files, ignorePageHeaderAndFooter)

            # 上传到对象存储
            start_time1 = time.time()
            keys = send_images(temp_img_files, taskId)
            # 清理临时文件
            for tp in temp_img_files:
                try: os.remove(tp)
                except: pass
            logger.info(f"[local] send images spend:{time.time() - start_time1:.1f}s")
            # --- debug: 裁剪OCR文本区域 ---
            if DEBUG_SAVE_DIR:
                def _safe_fn2(s):
                    return "".join(c for c in s if c.isalnum() or c in "._- ").strip()[:40]
                for ri, page_res in enumerate(result):
                    crop_dir = os.path.join(DEBUG_SAVE_DIR, taskId, "crops", f"page{ri+1:02d}")
                    os.makedirs(crop_dir, exist_ok=True)
                    src_img = img_result[ri] if not isinstance(img_result[ri], str) else cv2.imread(img_result[ri])
                    for li, line in enumerate(page_res.get("page_result", [])):
                        b = line.get("bbox", {})
                        x, y, w, h = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
                        if w <= 0 or h <= 0:
                            continue
                        pad = 4
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(src_img.shape[1], x + w + pad)
                        y2 = min(src_img.shape[0], y + h + pad)
                        crop = src_img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        safe_text = _safe_fn2(line.get("text", ""))
                        cpath = os.path.join(crop_dir, f"{li:03d}_{safe_text}.png")
                        cv2.imwrite(cpath, crop)
                logger.info(f"[debug] OCR裁剪图保存到 {os.path.join(DEBUG_SAVE_DIR, taskId, 'crops')}")
            # --- end debug ---
            logger.info(f"[local] ocr spend: {time.time() - start_time:.1f}s")

            if not redmask_result:
                redmask_result = [[] for _ in range(len(img_result))]

            office_result = []
            pages_result = list(range(pageRange[0][0], pageRange[0][1] + 1)) if pageRange else range(len(img_result))
            for i in range(len(img_result)):
                width = width_list[i]
                height = height_list[i]
                images_result = {
                    "key": keys[i],
                    "width": width,
                    "height": height
                }
                office_result.append({
                    "image": images_result,
                    "tables": result[i]["table_result"],
                    "lines": result[i]["page_result"],
                    "red_masks": redmask_result[i],
                    "page": pages_result[i] - 1 if pageRange else i
                })

            if ignoreContentTable:
                office_result = remove_directory(office_result)

            elapsed = (datetime.datetime.now() - start).total_seconds()
            app.logger.info(f"[local] {taskId} 完成, {elapsed:.1f}s")
            _clear_layout_cache()
            return jsonify({"code": 0, "data": office_result, "elapsed": elapsed})

        else:
            # 非 forceImage（如 office 直接返回图片坐标）
            office_result = []
            pages_result = list(range(pageRange[0][0], pageRange[0][1] + 1)) if pageRange else range(len(images))
            for i in range(len(images)):
                img_arr = images[i] if not isinstance(images[i], str) else cv2.imread(images[i])
                width = img_arr.shape[1]
                height = img_arr.shape[0]
                if isinstance(images[i], str):
                    os.remove(images[i])
                images_result = {
                    "key": "",
                    "width": width,
                    "height": height
                }
                office_result.append({
                    "image": images_result,
                    "page": pages_result[i] - 1 if pageRange else i
                })
            elapsed = (datetime.datetime.now() - start).total_seconds()
            return jsonify({"code": 0, "data": office_result, "elapsed": elapsed})

    except Exception as e:
        app.logger.error(f"[local] {taskId} 异常: {traceback.format_exc()}")
        return jsonify({"code": 1, "msg": str(e)}), 500
    finally:
        try:
            os.remove(local_path)
        except:
            pass


@app.route('/ocr_process', methods=['POST'])
def ocr_process():
    data = request.json
    fileType = data['fileType']
    taskId = data['taskId']
    fileUrl = data.get('fileKey') or data.get('originFileUrl') or data.get('targetFileUrl') or data.get('fileUrl') or data.get('file_key') or data.get('key') or data.get('url', '')
    forceImage = data['forceImage']
    ignoreSeals = data['ignoreSeals']
    ignorePageHeaderAndFooter = data['ignorePageHeaderAndFooter']
    ignoreContentTable = data['ignoreContentTable']
    detectOrientation = data.get('detectOrientation', True)
    removeWatermark = data.get('removeWatermark', True)
    pageRange = data.get('pageRange', [])
    fileExt = data.get('fileExt',"")
    start = datetime.datetime.now()
    formatted_time = start.strftime('%Y-%m-%d %H:%M:%S')
    app.logger.info(f"时间{formatted_time}-----{taskId}进入OCR任务")
    if not pageRange:
        pageRange = None
    pages_result = []
    if pageRange:
        for start, end in pageRange:
            pages_result.extend(range(start, end + 1))
    office_result = []
    footer_result = []
    header_result = []
    if fileType == "image":
        images = download_file(fileUrl)
    else:
        images, footer_result, header_result = download_and_convert_pdf_to_images(fileUrl, fileType, fileExt, pageRange)
    if fileType == "office" or fileType == "textPdf" or fileType == "imagePdf" or fileType == "image":
        if forceImage:
            start_time = time.time()
            if fileType == "image":
                img_result = [images]
            else:
                img_result = images
            # ★ 转为内存管道：一次性读入 numpy 数组，后续全程内存传递
            img_result = [cv2.imread(p) for p in img_result]
            for p in images if isinstance(images, list) else [images]:
                try: os.remove(p)
                except: pass
            if removeWatermark:
                img_result = remove_qrcode_process(img_result)
                img_result = watermark_removal_process(img_result)
            if fileType == "imagePdf":
                img_result = removeHandwriting(img_result)
            if detectOrientation:
                img_result = orientation_process(img_result)
            img_result, redmask_result = remove_redSeal_process(img_result, ignoreSeals)
            img_result, _, _ = remove_footer_header(img_result, ignorePageHeaderAndFooter)
            img_result = remove_num2_footer_header(img_result, ignorePageHeaderAndFooter)
            # --- debug: 保存传给前端和OCR的图片 ---
            if DEBUG_SAVE_DIR:
                debug_pages = os.path.join(DEBUG_SAVE_DIR, taskId, "pages")
                os.makedirs(debug_pages, exist_ok=True)
                for di, dp in enumerate(img_result):
                    d_img = dp if not isinstance(dp, str) else cv2.imread(dp)
                    d_path = os.path.join(debug_pages, f"page{di+1:02d}.png")
                    cv2.imwrite(d_path, d_img)
                logger.info(f"[debug] 保存{len(img_result)}张页面图到 {debug_pages}")
            # --- end debug ---
            # 将 numpy 数组保存为临时文件，供 OCR 和上传使用
            temp_img_files = []
            for img_np in img_result:
                tpath = f"/tmp/{uuid.uuid4()}.png"
                cv2.imwrite(tpath, img_np)
                temp_img_files.append(tpath)

            # 保存宽高
            width_list = [img.shape[1] for img in img_result]
            height_list = [img.shape[0] for img in img_result]

            # OCR 使用本地临时文件
            start_time = time.time()
            result = ocr_proces(temp_img_files, ignorePageHeaderAndFooter)

            # 上传到对象存储，获取 key 用于响应
            start_time1 = time.time()
            key = send_images(temp_img_files, taskId)
            # 清理临时文件
            for tp in temp_img_files:
                try: os.remove(tp)
                except: pass
            logger.info(f"send images spend:{time.time() - start_time1}s")
            # --- debug: 裁剪OCR文本区域 ---
            if DEBUG_SAVE_DIR:
                def _safe_fn(s):
                    return "".join(c for c in s if c.isalnum() or c in "._- ").strip()[:40]
                for ri, page_res in enumerate(result):
                    crop_dir = os.path.join(DEBUG_SAVE_DIR, taskId, "crops", f"page{ri+1:02d}")
                    os.makedirs(crop_dir, exist_ok=True)
                    src_img = img_result[ri] if not isinstance(img_result[ri], str) else cv2.imread(img_result[ri])
                    for li, line in enumerate(page_res.get("page_result", [])):
                        b = line.get("bbox", {})
                        x, y, w, h = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))
                        if w <= 0 or h <= 0:
                            continue
                        pad = 4
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(src_img.shape[1], x + w + pad)
                        y2 = min(src_img.shape[0], y + h + pad)
                        crop = src_img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        safe_text = _safe_fn(line.get("text", ""))
                        cpath = os.path.join(crop_dir, f"{li:03d}_{safe_text}.png")
                        cv2.imwrite(cpath, crop)
                logger.info(f"[debug] OCR裁剪图保存到 {os.path.join(DEBUG_SAVE_DIR, taskId, 'crops')}")
            # --- end debug ---
            logger.info(f"ocr spend:{time.time() - start_time}s")
            if not redmask_result:
                redmask_result = [[] for _ in range(len(img_result))]
            for i in range(len(img_result)):
                width = width_list[i]
                height = height_list[i]
                images_result = {
                    "key": key[i],
                    "width": width,
                    "height": height
                }
                if pageRange:
                    office_result.append({
                        "image": images_result,
                        "tables": result[i]["table_result"],
                        "lines": result[i]["page_result"],
                        "red_masks": redmask_result[i],
                        "page": pages_result[i] - 1
                    })
                else:
                    office_result.append({
                        "image": images_result,
                        "tables": result[i]["table_result"],
                        "lines": result[i]["page_result"],
                        "red_masks": redmask_result[i],
                        "page": i
                    })
            if ignoreContentTable:
                office_result = remove_directory(office_result)
            end = datetime.datetime.now()
            end_time = end.strftime('%Y-%m-%d %H:%M:%S')
            app.logger.info(f"时间{end_time}-----{taskId}完成OCR任务")
            _clear_layout_cache()
            return jsonify({"code": 0,
                            "data": office_result})
        elif not forceImage and fileType == "office":
            start_time1 = time.time()
            key = send_images(images, taskId)
            logger.info(f"send images spend:{time.time() - start_time1}s")
            for i in range(len(images)):
                width = cv2.imread(images[i]).shape[1]
                height = cv2.imread(images[i]).shape[0]
                os.remove(images[i])
                images_result = {
                    "key": key[i],
                    "width": width,
                    "height": height,
                }
                if pageRange:
                    office_result.append({"image": images_result,
                                          "page": pages_result[i] - 1
                                          })
                else:
                    office_result.append({"image": images_result,
                                          "page": i
                                          })
            end = datetime.datetime.now()
            end_time = end.strftime('%Y-%m-%d %H:%M:%S')
            app.logger.info(f"时间{end_time}-----{taskId}完成OCR任务")
            return jsonify({"code": 0,
                            "data": office_result})
        elif not forceImage:  # office, textPdf, imagePdf, image
            if isinstance(images, str):
                images = [images]
            if removeWatermark:
                images = remove_qrcode_process(images)
            start_time1 = time.time()
            key = send_images(images, taskId)
            length = range(len(images))
            width_list = []
            height_list = []
            for i in length:
                width = cv2.imread(images[i]).shape[1]
                height = cv2.imread(images[i]).shape[0]
                width_list.append(width)
                height_list.append(height)
            logger.info(f"send images spend:{time.time() - start_time1}s")
            img_result = images
            img_result, header_list, footer_list = remove_footer_header(img_result, ignorePageHeaderAndFooter)
            list_result = get_num2_footer_header(img_result, ignorePageHeaderAndFooter)
            try:
                for image in img_result:
                    os.remove(image)
            except:
                pass
            for i in length:
                try:
                    header = max(header_list[i], max(list_result[i]["header"], header_result[i]["header"]))
                except:
                    header = header_list[i]
                try:
                    footer = min(footer_list[i], min(list_result[i]["footer"], footer_result[i]["footer"]))
                except:
                    footer = footer_list[i]
                images_result = {
                    "key": key[i],
                    "width": width_list[i],
                    "height": height_list[i],
                }
                if pageRange:
                    office_result.append({
                        "image": images_result,
                        "header": header + 0.003,
                        "footer": footer - 0.006,
                        "page": pages_result[i] - 1
                    })
                else:
                    office_result.append({
                        "image": images_result,
                        "header": header + 0.003,
                        "footer": footer - 0.006,
                        "page": i
                    })
            end = datetime.datetime.now()
            end_time = end.strftime('%Y-%m-%d %H:%M:%S')
            app.logger.info(f"时间{end_time}-----{taskId}完成OCR任务")
            return jsonify({"code": 0,
                            "data": office_result})
    return None


if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
