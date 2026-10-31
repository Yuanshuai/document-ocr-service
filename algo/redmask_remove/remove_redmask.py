import os
import cv2
import numpy as np


def infer_start(img_path, ignoreSeals=False, img_array=None):
    from config import yolo_model, redmaskRec_model
    """
    YOLO11 检测印章 → RedMaskRec 识别文字（4 方向旋转取最高置信度）。
    支持文件路径或 numpy 数组。
    """
    result_bbox = []
    if img_array is not None:
        frame = img_array
        detections = yolo_model.detect_objects(img_array)
    else:
        frame = cv2.imread(img_path)
        detections = yolo_model.detect_objects(img_path)

    if not detections:
        return []

    # 旋转映射：在 BGR 图像上进行 cv2.rotate
    _ROTATE_MAP = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    for bbox in detections:
        x_min, y_min, w, h = bbox
        if x_min == 0 and y_min == 0:
            continue

        if not ignoreSeals:
            seal_image = frame[int(y_min):int(y_min + h), int(x_min):int(x_min + w)]

            best_text, best_score = "", 0
            for angle in [0, 90, 180, 270]:
                rotated = seal_image if angle == 0 else cv2.rotate(seal_image, _ROTATE_MAP[angle])
                # redmaskRec_model.run 期望 RGB 输入
                text, scores = redmaskRec_model.run(rotated[..., ::-1])
                avg_score = float(np.mean(scores)) if scores else 0.0
                if avg_score > best_score:
                    best_score, best_text = avg_score, text

            res_message = best_text
        else:
            res_message = "注意印章检查！"

        result_bbox.append({
            "bbox": {
                "x": int(x_min),
                "y": int(y_min),
                "w": int(w),
                "h": int(h),
            },
            "text": res_message,
        })

    return result_bbox


def remove_red_stamp(image, regions):
    """
    基于 HSV 色域去除红色印章像素（向量化版本）。
    将印章区域内的红色像素替换为白色。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    for region in regions:
        x, y, w, h = region
        if w == 0 and h == 0:
            continue
        # 向量化替换：mask==255 的像素 → 白色
        region_mask = mask[y:y + h, x:x + w]
        roi = image[y:y + h, x:x + w]
        roi[region_mask == 255] = [255, 255, 255]
        image[y:y + h, x:x + w] = roi
    return image


def remove_red_seal(img, regions):
    """
    基于 R 通道固定阈值去除红色印章像素。
    逻辑：R 通道阈值 → 二值化 → 亮度校验保护文字 → 替换为背景色。
    不做任何形态学操作，避免掩膜扩散到非红色区域。
    """
    result = img.copy()
    if regions == [[0, 0, 0, 0]]:
        return result
    for region in regions:
        x, y, w, h = region
        if w == 0 and h == 0:
            continue
        roi = result[y:y + h, x:x + w]
        B, G, R = cv2.split(roi)

        # 宽松红色判定：R 通道足够亮 + R 相对于 G/B 有优势
        raw_mask = (R > 120) & (R - G > 15) & (R - B > 15)

        # 亮度校验：黑色文字（灰度 < 80）保留
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        is_dark_text = gray < 80

        # 暗红保护：红色像素中 R 值最低的 3%（文字与红章叠加区域）保留原值
        # 纯红印章 R≈240~254，叠加区域 R≈120~200，用百分位数自适应分割
        if raw_mask.any():
            r_threshold = np.percentile(R[raw_mask], 3)
            is_dark_red = raw_mask & (R < r_threshold)
        else:
            is_dark_red = np.zeros_like(raw_mask, dtype=bool)

        # 估算背景色：ROI 中非红、非暗色像素的中位数
        bg_mask = (~raw_mask) & (~is_dark_text)
        if bg_mask.any():
            bg_color = np.median(roi[bg_mask], axis=0).astype(np.uint8)
        else:
            bg_color = np.array([255, 255, 255], dtype=np.uint8)

        # 替换：纯红印章像素 → 背景色；黑文字和暗红叠加区域 → 保留
        roi[raw_mask & (~is_dark_text) & (~is_dark_red)] = bg_color
        result[y:y + h, x:x + w] = roi

    return result


def process_red_seals(img_path, ignoreSeals=False):
    """
    检测并逐个去除红色印章。
    每检测到一个印章，立即识别文字并去除该区域的红色像素，再处理下一个。
    这是去除红章的默认入口函数。

    Args:
        img_path: 图片文件路径
        ignoreSeals: True 则跳过文字识别，仅去除红色像素

    Returns:
        cleaned_img: 去除印章后的图像 (BGR numpy array)
        stamp_info: [{bbox: {x, y, w, h}, text: str}, ...]
    """
    img = cv2.imread(img_path)
    stamp_info = infer_start(img_path, ignoreSeals)

    if stamp_info:
        # 逐个提取 bbox 区域，调用 remove_red_seal 进行像素级替换
        regions = [[r['bbox']['x'], r['bbox']['y'], r['bbox']['w'], r['bbox']['h']] for r in stamp_info]
        img = remove_red_seal(img, regions)

    return img, stamp_info
