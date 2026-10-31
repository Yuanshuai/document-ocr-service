import cv2
import os
import copy
import numpy as np
from config import slope_width


def avg_image_gray(imgs: list) -> np.array:
    # 辅助函数：支持中文路径读取
    def cv2_imread(file_path, flags=cv2.IMREAD_GRAYSCALE):
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)

    img1 = cv2_imread(imgs[0], 0)
    if img1 is None:
        raise Exception(f"Failed to read image: {imgs[0]}")
        
    img1_height = img1.shape[0]
    img1_width = img1.shape[1]
    img_size_same = True
    for img_path in imgs:
        img = cv2_imread(img_path, 0)
        if img is None:
            continue
            
        h_diff = abs(img.shape[0] - img1_height) / img1_height
        w_diff = abs(img.shape[1] - img1_width) / img1_width
        if max(h_diff, w_diff) > 0.04:
            img_size_same = False
            break
        if img.shape[0] != img1_height or img.shape[1] != img1_width:
            img = cv2.resize(img, (img1_width, img1_height))
            # 注意：imencode 保存中文路径也需要特殊处理，但这里原逻辑是覆盖原文件？
            # 暂时保持原逻辑 imwrite，如果报错可能需要修改
            # cv2.imwrite(img_path, img) 
            # 考虑到中文路径写入也会有问题，这里改为 imencode
            cv2.imencode(os.path.splitext(img_path)[1], img)[1].tofile(img_path)

    if not img_size_same:
        # 为了调试方便，不再直接抛异常，而是尝试跳过尺寸不一致的图（或者你可以选择抛出）
        # 这里维持原逻辑抛出异常
        raise Exception(f"avg_image: img size not match {img1_height}*{img1_width}")
    target_width = min(img1_width, slope_width)
    target_height = int(img1_height * target_width / img1_width)
    final_arr = np.zeros((target_height, target_width), np.uint32)
    
    count = 0
    for img_path in imgs:
        img = cv2_imread(img_path, 0)
        if img is None: continue
        
        img = cv2.resize(img, (target_width, target_height))
        final_arr = final_arr + img
        count += 1
        
    if count > 0:
        final_arr = final_arr / count
    final_arr = final_arr.astype(np.uint8)
    return final_arr


def avg_image(imgs: list) -> np.array:
    try:
        origin = avg_image_gray(imgs)
        final_arr = copy.copy(avg_image_gray(imgs))
    except Exception as e:
        raise e
    final_arr = final_arr[:, int(final_arr.shape[1] * 0.2): int(final_arr.shape[1])]
    avg = np.mean(final_arr) * 0.82
    print(f"avg {avg}")
    final_arr[final_arr < avg * 0.9] = 0
    final_arr[final_arr >= avg * 1.1] = 255
    return origin, final_arr
