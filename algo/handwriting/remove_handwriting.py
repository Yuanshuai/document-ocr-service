import cv2
import paddle
import numpy as np
import paddle.nn as nn


def preImage(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = img.transpose(2, 0, 1)[None, ...]  # NCHW
    return paddle.to_tensor(img)


def postprocess(tensor, orig_shape):
    img = tensor.squeeze(0).numpy().transpose(1, 2, 0)
    img = img[:orig_shape[0], :orig_shape[1], :]
    img = np.clip(img, 0, 1) * 255
    return cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)


def load_model(model_dir: str):
    if not hasattr(load_model, "_instance"):
        model = paddle.jit.load(model_dir)
        model.eval()
        load_model._instance = model
    return load_model._instance


def detect_clear_regions(img_gray):
    """
    检测清晰区域（高梯度区域）
    返回清晰区域的二值掩码
    """
    # 计算梯度幅值
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 归一化：防止除以 0 和出现 NaN/inf
    mag_min, mag_max = magnitude.min(), magnitude.max()
    mag_range = mag_max - mag_min
    if mag_range > 0:
        mag_norm = (magnitude - mag_min) / mag_range
    else:
        mag_norm = np.zeros_like(magnitude)
    mag_norm = np.nan_to_num(mag_norm)          # 把可能出现的 NaN/inf 强制置 0

    #  阈值处理
    _, clear_mask = cv2.threshold((mag_norm * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

    # 形态学后处理
    kernel = np.ones((3, 3), np.uint8)
    clear_mask = cv2.morphologyEx(clear_mask, cv2.MORPH_CLOSE, kernel)
    clear_mask = cv2.dilate(clear_mask, kernel, iterations=1)

    return clear_mask.astype(np.float32) / 255.0


def remove_handwriting(model, img_bgr):
    """
     只把顶部 0–12 % 的带状区域裁出来做手写去除，
     做完后再贴回去，保证其余像素 100 % 原样保留。
     """
    h0, w0 = img_bgr.shape[:2]
    top_h = int(h0 * 0.12)  # 12 % 高度
    if top_h == 0:
        return img_bgr  # 太小就跳过

    # 1. 裁出窄带
    top_band = img_bgr[:top_h, :, :]

    # 2. 对窄带做原来的滑窗推理（pad/restore 都在小图上进行）
    pad, step = 106, 300
    win = step + 2 * pad

    img_tensor = preImage(top_band)  # [1,3,top_h,w0]
    _, _, hb, wb = img_tensor.shape

    # 补齐到 706 整数倍
    pad_h = (win - hb % win) % win
    pad_w = (win - wb % win) % win
    img = nn.Pad2D((0, pad_w, 0, pad_h), mode='replicate')(img_tensor)
    _, _, h, w = img.shape

    # 外圈再 pad 106
    img = nn.Pad2D(pad, mode='replicate')(img)
    res = paddle.clone(img)

    # 滑窗只在窄带上做
    for y in range(0, h, step):
        for x in range(0, w, step):
            if y + win > h:
                y = h - win
            if x + win > w:
                x = w - win
            patch = img[:, :, y:y + win, x:x + win]

            # 清晰度掩码
            patch_np = patch.squeeze(0).numpy().transpose(1, 2, 0)
            patch_gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
            clear_mask = detect_clear_regions(patch_gray)
            clear_mask_tensor = paddle.to_tensor(
                clear_mask.astype(np.float32)).reshape((1, 1, win, win))

            with paddle.no_grad():
                out_patch = model(patch)

            # 只在清晰区域采用模型输出
            mixed_patch = out_patch * clear_mask_tensor + patch * (1 - clear_mask_tensor)
            res[:, :, y:y + win, x:x + win] = mixed_patch

    # 去掉 pad 并裁剪回原始窄带尺寸
    res = res[:, :, pad:-pad, pad:-pad]
    res = res[:, :, :hb, :wb]
    top_band_out = postprocess(res, (hb, wb))

    # 3. 贴回原图
    img_bgr = img_bgr.copy()
    img_bgr[:top_h, :, :] = top_band_out
    return img_bgr