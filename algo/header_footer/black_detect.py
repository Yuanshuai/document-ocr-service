import cv2
import numpy as np
from config import hot_area

core_length = 0.03  # 卷积核长度
core_height = 0.012  # 卷积核高度
throttle = 4  # 采信阀门


def line_max(line: list) -> int:
    min_core = 255
    line_len = line.shape[1]
    block_length = int(core_length * line_len)
    for i in range(0, line_len - block_length):
        core = np.mean(line[:, i:i + block_length])
        min_core = min(min_core, core)
    return 255 - min_core


# 黑区识别
def black_detect(arr) -> list:
    arr = cv2.resize(arr, (1273, 900))
    arr_core_height = int(core_height * len(arr))
    avg = (255 - np.mean(arr) + arr_core_height) * throttle
    headder = 0
    footer = len(arr) - 1
    img_height = len(arr)
    for y in range(0, int(hot_area * img_height)):
        u_avg = line_max(arr[y:y + arr_core_height])
        if u_avg > avg:
            headder = y
        yd = img_height - y
        d_avg = line_max(arr[yd - arr_core_height:yd])
        if d_avg > avg:
            footer = img_height - y - 1
    if headder > 0.10:
        headder = 0
    if footer < len(arr) * 0.99:
        footer -= 0.005 * len(arr)
    if footer < len(arr) * 0.89:
        footer = len(arr)
    return [headder / len(arr), footer / len(arr), "black_detect"]
