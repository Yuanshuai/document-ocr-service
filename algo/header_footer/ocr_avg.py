import os
import re
import copy
import uuid
import time
import numpy as np
from PIL import Image
from ocr import ocr
from config import hot_area


def ocr_avg(arr: np.array, paddleOCR) -> list:
    start_time = time.time()
    header, footer = 0, len(arr) - 1
    arr = copy.copy(arr)
    arr[int(0.25 * len(arr)):int(0.75 * len(arr)), :] = 255
    img = Image.fromarray(np.array(np.round(arr), dtype=np.uint8))
    file_name = f"/tmp/{uuid.uuid4()}.jpg"
    img.save(file_name)
    result = ocr(file_name, paddleOCR)
    os.remove(file_name)
    if len(result) > 7:
        return [0, 1, "ocr_avg_failed"]
    print(f"result: {result}")
    total_text = ""
    for item in result:
        total_text += item['text']
        if len(total_text) > 10:
            return [0, 1, "ocr_avg_failed"]
    for item in result:
        text = re.sub(r'[^\u4e00-\u9fff]+', '', item['text'])
        if len(text) <= 2:
            continue
        upline = hot_area * len(arr)
        downline = len(arr) - upline
        top = item["bbox"]['y']
        bottom = item["bbox"]['y'] + item["bbox"]['h']
        if bottom < upline:
            header = max(header, bottom)
            try:
                total_chars = len(result[0]['text'])
                if total_chars > 25:
                    header = 0
            except:
                pass
        if top > downline:
            footer = min(footer, top)
    spend_time = time.time() - start_time
    print("ocr_avg spend time: ", spend_time)
    if header / len(arr) > 0.10:
        header = 0
    return [header / len(arr), footer / len(arr), "ocr_avg"]
