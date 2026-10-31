import os
import random
import traceback
from avg_image import *
from black_detect import black_detect
from ocr_avg import ocr_avg
from line_cut import img_arr_highpass_filter
from paddleocr import PaddleOCR
import time

input_path = "Z:/包子/header_footer"
target_width = 480

paddleOCR = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)
dirs = os.listdir(input_path)
random.shuffle(dirs)
for dir in dirs:
    if "." in dir:
        continue
    result_list = []
    imgs = os.listdir(os.path.join(input_path, dir))
    imgs = [os.path.join(input_path, dir, img) for img in imgs]
    if len(imgs) < 8:
        continue
    try:
        origin, arr = avg_image(imgs)
    except:
        traceback.print_exc()
        continue
    target_height = int(target_width / arr.shape[1] * arr.shape[0])
    show = cv2.resize(arr, (target_width, target_height))
    start = time.time()
    result_list.append(img_arr_highpass_filter(arr))
    print(f"img_arr_highpass_filter:{time.time() - start}s")
    start = time.time()
    result_list.append(black_detect(arr))
    print(f"black_detect:{time.time() - start}s")
    start = time.time()
    result_list.append(ocr_avg(arr, paddleOCR))
    print(f"ocr_avg:{time.time() - start}s")
    header, footer = 0, arr.shape[0]
    for item in result_list:
        print(f"{item[2]}:header: {item[0]}, footer: {item[1]}")
        header = max(header, item[0])
        footer = min(footer, item[1])
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    cv2.line(arr, (0, int(header * len(arr))), (arr.shape[1], int(header * len(arr))), (255, 0, 0), 3)
    cv2.line(arr, (0, int(footer * len(arr))), (arr.shape[1],
                                                int(footer * len(arr))), (255, 0, 0), 3)
    target_height = int(target_width / show.shape[1] * show.shape[0])
    origin = cv2.cvtColor(origin, cv2.COLOR_GRAY2BGR)
    cv2.line(origin, (0, int(header * len(arr))), (origin.shape[1], int(header * len(arr))), (255, 0, 0), 3)
    cv2.line(origin, (0, int(footer * len(arr))), (origin.shape[1], int(footer * len(arr))), (255, 0, 0), 3)
    cv2.putText(origin, f"imgs_len:{len(imgs)}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
    target_height = int(target_width / origin.shape[1] * origin.shape[0])
    origin = cv2.resize(origin, (target_width, target_height))
    cv2.imshow("origin", origin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"header: {header}, footer: {footer}")
