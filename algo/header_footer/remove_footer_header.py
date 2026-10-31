import re
import cv2
import pdfplumber
from ocr import ocr_cell
from config import OCR
from shared_state import shared_state


def rm_footer_header(img_path, structure, boxes=None):
    if boxes is None:
        with shared_state.structure_lock:
            result = structure.predict(img_path, layout_nms=True)
        boxes = result[0]['boxes']
    img = img_path if not isinstance(img_path, str) else cv2.imread(img_path)
    h_img, w_img = img.shape[:2]

    for box in boxes:
        cls_name = box['label']
        x1, y1, x2, y2 = box['coordinate']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if cls_name in ('header', 'header_image'):
            cv2.rectangle(img, (x1, 0), (x2, y2), (255, 255, 255), -1)
        elif cls_name in ('footer', 'footer_image', 'footnote', 'vision_footnote'):
            cv2.rectangle(img, (0, y1), (w_img, h_img), (255, 255, 255), -1)

    if isinstance(img_path, str):
        cv2.imwrite(img_input, img)
    return img

def get_rm_footer_header(img, structure):
    with shared_state.structure_lock:
        result = structure.predict(img, layout_nms=True)
    boxes = result[0]['boxes']
    header_y2s, footer_y1s = [], []
    h_img = img.shape[0]
    for box in boxes:
        cls_name = box['label']
        x1, y1, x2, y2 = map(int, box['coordinate'])
        if cls_name == 'header':
            y1_after = max(y1 - 10, 0)
            roi = img[y1_after: y2 + 10, 0: x2]
            text = ocr_cell(roi, OCR)
            keywords = ["编号", "附表", "盖章", "签字", "无正文", "证明书", "签署", "目录", "示范"]
            if ("附" in text and "件" in text) or any(k in text for k in keywords):
                continue
            header_y2s.append(y2)
        elif cls_name == 'footer':
            footer_y1s.append(y1)

    # 3. 计算比例
    header_proportion = min(header_y2s) / h_img if header_y2s and min(header_y2s) <= 0.12 * h_img else 0
    footer_proportion = max(footer_y1s) / h_img if footer_y1s and max(footer_y1s) >= 0.88 * h_img else 1

    return {'header': header_proportion, 'footer': footer_proportion}


def get_header_footer(pdf_path, pageRange=None):
    footer_text_patterns = [
        r'^第\s*\d+\s*页\s*(?:共\s*[\[【]?\d+[\]】]?\s*页)?$',
        r'^\d+\s*/\s*\d+$',
        r'[—-]?\s*\d+\s*[—-]?',
        r'^第\s*\d+\s*页\s*/\s*共\s*\d+\s*页$'
    ]

    header_result = []
    footer_result = []
    header_data = []
    footer_data = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        if (pageRange is None and total_pages == 1) or \
           (pageRange and len(pageRange) == 1 and pageRange[0][0] == pageRange[0][1] == 1):
            process_header = False
        else:
            process_header = True

        for page_num, page in enumerate(pdf.pages, start=1):
            # 如果指定了page_range，且当前页不在范围内，跳过
            if pageRange and not any(start <= page_num <= end for start, end in pageRange):
                continue

            page_height = page.height

            # ========== 页脚处理 ==========
            text_lines = page.extract_text_lines()
            footer_ratio = 1

            if text_lines:
                last_line = text_lines[-1]
                last_text = last_line['text'].strip()

                # 检查是否符合页脚特征
                if any(re.fullmatch(p, last_text) for p in footer_text_patterns):
                    # 检查是否有其他文本行的 y 坐标与 last_line['top'] 在 10 以内
                    is_alone = True
                    for line in text_lines[:-1]:  # 遍历除最后一行外的其他行
                        if abs(line['top'] - last_line['top']) <= 10:
                            is_alone = False
                            break

                    if is_alone:
                        footer_ratio = last_line['top'] / page_height
                        if footer_ratio <= 0.8:
                            footer_ratio = 1
                    else:
                        footer_ratio = 1

            footer_data.append(footer_ratio)

            # ========== 页眉收集 ==========
            if process_header:
                header_text = None
                header_bottom = None
                if text_lines:
                    first_line = text_lines[0]
                    header_text = first_line['text'].strip()
                    header_bottom = first_line['bottom']

                header_data.append({
                    'text': header_text,
                    'bottom': header_bottom,
                    'height': page_height
                })
            else:
                header_data.append({'text': None, 'bottom': 0, 'height': page_height})

        # 如果不处理页眉，直接返回页脚结果
        if not process_header:
            header_result = [{"header": 0} for _ in footer_data]
            footer_result = [{'footer': f} for f in footer_data]
            return header_result, footer_result

        # ========== 页眉统一性校验 ==========
        # 获取有效的页眉基准文本（排除None和空字符串）
        valid_headers = [h['text'] for h in header_data if h and h['text']]
        common_header = valid_headers[0] if valid_headers else None

        # 检查所有有效页眉是否一致
        if common_header:
            for h in header_data[1:]:
                if h and h['text'] != common_header:
                    common_header = None
                    break

        # ========== 构建最终结果 ==========
        for idx in range(len(header_data)):
            # 处理页眉
            header_ratio = 0
            if common_header and header_data[idx] and header_data[idx]['text'] == common_header:
                header_ratio = header_data[idx]['bottom'] / header_data[idx]['height']
                if header_ratio >=0.12:
                    header_ratio = 0

            header_result.append({"header": header_ratio})
            footer_result.append({'footer': footer_data[idx]})
    return header_result, footer_result
