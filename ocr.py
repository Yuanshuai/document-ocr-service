import re
import numpy as np
from shared_state import shared_state
import math

punctuation_list = "_·?　—－丨-__——‘:!+=,%^：，,＂“”''‘’？、~@#%……&*《》<>〔〕[]「」{}【】（）()'\"°；丶;●個飲產間東黃電〇冏檸標銀頔適"


def _append_char_tokens(chars, ch, box, score, offset_x, offset_y):
    if not ch:
        return

    pts = np.array(box)
    cx, cy = int(pts[:, 0].min()), int(pts[:, 1].min())
    cw, chh = int(pts[:, 0].max()) - cx, int(pts[:, 1].max()) - cy
    final_score = score * 100 if score <= 1.0 else score
    single_chars = list(ch)

    if len(single_chars) == 1:
        chars.append({
            "bbox": {"x": int(cx + offset_x), "y": int(cy + offset_y), "w": cw, "h": chh},
            "char": ch,
            "score": int(final_score)
        })
        return

    step = cw / len(single_chars)
    for idx, single_char in enumerate(single_chars):
        x0 = cx + step * idx
        x1 = cx + step * (idx + 1)
        chars.append({
            "bbox": {
                "x": int(x0 + offset_x),
                "y": int(cy + offset_y),
                "w": max(1, int(x1) - int(x0)),
                "h": chh
            },
            "char": single_char,
            "score": int(final_score)
        })


def ocr(img_path: str, paddleOCR, offset_x=0, offset_y=0, doc_type: str = "",
        ignorePageHeaderAndFooter=None) -> list:
    try:
        with shared_state.gpu_lock:
            result = paddleOCR.predict(img_path)
    except Exception as e:
        print(f"OCR predict error: {e}")
        return []

    if not result or not result[0].get("rec_texts"):
        return []

    rec_texts = result[0].get('rec_texts', [])
    rec_scores = result[0].get('rec_scores', [])
    text_word = result[0].get('text_word', [])
    text_word_region = result[0].get('text_word_region', [])
    text_word_score = result[0].get('text_word_score', [])
    rec_polys = result[0].get('rec_polys', [])

    data_list = []
    for idx, (line_text, line_score, line_chars, line_char_boxes, line_poly) in enumerate(
            zip(rec_texts, rec_scores, text_word, text_word_region, rec_polys)):
        if not line_text:
            continue
            
        # 确定当前行的字符分数
        if text_word_score and idx < len(text_word_score):
            current_line_char_scores = text_word_score[idx]
        else:
            current_line_char_scores = [line_score] * len(line_chars)
        
        pts = np.array(line_poly, dtype=np.int16)
        xs, ys = pts[:, 0], pts[:, 1]
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
        chars = []
        for ch, box, score in zip(line_chars, line_char_boxes, current_line_char_scores):
            _append_char_tokens(chars, ch, box, score, offset_x, offset_y)

        for c in chars:
            if c["char"] in punctuation_list:
                chars.remove(c)

        to_del = []
        for i in range(len(chars)):
            if chars[i]["char"] == " ":
                prev = chars[i - 1]["char"] if i > 0 else ""
                nxt = chars[i + 1]["char"] if i < len(chars) - 1 else ""
                if not prev.isalpha() or nxt.isalpha():
                    to_del.append(chars[i])
                continue
            if chars[i]["char"] == ".":
                prev = chars[i - 1]["char"] if i > 0 else ""
                nxt = chars[i + 1]["char"] if i < len(chars) - 1 else ""
                if not prev.isdigit() or nxt.isdigit():
                    if i > 0 and chars[i - 1]["char"] == ".":
                        to_del.append(chars[i])
                        continue
                    to_del.append(chars[i])
        for c in to_del:
            chars.remove(c)

        i = 0
        while i < len(chars) - 1:
            if chars[i]["char"] == "." and chars[i + 1]["char"] == ".":
                chars[i]["bbox"]["w"] += chars[i + 1]["bbox"]["w"]
                chars.pop(i + 1)
            elif chars[i]["char"] == "m" and chars[i + 1]["char"] == "3":
                chars.pop(i + 1)
            else:
                i += 1

        if chars:
            last = chars[-1]
            chars.append({
                "bbox": {"x": last["bbox"]["x"] + last["bbox"]["w"], "y": last["bbox"]["y"],
                         "w": 2, "h": last["bbox"]["h"]},
                "char": "\n",
                "score": last["score"]
            })
            line_text += "\n"

        data_list.append({
            "bbox": {"x": x + offset_x, "y": y + offset_y, "w": w, "h": h},
            "chars": chars,
            "line": idx,
            "text": line_text
        })

    skip_keywords = ["深圳市政", "扫描全能王", "3亿人都在用"]
    if len(data_list) <= 3:
        data_list = [it for it in data_list if not any(kw in it["text"] for kw in skip_keywords)]
    else:
        data_list[:-3] = [it for it in data_list[:-3] if not any(kw in it["text"] for kw in skip_keywords)]
        data_list[-3:] = [it for it in data_list[-3:] if not any(kw in it["text"] for kw in skip_keywords)]

    if ignorePageHeaderAndFooter and doc_type == "doc" and data_list:
        last_text = data_list[-1]["text"].rstrip("\n")
        if last_text.isdigit():
            data_list.pop()
        else:
            patterns = [
                r'^第\s*\d+\s*页\s*(?:共\s*[\[【]?\d+[\]】]?\s*页)?$',
                r'^\d+\s*/\s*\d+$',
                r'[—-]?\s*\d+\s*[—-]?',
                r'^\d+\s*\uFF0F\s*\d+$',
            ]
            for p in patterns:
                if re.fullmatch(p, last_text):
                    data_list.pop()
                    break

    return data_list


def ocr_cell(im, paddleOCR) -> str:
    try:
        with shared_state.gpu_lock:
            result = paddleOCR.predict(im)
    except Exception as e:
        print(f"OCR cell predict error: {e}")
        return ""

    if not result:
        return ""
    rec_texts = result[0].get("rec_texts", [])
    if not rec_texts:
        return ""
    return "".join(rec_texts)


def ocr_text(img_path: str, paddleOCR):
    try:
        with shared_state.gpu_lock:
            result = paddleOCR.predict(img_path)
    except Exception as e:
        print(f"OCR text predict error: {e}")
        return [], []

    if not result:
        return [], []
    scores = result[0].get("rec_scores", [])
    texts = result[0].get("rec_texts", [])
    return texts, scores
