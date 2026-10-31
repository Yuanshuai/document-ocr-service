import cv2
import random
import numpy as np
from shared_state import shared_state


def safe_show(name, image):
    resize_height = 1600
    resize_width = 900
    if image.shape[0] > resize_height:
        rate = max(image.shape[0] / resize_height, image.shape[1] / resize_width)
        image = cv2.resize(image, (int(image.shape[1] / rate), int(image.shape[0] / rate)))
    cv2.imshow(name.encode('gbk').decode(errors='ignore'), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class BBox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __str__(self):
        return f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


def get_table_structure(tables):
    """获取每个单元格的行列信息，处理跨行跨列"""
    if not tables:
        return []

    # 提取所有x和y坐标
    x_coords = sorted(list(set([t[0] for t in tables])))
    y_coords = sorted(list(set([t[1] for t in tables])))

    def merge_close_coords(coords, threshold=13):
        if not coords:
            return []
        merged = [coords[0]]
        for coord in coords[1:]:
            if coord - merged[-1] <= threshold:
                continue
            merged.append(coord)
        return merged

    x_coords = merge_close_coords(x_coords)
    y_coords = merge_close_coords(y_coords)

    # 获取总行列数
    total_rows = len(y_coords)
    total_cols = len(x_coords)
    print(f"\n表格结构：{total_rows}行 x {total_cols}列")

    cells = []
    # 用于去重的集合
    unique_cells = set()

    for table in tables:
        x, y, w, h = table

        # 找到单元格跨越的所有行和列
        spanned_rows = []
        spanned_cols = []

        # 检查跨越的行
        for i, y_pos in enumerate(y_coords):
            if y <= y_pos + 13 and y_pos <= y + h:
                spanned_rows.append(i)

        # 检查跨越的列
        for i, x_pos in enumerate(x_coords):
            if x <= x_pos + 13 and x_pos <= x + w:
                spanned_cols.append(i)

        # 为每个跨越的行列组合创建单元格记录
        for row_idx in spanned_rows:
            for col_idx in spanned_cols:
                # 创建唯一标识
                cell_key = (row_idx, col_idx, x, y, w, h)
                if cell_key not in unique_cells:
                    unique_cells.add(cell_key)
                    cells.append({
                        'row': row_idx,
                        'col': col_idx,
                        'bbox': BBox(x, y, x + w, y + h)
                    })

    # 按行和x坐标排序
    cells.sort(key=lambda x: (x['row'], x['bbox'].x1))

    # 输出格式化结果
    result = []
    for cell in cells:
        result.append({
            'row': cell['row'],
            'bbox': cell['bbox']
        })

    return result
def image2tables(image) -> list:
    """

    表格检测主函数：
    1. 如果最大外框占据了整个图像并且包含了所有其他框，则剔除该外框
    2. 正确判断包含关系，保留内部真实表格
    3. 一张图只返回一个表格分组
    """
    # 1. 预处理：二值化 + 形态学操作
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, -10)
    rows, cols = binary.shape
    scale = 10

    # 膨胀增强表格线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 2. 分别提取横线和竖线
    # 提取横线
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    dilatedcol = cv2.dilate(cv2.erode(binary, kernel_h, iterations=1), kernel_h, iterations=1)

    # 提取竖线
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    dilatedrow = cv2.dilate(cv2.erode(binary, kernel_v, iterations=1), kernel_v, iterations=1)

    # 3. 合并横竖线，生成表格骨架
    merge = cv2.add(dilatedcol, dilatedrow)
    merge = cv2.dilate(merge, np.ones((5,5), np.uint8), iterations=1)

    # 4. 查找轮廓并过滤无效轮廓
    contours, _ = cv2.findContours(merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_shape = []
    img_area = image.shape[0] * image.shape[1]

    print(f"\n🌀 原始轮廓数量：{len(contours)} 个")

    for item in contours:
        x, y, w, h = cv2.boundingRect(item)
        area_ratio = w * h / img_area
        # 过滤极小轮廓（噪点）
        if area_ratio < 0.0003:
            continue
        # 过滤细长条轮廓（非单元格）
        if (w // h > 6 or h // w > 6) and area_ratio < 0.001448:
            continue
        if (w // h > 15 or h // w > 15) and area_ratio < 0.0018:
            continue
        if w // h > 36 or h // w > 36:
            continue
        contours_shape.append([x, y, w, h])

    print(f"✨ 过滤后有效候选框：{len(contours_shape)} 个")

    # 5. 核心逻辑：判断并剔除最大的外框
    if len(contours_shape) >= 1:
        # 按面积从大到小排序
        contours_shape_sorted = sorted(contours_shape, key=lambda b: b[2]*b[3], reverse=True)
        max_box = contours_shape_sorted[0]
        mx, my, mw, mh = max_box
        
        # 判断最大框是否占据了大部分图像面积（例如 > 90%）
        is_full_img = (mw * mh) / img_area > 0.9
        
        # 判断最大框是否包含了所有其他的框
        contains_all = True
        for i in range(1, len(contours_shape_sorted)):
            x, y, w, h = contours_shape_sorted[i]
            if not (x >= mx and y >= my and (x+w) <= (mx+mw) and (y+h) <= (my+mh)):
                contains_all = False
                break
        
        if is_full_img and contains_all:
            contours_shape = contours_shape_sorted[1:]
            print(f"🗑️  最大外框占据全图且包含所有子框，已剔除。剩余候选框：{len(contours_shape)} 个")
        else:
            contours_shape = contours_shape_sorted
            print(f"✅ 最大外框未满足剔除条件，予以保留。剩余候选框：{len(contours_shape)} 个")
    else:
        print("⚠️  无有效候选框，直接返回空")
        return []

    # 6. 剔除被包含的小框（正确的包含判断逻辑）
    tables = []
    for idx, shape in enumerate(contours_shape):
        x, y, w, h = shape
        idx_show = idx + 1

        # 过滤超大框（超出常规表格尺寸）
        if w * h > 1080 * 2600:
            print(f"   框{idx_show} → ❌ 剔除（面积超大）")
            continue

        contain = False
        contain_by_idx = -1

        # 检查当前框是否被其他框包含
        for idx1, s1 in enumerate(contours_shape):
            if idx == idx1:  # 跳过自身
                continue
            x1, y1, w1, h1 = s1

            # 正确的包含判断：当前框 是否被 其他框 包含
            is_contain = (x >= x1 and
                          y >= y1 and
                          (x + w) <= (x1 + w1) and
                          (y + h) <= (y1 + h1))

            if is_contain:
                contain = True
                contain_by_idx = idx1 + 1
                break

        if contain:
            print(f"   框{idx_show} → ❌ 剔除（被框{contain_by_idx}包含）")
        else:
            # print(f"   框{idx_show} → ✅ 保留")
            tables.append(shape)

    # 7. 最终处理：一张图只返回一个表格分组
    if not tables:
        # print("\n📋 最终检测结果：无有效表格")
        return []
    else:
        # 所有保留的框作为一个表格分组返回
        grouped_tables = [tables]
        return grouped_tables


def image2big_tables(image, min_table_area=2000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = [cnt for cnt in contours if cv2.contourArea(cnt) > min_table_area]
    table_coordinates = []
    for table in tables:
        x, y, w, h = cv2.boundingRect(table)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        table_coordinates.append((x1, y1, x2, y2))
    return table_coordinates


def is_table(img_path, structure):
    with shared_state.structure_lock:
        result = structure.predict(img_path, layout_nms=True)
    for box in result[0].get('boxes', []):
        if box.get('label') == 'table':
            return True
    return False


if __name__ == "__main__":
    dir_path = "C://tmp"
    import os

    files = os.listdir(dir_path)
    random.shuffle(files)
    for file in files:
        ext = file.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png", "bmp"]:
            image = cv2.imdecode(np.fromfile(f"{dir_path}/{file}", dtype=np.uint8), -1)
        else:
            continue
        tables = image2tables(image)
        for table in tables:
            x, y, w, h = table
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
        safe_show(file, image)
