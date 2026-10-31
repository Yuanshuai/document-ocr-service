# -*- coding: utf8 -*-
import io
import os
import cv2
import uuid
import math
import pikepdf
import pymupdf
import numpy as np
from PIL import Image


def remove_annotations(input_pdf_path):
    output_pdf_path = f"/tmp/{uuid.uuid4()}.pdf"
    with pikepdf.open(input_pdf_path) as pdf:
        for page in pdf.pages:
            if '/Annots' in page:
                del page['/Annots']

        pdf.save(output_pdf_path)
    return output_pdf_path


def doc2imgList(doc_path: str, quality=2.0, pageRange=None):
    try:
        pages = pymupdf.open(doc_path)
    except:
        return []
    files = []
    if pageRange is None:
        pageRange = [[1, len(pages)]]
    for start, end in pageRange:
        start = max(1, start)
        end = min(len(pages), end)

        for page_num in range(start - 1, end):
            page = pages[page_num]
            mat = pymupdf.Matrix(quality, quality)
            pix = page.get_pixmap(matrix=mat)
            pix_name = f"/tmp/{uuid.uuid4()}.png"
            pix.save(pix_name, output="PNG")
            files.append(pix_name)
    return files


def is_textPdf(pdf_path: str) -> bool:
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    full_text = "".join(full_text.split())

    if full_text:
        doc.close()
        return True

    for page in doc:
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height

        img_list = page.get_image_info(xrefs=True)
        if len(img_list) >= 2:
            doc.close()
            return True

        for img in img_list:
            try:
                bbox = page.get_image_bbox(img)
                img_area = bbox.width * bbox.height
                if img_area >= 0.95 * page_area:
                    doc.close()
                    return True
            except Exception:
                continue

    doc.close()
    return False


def extract_single_image_per_page(pdf_path: str, pageRange=None):
    """
    每页只提取真正被绘制的那1张图片，2×精度保存为 png。
    返回 [path1, path2, ...]
    """
    imageList = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return []

    if pageRange is None:
        pageRange = [[1, len(doc)]]

    for start, end in pageRange:
        start = max(1, start)
        end = min(len(doc), end)
        for page_idx in range(start - 1, end):
            page = doc[page_idx]

            image_info_list = page.get_image_info(xrefs=True)
            if not image_info_list:
                continue

            img_info = image_info_list[0]
            xref = img_info.get("xref", 0)
            if xref == 0:
                continue

            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "").lower()

            pil_img = Image.open(io.BytesIO(image_bytes))
            
            # 保持原始尺寸，不再放大

            tmp_png = f"/tmp/{uuid.uuid4()}.png"
            pil_img.save(tmp_png, format="PNG")

            # 恢复方向
            transform = pymupdf.Matrix(img_info.get("transform", (1, 0, 0, 1, 0, 0))) * page.rotation_matrix
            metadata = {"transform": transform, "height": base_image["height"]}
            restored_path = restore_image_orientation(tmp_png, metadata)

            bmp_path = f"/tmp/{uuid.uuid4()}.png"
            Image.open(restored_path).save(bmp_path, format="png")
            imageList.append(bmp_path)

            for p in (tmp_png, restored_path):
                if os.path.exists(p) and p != bmp_path:
                    os.remove(p)

    doc.close()
    return imageList


def pdf2images(pdf_path, pageRange):
    if not is_textPdf(pdf_path):
        return extract_single_image_per_page(pdf_path, pageRange=pageRange), True
    else:
        return doc2imgList(pdf_path, pageRange=pageRange), False


def restore_image_orientation(image_path, metadata):
    """
    根据PDF元数据恢复图像方向，支持5种变换情况：
    1. 垂直镜像 (上下翻转)
    2. 顺时针90°旋转
    3. 逆时针90°旋转
    4. 180°旋转
    5. 无变换

    :param image_path: 原始图像文件路径
    :param metadata: 包含'transform'的字典
    :return: (保存路径, 更新后的元数据)
    """
    _, file_extension = os.path.splitext(image_path)
    output_path = f"/tmp/{uuid.uuid4()}{file_extension}"
    img = Image.open(image_path)
    orig_width, orig_height = img.size

    # 获取变换矩阵
    transform = metadata.get('transform', (1, 0, 0, 1, 0, 0))
    a, b, c, d, e, f = transform

    # 计算旋转角度
    angle_rad = math.atan2(c, a)
    angle_deg = math.degrees(angle_rad)

    # 初始化更新后的元数据
    updated_meta = metadata.copy()
    updated_meta.setdefault('width', orig_width)
    updated_meta.setdefault('height', orig_height)

    # 处理不同变换情况
    if abs(angle_deg - 90) < 5:  # 逆时针90°旋转
        # 执行逆时针90°旋转
        restored_img = img.transpose(Image.ROTATE_90)

        # 更新宽高（互换）
        new_width, new_height = orig_height, orig_width
        updated_meta['width'] = new_width
        updated_meta['height'] = new_height

        # 计算新的变换矩阵（无旋转）
        updated_meta['transform'] = (1.0, 0.0, 0.0, 1.0, e, f)

    elif abs(angle_deg + 90) < 5 or abs(angle_deg - 270) < 5:  # 顺时针90°旋转
        # 执行顺时针90°旋转 (逆时针270°)
        restored_img = img.transpose(Image.ROTATE_270)

        # 更新宽高（互换）
        new_width, new_height = orig_height, orig_width
        updated_meta['width'] = new_width
        updated_meta['height'] = new_height

        # 计算新的变换矩阵（无旋转）
        updated_meta['transform'] = (1.0, 0.0, 0.0, 1.0, e, f)

    elif abs(angle_deg - 180) < 5 or (a < 0 and d < 0):  # 180°旋转
        # 执行180°旋转
        restored_img = img.transpose(Image.ROTATE_180)

        # 宽高不变
        updated_meta['transform'] = (1.0, 0.0, 0.0, 1.0, e, f)

    elif d < 0 and abs(b) < 1e-5 and abs(c) < 1e-5:  # 垂直镜像 (上下翻转)
        # 执行垂直翻转
        restored_img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 更新变换矩阵（正数d）
        updated_meta['transform'] = (a, b, c, -d, e, d * orig_height + f)

    else:  # 无变换
        restored_img = img

    restored_img.save(output_path)
    return output_path


def group_image_blocks(image_blocks, gap_threshold=10, overlap_threshold=0.1):
    """
    改进的分组算法：只合并相邻的图片碎片
    gap_threshold: 最大允许间隙（像素）
    overlap_threshold: 最小重叠比例（0-1）
    """
    if not image_blocks:
        return []

    # 按坐标和尺寸排序（优先Y坐标，其次X坐标）
    sorted_blocks = sorted(image_blocks, key=lambda b: (b['y'], b['x']))

    groups = []
    current_group = [sorted_blocks[0]]

    for i in range(1, len(sorted_blocks)):
        prev = current_group[-1]  # 当前组的最后一块
        curr = sorted_blocks[i]

        # 计算两个块之间的位置关系
        prev_right = prev['x'] + prev['w']
        prev_bottom = prev['y'] + prev['h']
        curr_right = curr['x'] + curr['w']
        curr_bottom = curr['y'] + curr['h']

        # 计算重叠区域
        x_overlap = max(0, min(prev_right, curr_right) - max(prev['x'], curr['x']))
        y_overlap = max(0, min(prev_bottom, curr_bottom) - max(prev['y'], curr['y']))

        # 计算间隙
        x_gap = max(0, curr['x'] - prev_right, prev['x'] - curr_right)
        y_gap = max(0, curr['y'] - prev_bottom, prev['y'] - curr_bottom)

        # 判断是否相邻的条件：
        # 1. 有显著重叠（任一方向重叠比例 > overlap_threshold）
        # 2. 间隙很小（任一方向间隙 < gap_threshold）
        significant_x_overlap = x_overlap > overlap_threshold * min(prev['w'], curr['w'])
        significant_y_overlap = y_overlap > overlap_threshold * min(prev['h'], curr['h'])
        small_x_gap = x_gap < gap_threshold
        small_y_gap = y_gap < gap_threshold

        # 判断是否在水平或垂直方向上相邻
        horizontal_neighbors = (abs(prev['y'] - curr['y']) < gap_threshold and
                                (significant_x_overlap or small_x_gap))

        vertical_neighbors = (abs(prev['x'] - curr['x']) < gap_threshold and
                              (significant_y_overlap or small_y_gap))

        # 对角线相邻（允许小间隙）
        diagonal_neighbors = (x_gap < gap_threshold and
                              y_gap < gap_threshold and
                              (x_gap + y_gap) < gap_threshold * 1.5)

        # 如果满足任一相邻条件，则加入当前组
        if horizontal_neighbors or vertical_neighbors or diagonal_neighbors:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]

    if current_group:
        groups.append(current_group)

    return groups


def merge_image_blocks(group):
    """
    改进的合并函数：确保图片块正确对齐不重叠
    """
    output_dir = "/tmp"
    # 计算合并后的总区域（精确计算）
    x0 = min(block['x'] for block in group)
    y0 = min(block['y'] for block in group)
    x1 = max(block['x'] + block['w'] for block in group)
    y1 = max(block['y'] + block['h'] for block in group)
    total_width = x1 - x0
    total_height = y1 - y0

    # 创建白色背景画布
    merged_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # 按坐标排序图片块（从左到右，从上到下）
    sorted_group = sorted(group, key=lambda b: (b['y'], b['x']))

    for block in sorted_group:
        try:
            img = Image.open(block['url'])

            # 确保图片尺寸匹配
            if img.size != (block['w'], block['h']):
                img = img.resize((block['w'], block['h']), Image.LANCZOS)

            # 计算在合并图中的精确位置
            offset_x = block['x'] - x0
            offset_y = block['y'] - y0

            # 检查位置是否在画布范围内
            if offset_x < 0 or offset_y < 0:
                print(f"警告: 图片块位置超出边界 {offset_x}, {offset_y}")
                continue

            # 检查是否重叠
            existing_region = merged_image.crop((offset_x, offset_y,
                                                 offset_x + block['w'],
                                                 offset_y + block['h']))
            if not all(pixel == (255, 255, 255) for pixel in existing_region.getdata()):
                print(f"警告: 在位置 ({offset_x}, {offset_y}) 检测到重叠")

            merged_image.paste(img, (offset_x, offset_y))
        except Exception as e:
            print(f"处理图片块时出错: {e}")
            continue

    save_path = f"{output_dir}/{uuid.uuid4()}.png"
    merged_image.save(save_path, quality=95)

    return {
        "page": group[0]['page'],
        "index": 0,
        "x": x0,
        "y": y0,
        "w": total_width,
        "h": total_height,
        "url": save_path,
        "size": os.path.getsize(save_path)
    }


def extract_images_from_pdf(pdf_path):
    pdf_document = pymupdf.open(pdf_path)
    result = []
    all_blocks = []  # 收集当前页所有图片块

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        img_info_list = page.get_image_info(xrefs=True)

        for img_index in img_info_list:
            xref = img_index.get("xref", 0)
            if xref == 0:
                continue
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")

            save_path = f"/tmp/{uuid.uuid4()}.{image_ext}"
            with open(save_path, "wb") as img_file:
                img_file.write(image_bytes)

            img_rect = img_index.get("bbox")
            if not img_rect:
                continue
            x0, y0, x1, y1 = img_rect
            width_pt = x1 - x0
            height_pt = y1 - y0

            x_px = int(x0 * 2)
            y_px = int(y0 * 2)
            width_px = int(width_pt * 2)
            height_px = int(height_pt * 2)

            transform = img_index.get('transform', (1, 0, 0, 1, 0, 0))

            # 修正图片方向
            metadata = {
                'transform': transform,
                'height': base_image['height']
            }
            restored_path = restore_image_orientation(save_path, metadata)

            image_block = {
                "page": page_num,
                "index": img_index,
                "x": x_px,
                "y": y_px,
                "w": width_px,
                "h": height_px,
                "url": restored_path,
                "tmp_path": save_path,
                "size": os.path.getsize(restored_path)
            }
            all_blocks.append(image_block)

    # 按页面分组处理
    grouped_by_page = {}
    for block in all_blocks:
        page = block['page']
        grouped_by_page.setdefault(page, []).append(block)

    # 处理每个页面的图片块
    for page, blocks in grouped_by_page.items():
        # 分组疑似碎片
        groups = group_image_blocks(blocks)

        # 合并碎片并生成新图片
        merged_blocks = []
        for group in groups:
            if len(group) > 1:  # 需要合并的组
                try:
                    merged = merge_image_blocks(group)
                    merged_blocks.append(merged)
                    # 删除原始碎片文件
                    for block in group:
                        if os.path.exists(block['tmp_path']):
                            os.remove(block['tmp_path'])
                        if os.path.exists(block['url']) and block['url'] != block['tmp_path']:
                            os.remove(block['url'])
                except Exception as e:
                    print(f"合并图片块时出错: {e}")
                    # 合并失败时保留原始图片块
                    merged_blocks.extend(group)
            else:  # 单张图片无需合并
                merged_blocks.append(group[0])

        result.extend(merged_blocks)

    pdf_document.close()

    # 过滤小图片（小于10KB）
    filtered_result = []
    for img_info in result:
        img_path = img_info['url']

        # 检查文件是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图片文件不存在 {img_path}")
            continue

        # 获取文件大小（KB）
        file_size_kb = os.path.getsize(img_path) / 1024

        # 过滤条件：文件大小 >= 10KB
        if file_size_kb >= 10:
            # 更新大小信息
            img_info['size'] = os.path.getsize(img_path)
            filtered_result.append(img_info)
        else:
            # 删除小图片文件
            try:
                os.remove(img_path)
                print(f"已删除小图片: {img_path} (大小: {file_size_kb:.1f}KB)")
            except Exception as e:
                print(f"删除图片失败: {img_path}, 错误: {e}")

    # 清理临时目录
    temp_files = set()
    for block in all_blocks:
        if 'tmp_path' in block:
            temp_files.add(block['tmp_path'])
        if 'url' in block and block['url'] != block.get('tmp_path', ''):
            temp_files.add(block['url'])

    for temp_file in temp_files:
        if os.path.exists(temp_file) and temp_file not in [img['url'] for img in filtered_result]:
            try:
                os.remove(temp_file)
                print(f"清理临时文件: {temp_file}")
            except Exception as e:
                print(f"清理临时文件失败: {temp_file}, 错误: {e}")

    return filtered_result


# 提取PDF批注以及所在位置
def extract_annotations_with_content(pdf_path):
    doc = pymupdf.open(pdf_path)
    annotations = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_annotations = page.annots()

        for annotation in page_annotations:
            content = annotation.info['content']
            rect = annotation.rect

            # 获取批注位置附近的文本
            text_rect = pymupdf.Rect(rect)
            text = page.get_textbox(text_rect)

            annotations.append({
                'page': page_num + 1,  # 页码从1开始
                'content': content,
                'position': (rect.x0, rect.y0, rect.x1, rect.y1),
                'corresponding_text': text
            })

    return annotations


## ------ 简单判断扫描件还是拍照的照片 ---
def is_scan_by_corners(image_path, threshold=220, corner_size=50):
    """
    通过分析图片四个角的颜色来判断是否为扫描件

    参数:
        image_path: 图片路径
        threshold: 判断为白色的阈值(0-255)
        corner_size: 采样的角落区域大小

    返回:
        (is_scan, confidence, details)
        is_scan: True表示可能是扫描件，False表示可能是拍照
        confidence: 置信度 0-1
        details: 详细信息
    """

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 转换为RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    # 定义四个角的采样区域
    corners = {
        'top_left': img_rgb[0:corner_size, 0:corner_size],
        'top_right': img_rgb[0:corner_size, width - corner_size:width],
        'bottom_left': img_rgb[height - corner_size:height, 0:corner_size],
        'bottom_right': img_rgb[height - corner_size:height, width - corner_size:width]
    }

    # 分析每个角的颜色
    corner_stats = {}
    white_corners = 0

    for name, corner in corners.items():
        # 计算平均RGB值
        avg_color = np.mean(corner, axis=(0, 1))

        # 判断是否接近白色（RGB都大于阈值）
        is_white = all(avg_color > threshold)
        if is_white:
            white_corners += 1

        corner_stats[name] = {
            'avg_color': avg_color,
            'is_white': is_white,
            'brightness': np.mean(avg_color)  # 平均亮度
        }

    # 判断逻辑：至少3个角是白色则认为是扫描件
    is_scan = white_corners >= 3
    confidence = white_corners / 4.0

    details = {
        'white_corners_count': white_corners,
        'corner_stats': corner_stats,
        'threshold_used': threshold,
        'image_size': (width, height)
    }

    return is_scan, confidence, details


def analyze_image(image_path):
    """主分析函数，输出详细结果"""
    print(f"分析图片: {image_path}")
    print("=" * 50)

    # 使用默认参数判断
    is_scan, confidence, details = is_scan_by_corners(image_path)

    # 输出结果
    result = "扫描件" if is_scan else "拍照件"
    print(f"\n判断结果: {result}")
    print(f"置信度: {confidence * 100:.1f}%")
    print(f"白色角数量: {details['white_corners_count']}/4")

    # 输出每个角的详细信息
    print("\n四角颜色分析:")
    for name, stats in details['corner_stats'].items():
        color = stats['avg_color']
        print(f"  {name:12s}: RGB({color[0]:3.0f}, {color[1]:3.0f}, {color[2]:3.0f}) "
              f"亮度={stats['brightness']:.1f} {'✓白' if stats['is_white'] else '✗非白'}")

    return is_scan, confidence, details
