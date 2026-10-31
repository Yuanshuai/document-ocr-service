import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from algo.pdf.pdf2images import extract_images_from_pdf


# import cv2
# from skimage.transform import resize
# from skimage.metrics import structural_similarity as compare_ssim
# def calculate_ssim(img1, img2, channel_axis, win_size, data_range):
#     ssim_value = compare_ssim(img1, img2, multichannel=(channel_axis is not None), win_size=win_size,
#                               channel_axis=channel_axis, data_range=data_range)
#     ssim_normalized = (ssim_value + 1) / 2
#     return ssim_normalized
#
#
# def image_similarity(img1_path: str, img2_path: str, threshold=0.9):
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#     if img1.shape != img2.shape:
#         min_height = min(img1.shape[0], img2.shape[0])
#         min_width = min(img1.shape[1], img2.shape[1])
#         img1 = resize(img1, (min_height, min_width), anti_aliasing=True)
#         img2 = resize(img2, (min_height, min_width), anti_aliasing=True)
#
#     if img1.ndim == 2:
#         channel_axis = None
#     else:
#         channel_axis = -1
#
#     min_dim = min(img1.shape[:2])
#     win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
#
#     if img1.dtype == np.float32 or img1.dtype == np.float64:
#         data_range = 1.0
#     else:
#         data_range = 255.0
#
#     # 计算原始图片的结构相似性指数（SSIM）
#     ssim = calculate_ssim(img1, img2, channel_axis, win_size, data_range)
#     print(f"图片相似度：{ssim}")
#     if ssim >= threshold:
#         return True, ssim
#     else:
#         return False, ssim


def get_feature(path):
    """
    VGG16模型抽取图片特征
    """
    from config import model_similarity
    if model_similarity is None:
        raise RuntimeError("VGG16模型未加载，无法提取图片特征")
    # 图片转成PIL的Image对象，并且对图片做了缩放
    img = image.load_img(path, target_size=(224, 224))
    # 图片转成矩阵、并扩充了维度、最后是预处理
    predict_img = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
    # 丢入 vgg16 网络做特征抽取，返回特征并展平成一维向量方便计算余弦相似度
    return model_similarity.predict(predict_img).flatten()


def cos_sim(imag_path1, imag_path2, threshold=0.8):
    """
    计算两个向量之间的余弦相似度
    """
    a = get_feature(imag_path1)
    b = get_feature(imag_path2)
    a = np.mat(a)
    b = np.mat(b)
    similarity = (float((a * b.T)[0, 0]) / (np.linalg.norm(a) * np.linalg.norm(b))+1)/2
    print(f"图片相似度: {similarity}")
    if similarity >= threshold:
        return True, similarity
    else:
        return False, similarity


def plag_pdfImages(pdf1_path, pdf2_path, window_size=5, same_threshold=0.85, changed_threshold=0.6):
    """
    优化后的图片比对算法（显著降低时间复杂度）
    关键优化：
     1. 按页分组图片，减少跨页比较范围
     2. 尺寸预过滤，避免不必要的特征计算
     3. 特征预计算，避免重复计算
     4. 使用字典索引加速查找
     5. 限制反向匹配范围
    """
    # 提取图片
    images1_list = extract_images_from_pdf(pdf1_path)
    images2_list = extract_images_from_pdf(pdf2_path)

    # 预计算所有特征（避免重复计算）
    feature_cache = {}

    def get_cached_feature(img_path):
        if img_path not in feature_cache:
            feature_cache[img_path] = get_feature(img_path)
        return feature_cache[img_path]

    all_images = images1_list + images2_list
    for img in all_images:
        get_cached_feature(img['url'])  # 触发预计算

    # 相似度计算函数（使用预计算特征）
    def calculate_sim(img1, img2):
        a = np.mat(feature_cache[img1['url']])
        b = np.mat(feature_cache[img2['url']])
        dot_product = np.dot(a, b.T)[0, 0]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return float(dot_product) / (norm_a * norm_b)

    # ========== 关键优化：按页分组 + 页面范围过滤 ==========
    # 构建按页分组的字典 {page: [img_data]}
    group1 = {}
    for img in images1_list:
        group1.setdefault(img['page'], []).append(img)

    group2 = {}
    for img in images2_list:
        group2.setdefault(img['page'], []).append(img)

    # 收集所有可能的候选匹配对
    candidate_pairs = []
    matched_pairs = []
    matched_set1 = set()  # 存储已匹配图片的ID
    matched_set2 = set()

    # 遍历PDF1的每一页
    for page1, imgs1 in group1.items():
        # 确定比较的页面范围 (page1 ± window_size)
        min_page = max(0, page1 - window_size)
        max_page = page1 + window_size

        # 在范围内遍历PDF2的页面
        for page2 in range(min_page, max_page + 1):
            if page2 not in group2:
                continue

            imgs2 = group2[page2]
            # 遍历当前页的所有图片组合
            for img1 in imgs1:
                # 如果图片已匹配则跳过
                if id(img1) in matched_set1:
                    continue

                for img2 in imgs2:
                    # 如果图片已匹配则跳过
                    if id(img2) in matched_set2:
                        continue

                    # 快速尺寸过滤 (宽高比差异>20%直接跳过)
                    ratio1 = img1['w'] / img1['h']
                    ratio2 = img2['w'] / img2['h']
                    if abs(ratio1 - ratio2) > 0.2:
                        continue

                    # 计算相似度
                    similarity = calculate_sim(img1, img2)
                    if similarity >= changed_threshold:
                        # 记录候选对 (直接使用对象引用)
                        candidate_pairs.append((img1, img2, similarity))

    # 按相似度降序排序候选对
    candidate_pairs.sort(key=lambda x: x[2], reverse=True)

    # 匹配最佳候选对
    for img1, img2, similarity in candidate_pairs:
        # 检查是否已被匹配（可能在前面的匹配中已匹配）
        if id(img1) in matched_set1 or id(img2) in matched_set2:
            continue

        # 确定匹配状态
        status = "same" if similarity >= same_threshold else "changed"
        matched_pairs.append((img1, img2, status, similarity))

        # 标记已匹配
        matched_set1.add(id(img1))
        matched_set2.add(id(img2))

    # ========== 优化反向匹配 ==========
    # 仅处理未匹配的图片，且限制页面范围
    for img2 in images2_list:
        if id(img2) in matched_set2:
            continue

        best_match = None
        best_sim = -1

        # 确定搜索范围
        min_page = max(0, img2['page'] - window_size)
        max_page = img2['page'] + window_size

        # 在范围内搜索最佳匹配
        for img1 in images1_list:
            if id(img1) in matched_set1:
                continue
            if not (min_page <= img1['page'] <= max_page):
                continue

            # 尺寸过滤
            ratio1 = img1['w'] / img1['h']
            ratio2 = img2['w'] / img2['h']
            if abs(ratio1 - ratio2) > 0.2:
                continue

            similarity = calculate_sim(img1, img2)
            if similarity > best_sim:
                best_sim = similarity
                best_match = img1

        # 检查是否找到有效匹配
        if best_match and best_sim >= changed_threshold:
            status = "same" if best_sim >= same_threshold else "changed"
            matched_pairs.append((best_match, img2, status, best_sim))
            matched_set1.add(id(best_match))
            matched_set2.add(id(img2))

    # ========== 构建结果列表 ==========
    result_list = []

    # 创建一个页面映射字典，记录匹配图片的页面对应关系
    page_mapping = {}
    for img1, img2, status, similarity in matched_pairs:
        # 记录页面映射关系
        page_mapping[img1['page']] = img2['page']
        page_mapping[img2['page']] = img1['page']

    # 处理匹配的图片对
    for img1, img2, status, similarity in matched_pairs:
        result_list.append({
            'status': status,
            "origin": {
                "value": [{
                    "char": "图片内容相似。" if status == "same" else "图片内容发生修改",
                    'status': status,
                    "pageNo": img1['page'],
                    "bbox": {k: img1[k] for k in ['h', 'w', 'x', 'y']}
                }],
                "position": [{
                    "pageNo": img1['page'],
                    "bbox": {k: img1[k] for k in ['h', 'w', 'x', 'y']}
                }]
            },
            "target": {
                "value": [{
                    "char": "图片内容相似。" if status == "same" else "图片内容发生修改",
                    'status': status,
                    "pageNo": img2['page'],
                    "bbox": {k: img2[k] for k in ['h', 'w', 'x', 'y']}
                }],
                "position": [{
                    "pageNo": img2['page'],
                    "bbox": {k: img2[k] for k in ['h', 'w', 'x', 'y']}
                }]
            }
        })

    # 处理原始文档中未匹配的图片（缺失）
    for img in images1_list:
        if id(img) not in matched_set1:
            # 获取对应的目标页面：优先使用页面映射，否则使用原始页面
            target_page = page_mapping.get(img['page'], img['page'])

            result_list.append({
                "status": "removed",
                "origin": {
                    "value": [{
                        "char": "比对文档缺少图片！",
                        'status': "removed",
                        "pageNo": img['page'],
                        "bbox": {k: img[k] for k in ['h', 'w', 'x', 'y']}
                    }],
                    "position": [{
                        "pageNo": img['page'],
                        "bbox": {k: img[k] for k in ['h', 'w', 'x', 'y']}
                    }]
                },
                "target": {
                    "value": [],
                    "position": [{
                        "pageNo": target_page,
                        "bbox": {'h': 0, 'w': 0, 'x': 0, 'y': 0}
                    }]
                }
            })

    # 处理目标文档中未匹配的图片（新增）
    for img in images2_list:
        if id(img) not in matched_set2:
            # 获取对应的原始页面：优先使用页面映射，否则使用目标页面
            origin_page = page_mapping.get(img['page'], img['page'])

            result_list.append({
                "status": "added",
                "origin": {
                    "value": [],
                    "position": [{
                        "pageNo": origin_page,
                        "bbox": {'h': 0, 'w': 0, 'x': 0, 'y': 0}
                    }]
                },
                "target": {
                    "value": [{
                        "char": "原始文档缺失图片！",
                        'status': "added",
                        "pageNo": img['page'],
                        "bbox": {k: img[k] for k in ['h', 'w', 'x', 'y']}
                    }],
                    "position": [{
                        "pageNo": img['page'],
                        "bbox": {k: img[k] for k in ['h', 'w', 'x', 'y']}
                    }]
                }
            })

    # 清理临时文件
    for img in all_images:
        try:
            if os.path.exists(img['url']):
                os.remove(img['url'])
        except Exception:
            pass

    return result_list
