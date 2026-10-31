import sys, os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def binarize(img, method="otsu", bg_kernel=None):
    """文档二值化预处理
    method: "otsu" — 背景归一化 + OTSU（去背景线条/光照不均）
            "raw_otsu" — 纯 OTSU（无背景归一化）
            "adaptive" — 自适应阈值
    bg_kernel: 背景估计的高斯核大小（仅 otsu 模式），越大保留越多细节
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    if method == "raw_otsu":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == "adaptive":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)

    else:  # "otsu" — 背景归一化 + OTSU（默认）
        if bg_kernel is None:
            bg_kernel = max(gray.shape) // 15
            if bg_kernel % 2 == 0:
                bg_kernel += 1
            bg_kernel = max(5, bg_kernel)
        # ① 大核高斯模糊 → 粗略背景（抹平文字，保留光照）
        bg_rough = cv2.GaussianBlur(gray, (bg_kernel, bg_kernel), 0)
        # ② 3×3 腐蚀 → 去除高斯模糊残留纹理
        bg_refined = cv2.erode(bg_rough, np.ones((3, 3), np.uint8), iterations=1)
        norm = (gray.astype(np.float32) / (bg_refined.astype(np.float32) + 1e-6)) * 255
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3x3 corrosion -> expand dark regions for grouping
    binary = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)

    # 背景→白，前景→原图
    result = img.copy()
    result[binary == 255] = [255, 255, 255]
    return result


def enhancement(img_path):
    if isinstance(img_path, str):
        image = cv2.imread(img_path)
    elif isinstance(img_path, np.ndarray):
        image = img_path
    else:
        raise TypeError("Invalid input type. Expected a file path or an image array.")
    return binarize(image)


# ================================================================
# 多页投票去水印
# ================================================================


def _compute_background_mask(img, threshold=255, erode=True):
    """
    单页背景暗部检测：背景归一化 + OTSU，保留非文字区域的暗色像素。
    阈值 <threshold 的灰度像素 → 暗部候选（水印、底色等）。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    bg_kernel = max(gray.shape) // 15
    if bg_kernel % 2 == 0:
        bg_kernel += 1
    bg_kernel = max(5, bg_kernel)
    bg_rough = cv2.GaussianBlur(gray, (bg_kernel, bg_kernel), 0)
    # 3×3 腐蚀 → 去除高斯模糊残留纹理
    bg_refined = cv2.erode(bg_rough, np.ones((3, 3), np.uint8), iterations=1)
    norm = (gray.astype(np.float32) / (bg_refined.astype(np.float32) + 1e-6)) * 255
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 3×3 腐蚀 → 去除噪点，保留水印等大尺度暗区
    if erode:
        binary = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    bg_keep = img.copy()
    bg_keep[binary == 0] = [255, 255, 255]
    gray_keep = cv2.cvtColor(bg_keep, cv2.COLOR_BGR2GRAY)
    return gray_keep < threshold


def _watermark_vote_mask(images, min_pages=9, threshold=255):
    """
    多页投票：≥min_pages 页一致的暗部像素 → 水印候选区域。
    返回 (watermark_mask, min_h, min_w)。
    """
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    votes = np.zeros((min_h, min_w), dtype=np.int32)
    page_dark_masks = []
    for img in images:
        dark = _compute_background_mask(img, threshold, erode=False)
        votes += dark[:min_h, :min_w].astype(np.int32)
        page_dark_masks.append(dark[:min_h, :min_w])
    return votes >= min_pages, min_h, min_w, page_dark_masks


def _compute_watermark_threshold(images, voted_mask):
    """
    从投票确认的水印区域中，提取所有页原图的灰度值，计算均值。
    该均值作为动态阈值：灰度 < 均值 → 水印（涂白），灰度 > 均值 → 文字（保护）。
    如果水印区域为空，回退到 150。
    """
    min_h = min(voted_mask.shape[0], min(img.shape[0] for img in images))
    min_w = min(voted_mask.shape[1], min(img.shape[1] for img in images))
    mask_crop = voted_mask[:min_h, :min_w]

    if not mask_crop.any():
        return 150.0

    gray_values = []
    for img in images:
        gray = cv2.cvtColor(img[:min_h, :min_w], cv2.COLOR_BGR2GRAY)
        gray_values.append(gray[mask_crop])

    all_gray = np.concatenate(gray_values)
    mean_gray = float(np.mean(all_gray))
    # 阶段三实际使用 _compute_background_mask 的暗部 mask 来涂白（复用阶段一的背景前景分离）
    return mean_gray


def _separate_text_watermark(images, voted_mask, text_threshold=150):
    """
    [显式阈值模式] 在投票 mask 中分离文字和水印：
    原图像素灰度 < text_threshold 且 ≥50% 页都暗 → 文字，从水印 mask 中剔除。
    仅在调用方显式传入 text_threshold 时使用；默认走动态阈值路径。
    """
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    mask_h = min(voted_mask.shape[0], min_h)
    mask_w = min(voted_mask.shape[1], min_w)
    cropped_mask = voted_mask[:mask_h, :mask_w]

    text_votes = np.zeros((mask_h, mask_w), dtype=np.int32)
    total_pages = len(images)
    for img in images:
        cropped = img[:min_h, :min_w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
        dark_text = gray < text_threshold
        text_votes += dark_text[:mask_h, :mask_w].astype(np.int32)

    is_text = text_votes >= total_pages * 0.5
    refined_mask = cropped_mask.copy()
    refined_mask[is_text] = False

    total_voted = int(np.sum(cropped_mask))
    text_count = int(np.sum(cropped_mask & is_text))
    wm_count = int(np.sum(refined_mask))
    if total_voted > 0:
        print(f"  投票暗部: {total_voted:,} px"
              f" → 文字(每页都暗): {text_count:,} px"
              f" → 水印: {wm_count:,} px", flush=True)
    return refined_mask


def enhancement_multi(images, watermark_mask=None, min_pages=9, threshold=255, text_threshold=None):
    """
    多页水印去除：跨页投票 → 动态阈值分离文字/水印 → 仅涂白水印像素。

    Args:
        images: 图片列表（numpy 数组），要求尺寸一致
        watermark_mask: 预计算的水印 mask，None 则自动投票生成
        min_pages: 最小投票页数
        threshold: 单页暗部灰度阈值
        text_threshold: None → 动态计算（投票水印区域的灰度均值）
                        传入数值 → 显式阈值，走旧版 _separate_text_watermark

    Returns:
        增强后的图片列表（numpy 数组）
    """
    if len(images) == 0:
        return []

    if watermark_mask is None:
        # 只有 1 张图片无法跨页投票，回退到单页二值化
        if len(images) == 1:
            return [enhancement(images[0])]

        actual_min_pages = max(1, int(np.ceil(len(images) * 0.5)))
        raw_mask, _, _, page_dark_masks = _watermark_vote_mask(images, actual_min_pages, threshold)

        if text_threshold is not None:
            # 显式阈值模式：投票 mask 中剔除每页都暗的文字像素
            watermark_mask = _separate_text_watermark(images, raw_mask, text_threshold)
        else:
            # 动态阈值模式：从水印区域灰度均值得到阈值，不修改 mask
            text_threshold = _compute_watermark_threshold(images, raw_mask)
            watermark_mask = raw_mask

    result = []
    for i, img in enumerate(images):
        enhanced = img.copy()
        h, w = img.shape[:2]
        wm_h = min(watermark_mask.shape[0], h)
        wm_w = min(watermark_mask.shape[1], w)
        wm_crop = watermark_mask[:wm_h, :wm_w]

        # 复用阶段一的暗部mask: 非文字区域中灰度 < 250 的像素 → 水印涂白
        dark = page_dark_masks[i][:wm_h, :wm_w]
        safe_to_paint = wm_crop & dark

        enhanced[:wm_h, :wm_w][safe_to_paint] = [255, 255, 255]
        result.append(enhanced)

    return result


def _group_by_y_projection(images, corr_threshold=0.8, sil_margin=0.15):
    n = len(images)
    if n <= 1:
        return [[i for i in range(n)]], "k=1 (1 page)"

    profiles = []
    for img in images:
        mask = _compute_background_mask(img)
        y = np.sum(mask, axis=1).astype(np.float32)
        if y.max() > 0:
            y /= y.max()
        profiles.append(y)

    T = 256
    aligned = np.array([np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(p)), p) for p in profiles])

    corr = np.corrcoef(aligned)
    if corr.min() >= corr_threshold:
        return [[i for i in range(n)]], f"k=1 (corr>={corr_threshold})"

    X = StandardScaler().fit_transform(aligned)

    best_clean_k, best_clean_sil = 2, -1
    all_results = []
    for k in range(2, min(8, n)):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sizes = sorted([int(np.sum(labels == lbl)) for lbl in set(labels)])
        all_results.append((k, sil, labels, sizes))
        if 1 not in sizes and sil > best_clean_sil:
            best_clean_k, best_clean_sil = k, sil

    for k, sil, labels, sizes in all_results:
        if k > best_clean_k and 1 in sizes and sil >= best_clean_sil * (1 - sil_margin):
            groups = {}
            for i, lbl in enumerate(labels):
                groups.setdefault(int(lbl), []).append(i)
            return list(groups.values()), f"k={k} (outlier bonus, sil={sil:.3f} vs k={best_clean_k} sil={best_clean_sil:.3f})"

    km = KMeans(n_clusters=best_clean_k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    groups = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(i)
    return list(groups.values()), f"k={best_clean_k} (sil={best_clean_sil:.3f})"
def _group_by_overlap(images, overlap_threshold=0.95):
    """
    大核模糊 + 降采样 + NCC（归一化互相关系数）分组。
    
    大核高斯模糊抹掉文字等高频小尺度特征，保留水印等大尺度结构。
    降采样到 64x64 后计算 NCC，NCC >= overlap_threshold → 同组。
    
    返回: [[original_indices], ...] 每个子列表是一个分组
    """
    n = len(images)
    if n <= 1:
        return [[i] for i in range(n)]

    # 大核高斯模糊：抹掉文字，保留水印等大尺度结构
    kernel = max(images[0].shape[0], images[0].shape[1]) // 15
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(31, kernel)

    smalls = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        bl = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        small = cv2.resize(bl, (64, 64)).astype(np.float32)
        smalls.append(small)

    # 并查集：连通分量聚类
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 计算所有页对的 NCC
    for i in range(n):
        for j in range(i + 1, n):
            a = smalls[i]
            b = smalls[j]
            a_mean = float(np.mean(a))
            b_mean = float(np.mean(b))
            a_std = float(np.std(a))
            b_std = float(np.std(b))
            if a_std < 1e-6 or b_std < 1e-6:
                continue
            ncc = float(np.mean((a - a_mean) * (b - b_mean))) / (a_std * b_std)
            if ncc >= overlap_threshold:
                union(i, j)

    # 按连通分量分组
    groups_map = {}
    for i in range(n):
        root = find(i)
        groups_map.setdefault(root, []).append(i)

    groups = list(groups_map.values())
    for g in groups:
        g.sort()
    groups.sort(key=lambda g: g[0])

    if len(groups) > 1:
        sizes = ",".join(f"({len(g)}页)" for g in groups)
        print(f"  NCC分组({overlap_threshold*100:.0f}%): {len(groups)} 组[{sizes}]", flush=True, file=sys.stderr)

    return groups

def enhancement_multi_by_size(images, min_pages=9, threshold=255, text_threshold=None):
    if len(images) == 0:
        return []

    groups = {}
    for i, img in enumerate(images):
        key = (img.shape[0], img.shape[1])
        groups.setdefault(key, []).append((i, img))

    print(f"[watermark] size groups: {len(groups)} sizes", flush=True, file=sys.stderr)

    result = [None] * len(images)
    for (h, w), group_imgs in groups.items():
        indices = [idx for idx, _ in group_imgs]
        imgs_only = [img for _, img in group_imgs]
        print(f"  [watermark] {h}x{w}: {len(imgs_only)} pages", flush=True, file=sys.stderr)

        y_groups = [list(range(len(imgs_only)))]
        print(f"    [watermark] Y-proj: skipped (all {len(imgs_only)}p one group)", flush=True, file=sys.stderr)
        for yg in y_groups:
            yg_indices = [indices[i] for i in yg]
            yg_pages = [idx + 1 for idx in yg_indices]
            yg_imgs = [imgs_only[i] for i in yg]
            print(f"      [watermark] [{len(yg_imgs)}p] pages {yg_pages}", flush=True, file=sys.stderr)

            if len(yg_imgs) == 1:
                print(f"        [watermark] single page -> binarize", flush=True, file=sys.stderr)
                yg_result = [enhancement(yg_imgs[0])]
            else:
                yg_min_pages = max(1, int(np.ceil(len(yg_imgs) * 0.5)))
                print(f"        [watermark] multi-page voting, min_pages={yg_min_pages}", flush=True, file=sys.stderr)
                yg_result = enhancement_multi(
                    yg_imgs,
                    min_pages=yg_min_pages,
                    threshold=threshold,
                    text_threshold=text_threshold,
                )

            for idx, enhanced_img in zip(yg_indices, yg_result):
                result[idx] = enhanced_img

    return result
