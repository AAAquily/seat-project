# -*- coding: utf-8 -*-
"""标定场景匹配：判断上传图片是否属于已标定的固定机位。

原理：
    对「参考图」与「上传图」分别提取 ORB 特征，做汉明距离 KNN 匹配 +
    Lowe 比率测试，再对好匹配做单应性 RANSAC 几何验证。
    ORB 基于结构特征，配合直方图均衡化可跨昼夜匹配同一机位的照片；
    异场景的巧合匹配撑不起统一的单应性变换，会被内点数阈值挡掉。

匹配成功 → 上传接口走固定座位表（标定场景，精度最高）；
匹配失败 → 回退动态座位识别（任意场景可用）。
"""

import os

import cv2
import numpy as np

# ---------- 匹配参数 ----------
ORB_FEATURES = 4000        # 每张图提取的特征点上限：机位场景纹理中等，4000 够用
MATCH_RATIO = 0.75         # Lowe 比率测试：最近邻/次近邻距离比，越小越严格
MIN_GOOD_MATCHES = 30      # 初筛：最少好匹配数（低于它直接判不匹配）
MIN_INLIERS = 60           # 终判：单应性 RANSAC 内点数下限
                           # 实测同机位昼夜照片 ≥186，异场景网络图 ≤6，余量充足
RANSAC_REPROJ = 5.0        # 单应性 RANSAC 重投影误差容限（像素）

# 参考图特征缓存：{绝对路径: (文件修改时间, {keypoints, descriptors})}
# 同一张参考图只在首次使用时计算一次特征，之后直接复用
_reference_cache = {}


def _gray_equalized(image):
    """转灰度并做 CLAHE 直方图均衡，削弱昼夜光照差异对特征点的影响。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _features(image):
    """计算单张图像的 ORB 关键点与描述子。"""
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    return orb.detectAndCompute(_gray_equalized(image), None)


def _reference_of(path: str, project_root: str):
    """加载参考图的关键点与描述子，带修改时间缓存。"""
    abs_path = os.path.normpath(os.path.join(project_root, path))
    if not os.path.isfile(abs_path):
        return None
    mtime = os.path.getmtime(abs_path)
    cached = _reference_cache.get(abs_path)
    if cached and cached[0] == mtime:
        return cached[1]

    image = cv2.imread(abs_path)
    if image is None:
        return None
    keypoints, descriptors = _features(image)
    entry = {"keypoints": keypoints, "descriptors": descriptors}
    _reference_cache[abs_path] = (mtime, entry)
    return entry


def _good_matches(frame_desc, reference_desc) -> list:
    """比率测试过滤后的好匹配列表。"""
    if frame_desc is None or reference_desc is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(frame_desc, reference_desc, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance:
            good.append(pair[0])
    return good


def _homography_inliers(frame_kp, frame_desc, reference, good) -> int:
    """对好匹配做单应性 RANSAC，返回几何一致的内点数。"""
    if len(good) < 8:
        return 0
    src = np.float32([frame_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([reference["keypoints"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    return int(mask.sum()) if mask is not None else 0


def _entry_image(entry) -> str:
    """取参考图条目的路径：兼容旧格式（纯路径字符串）与新格式（dict 含 image 键）。"""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("image", "")
    return ""


def match_scene(frame, config: dict, project_root: str) -> tuple:
    """判断上传图是否属于标定场景。

    返回 (是否匹配, 命中的参考图路径, 内点数)。
    上传图特征只算一次，逐张参考图匹配，取内点数最高者。
    """
    frame_kp, frame_desc = _features(frame)
    best_inliers = 0
    best_ref = ""
    for entry in config.get("scene_references") or []:
        ref_path = _entry_image(entry)
        if not ref_path:
            continue
        reference = _reference_of(ref_path, project_root)
        if reference is None:
            continue
        good = _good_matches(frame_desc, reference["descriptors"])
        if len(good) < MIN_GOOD_MATCHES:
            continue
        inliers = _homography_inliers(frame_kp, frame_desc, reference, good)
        if inliers > best_inliers:
            best_inliers = inliers
            best_ref = ref_path
    return best_inliers >= MIN_INLIERS, best_ref, best_inliers


def resolve_scene_seats(config: dict, reference_path: str) -> list:
    """取命中参考图对应的座位表。

    规则：scene_references 中与 reference_path 同名的条目若内联了 seats，
    则优先使用该场景专属表（不同昼夜/布局照片椅子位置可能不同）；
    否则回退到全局 seats 表。
    """
    for entry in config.get("scene_references") or []:
        path = _entry_image(entry)
        if path and os.path.normpath(path) == os.path.normpath(reference_path or ""):
            if isinstance(entry, dict) and entry.get("seats"):
                return entry["seats"]
            break
    return config.get("seats") or []


# ------------------------------------------------------------------ 自检入口
if __name__ == "__main__":
    """命令行自检：python scripts/scene_match.py [图片路径]"""
    import sys

    import yaml

    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project, "config", "config.yaml"), encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project, "data/raw_real/real_03.jpg")
    img = cv2.imread(target)
    hit, ref, inliers = match_scene(img, conf, project)
    print(f"图片: {target}")
    print(f"匹配: {'命中' if hit else '未命中'} 参考图={ref} 内点={inliers} (阈值 {MIN_INLIERS})")
