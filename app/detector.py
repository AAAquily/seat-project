"""模型推理封装：进程内复用同一份权重，供 Web 接口与脚本共用。

加载 yolov8 权重约需 1～3 秒，若每次 HTTP 请求都新建 YOLO 对象，看板会出现明显卡顿，
因此这里做进程级缓存：同一权重路径只加载一次。
"""

import os
from typing import Tuple

import numpy as np
from ultralytics import YOLO

# 权重路径 -> YOLO 实例
_MODEL_CACHE = {}


def get_model(weights: str) -> YOLO:
    """按路径取模型，首次调用时加载并缓存。"""
    absolute_path = os.path.abspath(weights)
    if absolute_path not in _MODEL_CACHE:
        _MODEL_CACHE[absolute_path] = YOLO(absolute_path)
    return _MODEL_CACHE[absolute_path]


def detect_image(
    image: np.ndarray,
    weights,
    conf: float,
    imgsz: int,
    device: str = "cpu",
) -> Tuple[list, np.ndarray]:
    """对单帧图像推理，返回 (检测列表, 原始画面)。

    weights 可以是单个路径，也可以是路径列表——列表时逐个推理并取并集去重
    （标定通道的「双模型融合」，与 scripts/judge_seats.py 行为一致）。
    检测列表元素结构：{"class_id": int, "conf": float, "box": [x1, y1, x2, y2]}
    """
    from judge_seats import merge_detections

    paths = [weights] if isinstance(weights, str) else list(weights)
    detections = []
    canvas = image
    for path in paths:
        model = get_model(path)
        prediction = model.predict(source=image, conf=conf, imgsz=imgsz, device=device, verbose=False)
        result = prediction[0]
        canvas = result.orig_img

        current = []
        for box in result.boxes:
            coordinates = box.xyxy[0].tolist()
            current.append(
                {
                    "class_id": int(box.cls[0]),
                    "conf": float(box.conf[0]),
                    "box": [int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])],
                }
            )
        detections = current if not detections else merge_detections(detections, current)

    return detections, canvas
