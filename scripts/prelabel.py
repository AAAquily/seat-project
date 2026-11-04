"""预标注脚本：用 COCO 预训练模型生成 YOLO 格式标签初稿，减少人工画框工作量。

设计要点：
    1. 类别映射 —— COCO 的 person / book / backpack / bottle 等映射到项目的四类
       （person / book / bag / cup），不在映射表中的类别一律丢弃；
    2. 坐标归一化 —— 直接输出 YOLO 要求的 class cx cy w h（均为 0—1 的相对值）；
    3. 双重产物 —— 每张图同时输出一个 _preview.jpg，画好框便于快速判断预标注质量；
    4. 过滤小框 —— 面积占比过低的框（远处噪点）直接丢弃，减少人工删框量。

重要说明：
    预标注只是「减轻画框工作量」的起点，必须人工逐张核对修正。
    直接拿预标注结果训练，会把预训练模型的漏检与误检继承进新模型。

用法示例：
    python scripts/prelabel.py --src data\\raw --pattern "20261104_0900_web_*.jpg" \\
        --out-images data\\workspace\\images --out-labels data\\workspace\\labels
"""

import argparse
import glob
import os
import sys

import cv2

from ultralytics import YOLO

# 项目固定的四个类别，顺序即类别索引，不要随意调整
PROJECT_CLASSES = ["person", "book", "bag", "cup"]

# COCO 类别名到项目类别的映射；未列出的类别（chair、laptop、dining table 等）直接丢弃
COCO_TO_PROJECT = {
    "person": "person",
    "book": "book",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "bottle": "cup",
    "cup": "cup",
}

# 框面积占整图比例的下限，低于此值视为远处噪点
MIN_AREA_RATIO = 0.0004

# 预览图里每个类别的绘制颜色（BGR）
CLASS_COLORS = {
    "person": (60, 60, 220),
    "book": (60, 180, 60),
    "bag": (200, 120, 40),
    "cup": (200, 60, 200),
}


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="用 COCO 预训练模型生成 YOLO 标注初稿")
    parser.add_argument("--src", required=True, help="原图目录")
    parser.add_argument("--pattern", default="*.jpg", help="文件名通配符，默认 *.jpg")
    parser.add_argument("--out-images", required=True, help="工作图片输出目录")
    parser.add_argument("--out-labels", required=True, help="标签输出目录")
    parser.add_argument("--weights", default="models/yolov8n.pt", help="预训练权重路径")
    parser.add_argument("--conf", type=float, default=0.20, help="置信度阈值，默认 0.20")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理尺寸，默认 1280（远端小目标必须放大）")
    parser.add_argument("--preview-dir", default="data/prelabel_preview", help="预览图输出目录")
    return parser.parse_args()


def list_source_images(src_dir: str, pattern: str) -> list:
    """按通配符列出源图片，返回排好序的绝对路径列表。"""
    search_pattern = os.path.join(src_dir, pattern)
    return sorted(glob.glob(search_pattern))


def map_class_name(coco_name: str) -> str:
    """把 COCO 类别名映射为项目类别名，不需要的类别返回空字符串。"""
    return COCO_TO_PROJECT.get(coco_name, "")


def convert_box_to_yolo(box_xyxy: list, image_width: int, image_height: int) -> tuple:
    """把像素坐标框 (x1, y1, x2, y2) 转为 YOLO 的归一化 (cx, cy, w, h)。"""
    x1, y1, x2, y2 = box_xyxy

    # 先裁到画面内，避免模型输出的越界框产生负数或大于 1 的归一化值
    x1 = max(0.0, min(float(x1), image_width))
    x2 = max(0.0, min(float(x2), image_width))
    y1 = max(0.0, min(float(y1), image_height))
    y2 = max(0.0, min(float(y2), image_height))

    center_x = (x1 + x2) / 2.0 / image_width
    center_y = (y1 + y2) / 2.0 / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    return center_x, center_y, box_width, box_height


def draw_preview(image, detections: list, output_path: str) -> None:
    """把预标注框画到图上另存，方便一眼判断质量。"""
    canvas = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["pixel_box"]
        color = CLASS_COLORS[detection["class_name"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label_text = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(canvas, label_text, (x1 + 3, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.imwrite(output_path, canvas)


def prelabel_image(model: YOLO, image_path: str, arguments: argparse.Namespace) -> dict:
    """对单张图推理并按类别映射，返回检测结果与统计。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[失败] 图片读取失败：{image_path}")
        return {"ok": False}

    image_height, image_width = image.shape[:2]
    results = model.predict(
        source=image_path,
        conf=arguments.conf,
        imgsz=arguments.imgsz,
        device="cpu",
        verbose=False,
    )

    detections = []
    result = results[0]
    class_names = result.names

    for box in result.boxes:
        coco_name = class_names[int(box.cls.item())]
        project_name = map_class_name(coco_name)
        if not project_name:
            continue

        x1, y1, x2, y2 = [round(value, 1) for value in box.xyxy[0].tolist()]
        area_ratio = ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
        if area_ratio < MIN_AREA_RATIO:
            continue

        detections.append(
            {
                "class_name": project_name,
                "confidence": float(box.conf.item()),
                "pixel_box": [int(x1), int(y1), int(x2), int(y2)],
                "yolo_box": convert_box_to_yolo([x1, y1, x2, y2], image_width, image_height),
            }
        )

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 原图复制到工作目录，标注时就地出标签，路径关系简单不易错
    working_image_path = os.path.join(arguments.out_images, f"{base_name}.jpg")
    cv2.imwrite(working_image_path, image)

    label_path = os.path.join(arguments.out_labels, f"{base_name}.txt")
    with open(label_path, "w", encoding="utf-8") as label_file:
        for detection in detections:
            class_index = PROJECT_CLASSES.index(detection["class_name"])
            center_x, center_y, box_width, box_height = detection["yolo_box"]
            line = f"{class_index} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"
            label_file.write(line)

    preview_path = os.path.join(arguments.preview_dir, f"{base_name}_preview.jpg")
    draw_preview(image, detections, preview_path)

    class_counts = {name: 0 for name in PROJECT_CLASSES}
    for detection in detections:
        class_counts[detection["class_name"]] += 1

    return {"ok": True, "name": base_name, "total": len(detections), "counts": class_counts}


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()

    source_images = list_source_images(arguments.src, arguments.pattern)
    if not source_images:
        print(f"[错误] 未匹配到图片：{arguments.src}\\{arguments.pattern}")
        return 1

    for directory in [arguments.out_images, arguments.out_labels, arguments.preview_dir]:
        os.makedirs(directory, exist_ok=True)

    # 类别清单文件，LabelImg 切 YOLO 模式时读取它作为可选类别
    classes_path = os.path.join(os.path.dirname(arguments.out_labels), "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as classes_file:
        classes_file.write("\n".join(PROJECT_CLASSES) + "\n")

    print(f"[信息] 待预标注 {len(source_images)} 张，类别：{' / '.join(PROJECT_CLASSES)}")
    model = YOLO(arguments.weights)

    total_counts = {name: 0 for name in PROJECT_CLASSES}
    total_boxes = 0
    empty_images = 0

    for image_path in source_images:
        summary = prelabel_image(model, image_path, arguments)
        if not summary["ok"]:
            continue

        total_boxes += summary["total"]
        for class_name, count in summary["counts"].items():
            total_counts[class_name] += count

        if summary["total"] == 0:
            empty_images += 1

        count_text = "，".join(f"{name}={count}" for name, count in summary["counts"].items() if count > 0)
        print(f"[预标注] {summary['name']}  共 {summary['total']} 框  {count_text}")

    print("-" * 70)
    print(f"[汇总] 图片 {len(source_images)} 张，框总数 {total_boxes}")
    for class_name, count in total_counts.items():
        print(f"  {class_name}: {count}")
    print(f"[注意] {empty_images} 张图预标注为空，需人工判断是真空场景还是漏检")

    if empty_images > len(source_images) * 0.5:
        print("[警告] 超过一半图片无框，请降低 --conf 或核对 --pattern 是否正确")

    print(f"[完成] 工作图片：{os.path.abspath(arguments.out_images)}")
    print(f"[完成] 标签初稿：{os.path.abspath(arguments.out_labels)}")
    print(f"[完成] 质量预览：{os.path.abspath(arguments.preview_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
