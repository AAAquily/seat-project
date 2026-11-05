"""标签校验工具：检查 YOLO 标签合法性，并把指定图片的标签画回原图供人工核对。

检查项：
    1. 类别索引 —— 必须落在 0 到类别数-1 之间，防止 LabelImg 里误选了不存在的类别；
    2. 坐标范围 —— cx / cy / w / h 必须在 0—1 之间，防止坐标越界或没做归一化；
    3. 框尺寸 —— 宽高必须大于 0，防止保存了空框；
    4. 列数 —— 每行必须恰好 5 列。

用法示例：
    python scripts/verify_labels.py --labels data\\workspace\\labels
    python scripts/verify_labels.py --labels data\\workspace\\labels \\
        --images data\\workspace\\images --visualize 20261104_0900_web_f0001
"""

import argparse
import os
import sys

import cv2

# 与 prelabel.py 保持一致的类别清单
PROJECT_CLASSES = ["person", "book", "bag", "cup", "item"]

# 预览图绘制颜色（BGR），与 prelabel.py 一致
CLASS_COLORS = {
    "person": (60, 60, 220),
    "book": (60, 180, 60),
    "bag": (200, 120, 40),
    "cup": (200, 60, 200),
    "item": (200, 200, 60),
}


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="校验 YOLO 标签并可视化指定图片")
    parser.add_argument("--labels", required=True, help="标签目录")
    parser.add_argument("--images", default="", help="图片目录，配合 --visualize 使用")
    parser.add_argument("--visualize", default="", help="要画回原图的文件基名")
    parser.add_argument("--preview-dir", default="data/verify_preview", help="预览图输出目录")
    return parser.parse_args()


def verify_label_file(label_path: str) -> list:
    """校验单个标签文件，返回问题列表（空列表表示通过）。"""
    problems = []

    with open(label_path, "r", encoding="utf-8") as label_file:
        for line_number, raw_line in enumerate(label_file, start=1):
            parts = raw_line.split()
            if not parts:
                continue

            if len(parts) != 5:
                problems.append(f"第 {line_number} 行列数为 {len(parts)}，应为 5")
                continue

            class_index = int(float(parts[0]))
            if class_index < 0 or class_index >= len(PROJECT_CLASSES):
                problems.append(f"第 {line_number} 行类别索引 {class_index} 越界（合法 0—{len(PROJECT_CLASSES) - 1}）")

            values = [float(value) for value in parts[1:]]
            for value in values:
                if value < 0.0 or value > 1.0:
                    problems.append(f"第 {line_number} 行坐标 {value} 超出 0—1 范围（未归一化或越界）")

            box_width, box_height = values[2], values[3]
            if box_width <= 0 or box_height <= 0:
                problems.append(f"第 {line_number} 行框宽高非正数")

    return problems


def visualize_label(label_path: str, image_path: str, output_path: str) -> None:
    """把标签框按归一化坐标换算回像素，画到原图上另存。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[失败] 图片读取失败：{image_path}")
        return

    image_height, image_width = image.shape[:2]

    with open(label_path, "r", encoding="utf-8") as label_file:
        for raw_line in label_file:
            parts = raw_line.split()
            if len(parts) != 5:
                continue

            class_index = int(float(parts[0]))
            class_name = PROJECT_CLASSES[class_index]
            center_x, center_y, box_width, box_height = [float(value) for value in parts[1:]]

            x1 = int((center_x - box_width / 2) * image_width)
            y1 = int((center_y - box_height / 2) * image_height)
            x2 = int((center_x + box_width / 2) * image_width)
            y2 = int((center_y + box_height / 2) * image_height)

            color = CLASS_COLORS[class_name]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1 + 3, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"[完成] 预览图已保存：{os.path.abspath(output_path)}")


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()

    if not os.path.isdir(arguments.labels):
        print(f"[错误] 标签目录不存在：{arguments.labels}")
        return 1

    label_files = sorted(
        name for name in os.listdir(arguments.labels)
        if name.endswith(".txt") and name != "classes.txt"
    )

    total_boxes = 0
    files_with_problems = 0

    for file_name in label_files:
        problems = verify_label_file(os.path.join(arguments.labels, file_name))
        box_count = sum(
            1 for line in open(os.path.join(arguments.labels, file_name), encoding="utf-8")
            if line.split()
        )
        total_boxes += box_count

        if problems:
            files_with_problems += 1
            print(f"[问题] {file_name}")
            for problem in problems:
                print(f"  - {problem}")

    print("-" * 70)
    print(f"[汇总] 标签文件 {len(label_files)} 个，框总数 {total_boxes}，"
          f"有问题的文件 {files_with_problems} 个")

    if files_with_problems == 0:
        print("[结论] 全部标签通过校验")

    if arguments.visualize:
        label_path = os.path.join(arguments.labels, f"{arguments.visualize}.txt")
        image_path = os.path.join(arguments.images, f"{arguments.visualize}.jpg")
        if not os.path.isfile(label_path) or not os.path.isfile(image_path):
            print(f"[错误] 找不到待可视化文件：{arguments.visualize}")
            return 1
        output_path = os.path.join(arguments.preview_dir, f"{arguments.visualize}_check.jpg")
        visualize_label(label_path, image_path, output_path)

    return 1 if files_with_problems > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
