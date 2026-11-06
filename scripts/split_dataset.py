"""数据集划分脚本：把标注工作区按比例切分为 train / val / test 三个子集。

设计要点：
    1. 按整图划分 —— 同一张图的所有标注框只会出现在一个子集里，杜绝「同一画面
       既当训练又当验证」造成的指标虚高；
    2. 只带标签 —— 没有对应 txt 的图片跳过并警告，避免混入未标注样本；
    3. 结果可复现 —— 固定随机种子，同一批数据每次划分结果一致；
    4. 同时生成 data.yaml —— 路径、类别数、类别名一次写好，训练命令直接引用。

用法示例：
    python scripts/split_dataset.py --workspace data\\workspace --out data\\seat_dataset
"""

import argparse
import os
import random
import shutil
import sys

# 项目类别，顺序与 prelabel.py 保持一致
PROJECT_CLASSES = ["person", "book", "bag", "cup", "item"]

# 默认划分比例：训练 7 / 验证 1.5 / 测试 1.5
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# 固定随机种子，保证划分结果可复现
SPLIT_SEED = 20261104


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="按整图划分训练 / 验证 / 测试集")
    parser.add_argument("--workspace", required=True, help="标注工作区，含 images 与 labels 子目录")
    parser.add_argument("--out", required=True, help="数据集输出目录，如 data\\seat_dataset")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"], help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"], help="验证集比例")
    return parser.parse_args()


def collect_labeled_pairs(workspace: str) -> list:
    """收集「图片 + 标签」成对的文件基名；缺标签的图片记入警告列表。"""
    images_dir = os.path.join(workspace, "images")
    labels_dir = os.path.join(workspace, "labels")

    paired_names = []
    missing_labels = []

    for file_name in sorted(os.listdir(images_dir)):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        base_name = os.path.splitext(file_name)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        if os.path.isfile(label_path):
            paired_names.append(base_name)
        else:
            missing_labels.append(file_name)

    return paired_names, missing_labels


def split_names(names: list, train_ratio: float, val_ratio: float) -> dict:
    """按比例随机划分文件基名，返回 {子集名: [基名列表]}。"""
    shuffled = list(names)
    random.Random(SPLIT_SEED).shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio))

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


def copy_subset(workspace: str, out_root: str, subset_name: str, base_names: list) -> None:
    """把某个子集的图片与标签复制到输出目录对应位置。"""
    images_src = os.path.join(workspace, "images")
    labels_src = os.path.join(workspace, "labels")

    images_dst = os.path.join(out_root, "images", subset_name)
    labels_dst = os.path.join(out_root, "labels", subset_name)
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(labels_dst, exist_ok=True)

    for base_name in base_names:
        for extension in (".jpg", ".jpeg", ".png"):
            image_path = os.path.join(images_src, base_name + extension)
            if os.path.isfile(image_path):
                shutil.copy2(image_path, os.path.join(images_dst, base_name + extension))
                break

        shutil.copy2(
            os.path.join(labels_src, f"{base_name}.txt"),
            os.path.join(labels_dst, f"{base_name}.txt"),
        )


def write_data_yaml(out_root: str) -> None:
    """生成 ultralytics 训练用的 data.yaml。

    不写 path 字段：ultralytics 会把 yaml 所在目录当作数据集根目录，
    这样整个项目拷到别的机器（含换盘符）仍可直接训练。
    """
    yaml_lines = [
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(PROJECT_CLASSES)}",
        f"names: {PROJECT_CLASSES}",
    ]

    yaml_path = os.path.join(out_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write("\n".join(yaml_lines) + "\n")


def count_boxes(labels_dir: str) -> dict:
    """统计某个标签目录下每个类别的框数量，用于检查类别均衡度。"""
    class_counts = {index: 0 for index in range(len(PROJECT_CLASSES))}

    for file_name in os.listdir(labels_dir):
        if not file_name.endswith(".txt"):
            continue
        with open(os.path.join(labels_dir, file_name), "r", encoding="utf-8") as label_file:
            for line in label_file:
                parts = line.split()
                if parts:
                    class_counts[int(float(parts[0]))] += 1
    return class_counts


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()

    if not os.path.isdir(arguments.workspace):
        print(f"[错误] 工作区不存在：{arguments.workspace}")
        return 1

    paired_names, missing_labels = collect_labeled_pairs(arguments.workspace)
    if not paired_names:
        print("[错误] 工作区里没有任何「图片 + 标签」成对的样本")
        return 1

    for file_name in missing_labels:
        print(f"[警告] 缺少标签，已跳过：{file_name}")

    subsets = split_names(paired_names, arguments.train_ratio, arguments.val_ratio)

    for subset_name, base_names in subsets.items():
        copy_subset(arguments.workspace, arguments.out, subset_name, base_names)
        print(f"[划分] {subset_name}: {len(base_names)} 张")

    write_data_yaml(arguments.out)
    print(f"[完成] data.yaml 已生成：{os.path.join(arguments.out, 'data.yaml')}")

    print("-" * 70)
    for subset_name in subsets:
        labels_dir = os.path.join(arguments.out, "labels", subset_name)
        class_counts = count_boxes(labels_dir)
        count_text = "，".join(
            f"{PROJECT_CLASSES[index]}={count}" for index, count in class_counts.items()
        )
        print(f"[统计] {subset_name}: {count_text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
