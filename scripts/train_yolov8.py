"""YOLOv8 座位识别模型训练脚本。

用法：
    python scripts/train_yolov8.py                        # 默认 100 轮，imgsz=640
    python scripts/train_yolov8.py --epochs 60 --imgsz 960

设计说明：
    1. 单进程数据加载（workers=0）——Windows 下多进程易触发共享内存报错，训练集仅 13 张，速度无影响；
    2. imgsz=640 训练——CPU 训练在 1280 下耗时约为 640 的 2.5 倍，小数据集不划算；
       推理阶段再单独提高 imgsz（config.yaml 中 inference.imgsz=1280）即可兼顾远端小目标；
    3. 关闭 AMP——混合精度仅对 GPU 有效，CPU 上强制关闭避免告警噪声；
    4. 结果统一落在 runs/train/<name>/，权重文件为 weights/best.pt。
"""

import argparse
import os
import sys

from ultralytics import YOLO

# 项目内的预训练权重位置；不存在时 ultralytics 会自动按名字联网下载
DEFAULT_MODEL = os.path.join("models", "yolov8n.pt")

# 数据集配置文件，由 scripts/split_dataset.py 生成
DEFAULT_DATA = os.path.join("data", "seat_dataset", "data.yaml")


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数，全部带默认值，可直接无参运行。"""
    parser = argparse.ArgumentParser(description="训练 YOLOv8 座位与物品检测模型")

    parser.add_argument("--model", default=DEFAULT_MODEL, help="基础权重路径，默认 models/yolov8n.pt")
    parser.add_argument("--data", default=DEFAULT_DATA, help="数据集配置路径，默认 data/seat_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数，默认 100")
    parser.add_argument("--imgsz", type=int, default=640, help="训练输入尺寸，默认 640")
    parser.add_argument("--batch", type=int, default=8, help="批大小，默认 8")
    parser.add_argument("--device", default="cpu", help="训练设备，默认 cpu；有显卡可改为 0")
    parser.add_argument("--name", default="seat_v1", help="本次训练的实验名，默认 seat_v1")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42，保证结果可复现")
    return parser.parse_args()


def main() -> int:
    """执行训练并输出关键结论。"""
    arguments = parse_arguments()

    if not os.path.isfile(arguments.data):
        print(f"[错误] 数据集配置不存在：{arguments.data}")
        print("       请先运行：python scripts/split_dataset.py")
        return 1

    print(f"[训练] 基础权重：{arguments.model}")
    print(f"[训练] 数据集　：{arguments.data}")
    print(f"[训练] 轮数={arguments.epochs} 尺寸={arguments.imgsz} 批={arguments.batch} 设备={arguments.device}")

    model = YOLO(arguments.model)

    # 必须传绝对路径：ultralytics 会把相对 project 拼到全局 runs_dir 之下，产生 runs/detect/runs 的多层嵌套
    train_project = os.path.abspath(os.path.join("runs", "train"))
    val_project = os.path.abspath(os.path.join("runs", "val"))

    # 训练过程：数据增强采用 mosaic + 小角度旋转 + 颜色抖动，适配不同机房光照
    train_results = model.train(
        data=arguments.data,
        epochs=arguments.epochs,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=arguments.device,
        workers=0,
        amp=False,
        seed=arguments.seed,
        patience=40,
        cache="ram",
        project=train_project,
        name=arguments.name,
        exist_ok=True,
        verbose=True,
        plots=True,
        degrees=5.0,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=10,
    )

    # 训练结束后立刻在测试集上评估一次，给出 mAP 供判断模型可用性
    print("\n[评估] 开始测试集评估 ...")
    test_metrics = model.val(
        data=arguments.data,
        split="test",
        imgsz=arguments.imgsz,
        device=arguments.device,
        workers=0,
        project=val_project,
        name=arguments.name + "_test",
        exist_ok=True,
    )

    # 权重目录以训练器实际保存位置为准，避免 project 路径解析差异导致取错
    save_dir = str(model.trainer.save_dir)
    best_weight = os.path.join(save_dir, "weights", "best.pt")
    print("\n================ 训练完成 ================")
    print(f"结果目录　：{save_dir}")
    print(f"最佳权重　：{best_weight}（存在={os.path.isfile(best_weight)}）")
    print(f"整体 mAP50：{test_metrics.box.map50:.4f}")
    print(f"整体 mAP50-95：{test_metrics.box.map:.4f}")

    # 逐类指标，便于定位哪一类还需要补样本
    class_names = test_metrics.names
    for index, class_map in enumerate(test_metrics.box.maps):
        print(f"  {class_names[index]:8s} mAP50-95 = {class_map:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
