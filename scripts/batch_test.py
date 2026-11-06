# -*- coding: utf-8 -*-
"""28 张自习室图片批量测试：动态座位识别模式。

图片 = 原有 19 张网络自习室图 + 9 张真实自习室照片。
人工基准（Ground Truth）：
    - 数据集 YOLO 标注里的 person 框数（用户逐张标注）；
    - 无标注的 3 张用人工目测值（f0001=0 空教室、f0010≈75 图书馆俯瞰、real_08=4）。

判定标准（与"人数优先"一致）：
    每张图输出 全图人数 与 动态座位三态；人数与基准一致即该图通过：
      基准 < 10 人时允许 ±2 人误差；
      基准 >= 10 人时允许 ±20% 相对误差（百人级俯瞰图不可能个位精确）。

用法：
    python scripts/batch_test.py            # 全量 28 张
    python scripts/batch_test.py --limit 5  # 快速冒烟
"""

import argparse
import json
import os
import sys
import time

import cv2
import yaml

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from dynamic_seats import analyze_image, draw_dynamic, total_person_count  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATASET = "data/seat_dataset"
LABEL_DIRS = [os.path.join(DATASET, "labels/train"), os.path.join(DATASET, "labels/test")]
IMAGE_DIRS = ["data/raw", "data/raw_real", os.path.join(DATASET, "images/train"), os.path.join(DATASET, "images/test")]
OUT_DIR = "output/batch_28"

# 19 张网络图 + 9 张真实照片
WEB_IMAGES = [f"20261104_0900_web_f{i:04d}.jpg" for i in range(1, 20)]
REAL_IMAGES = [f"20261105_real_{i:02d}.jpg" for i in range(1, 10)]

# 无标注图片的人工目测基准（2026-11-06 逐张放大核对）
MANUAL_GT = {
    "20261104_0900_web_f0001.jpg": 0,    # 空自习室
    "20261104_0900_web_f0010.jpg": 75,   # 图书馆俯瞰，人工估计 70~80 取中值
    "20261105_real_08.jpg": 4,           # 左 2 人 + 白帽衫 1 人 + 右桌 1 人
}

SCENE_NOTES = {
    "20261104_0900_web_f0001.jpg": "空隔断自习室",
    "20261104_0900_web_f0002.jpg": "空隔断自习室",
    "20261104_0900_web_f0003.jpg": "空隔断自习室",
    "20261104_0900_web_f0004.jpg": "隔断自习室有人",
    "20261104_0900_web_f0005.jpg": "大教室成排座位",
    "20261104_0900_web_f0006.jpg": "大教室成排座位",
    "20261104_0900_web_f0007.jpg": "自习室斜视",
    "20261104_0900_web_f0008.jpg": "书架图书馆近景",
    "20261104_0900_web_f0009.jpg": "图书馆多层俯瞰",
    "20261104_0900_web_f0010.jpg": "图书馆俯瞰",
    "20261104_0900_web_f0011.jpg": "低分辨率密集教室",
    "20261104_0900_web_f0012.jpg": "空自习区",
    "20261104_0900_web_f0013.jpg": "自习室中景",
    "20261104_0900_web_f0014.jpg": "开放自习区",
    "20261104_0900_web_f0015.jpg": "宽敞自习室",
    "20261104_0900_web_f0016.jpg": "大型自习厅",
    "20261104_0900_web_f0017.jpg": "图书馆多层俯瞰（最密）",
    "20261104_0900_web_f0018.jpg": "图书馆斜俯瞰",
    "20261104_0900_web_f0019.jpg": "教室斜视",
    "20261105_real_01.jpg": "真实自习室",
    "20261105_real_02.jpg": "真实自习室（ROI 标定底图）",
    "20261105_real_03.jpg": "真实自习室",
    "20261105_real_04.jpg": "真实自习室",
    "20261105_real_05.jpg": "真实自习室",
    "20261105_real_06.jpg": "真实自习室",
    "20261105_real_07.jpg": "真实自习室",
    "20261105_real_08.jpg": "真实自习室（含证件照墙）",
    "20261105_real_09.jpg": "真实自习室",
}


def find_image(name):
    for d in IMAGE_DIRS:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    if name == "20261105_real_08.jpg" and os.path.isfile("data/raw_real/real_08.jpg"):
        return "data/raw_real/real_08.jpg"
    return None


def gt_person_count(name):
    if name in MANUAL_GT:
        return MANUAL_GT[name], "人工目测"
    for d in LABEL_DIRS:
        path = os.path.join(d, os.path.splitext(name)[0] + ".txt")
        if os.path.isfile(path):
            count = 0
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if parts and parts[0] == "0":
                        count += 1
            return count, "数据集标注"
    return None, "无基准"


def is_correct(gt, detected):
    if gt < 10:
        return abs(detected - gt) <= 2
    return abs(detected - gt) / gt <= 0.20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只测前 N 张（冒烟用）")
    parser.add_argument("--weights", default="", help="覆盖 config 的模型权重（A/B 对比用）")
    parser.add_argument("--out", default=OUT_DIR, help="结果输出目录")
    arguments = parser.parse_args()

    config = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
    device = config["model"].get("device", "cpu")
    weights = arguments.weights or config["model"]["weights"]
    seat_model = YOLO(weights)
    coco_model = YOLO("models/yolov8n.pt")
    print(f"[配置] 权重={weights}")

    out_dir = arguments.out
    os.makedirs(out_dir, exist_ok=True)
    names = WEB_IMAGES + REAL_IMAGES
    if arguments.limit:
        names = names[: arguments.limit]

    rows = []
    for name in names:
        path = find_image(name)
        gt, gt_source = gt_person_count(name)
        if path is None:
            print(f"[跳过] {name} 图片不存在")
            continue
        started = time.perf_counter()
        image = cv2.imread(path)
        detection, seat_results = analyze_image(image, seat_model, coco_model, device=device)
        elapsed = time.perf_counter() - started

        persons = total_person_count(detection)
        occupied = sum(1 for s in seat_results if s["state"] == "OCCUPIED")
        object_only = sum(1 for s in seat_results if s["state"] == "OBJECT_ONLY")
        empty = sum(1 for s in seat_results if s["state"] == "EMPTY")
        correct = is_correct(gt, persons) if gt is not None else None

        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_标注图.jpg"), draw_dynamic(image, seat_results, detection))
        rows.append(
            {
                "image": name,
                "scene": SCENE_NOTES.get(name, ""),
                "gt_persons": gt,
                "gt_source": gt_source,
                "detected_persons": persons,
                "seats": len(seat_results),
                "occupied": occupied,
                "object_only": object_only,
                "empty": empty,
                "mode": detection["mode"],
                "elapsed_sec": round(elapsed, 1),
                "correct": correct,
            }
        )
        mark = {True: "通过", False: "失败", None: "无基准"}[correct]
        print(
            f"[{mark}] {name}  基准={gt}  检出人数={persons}  "
            f"座位 {len(seat_results)}（有人{occupied}/有物{object_only}/空置{empty}）  "
            f"模式={detection['mode']}  {elapsed:.1f}s"
        )

    # 汇总
    evaluated = [r for r in rows if r["correct"] is not None]
    hits = sum(1 for r in evaluated if r["correct"])
    summary = {
        "total_images": len(rows),
        "evaluated": len(evaluated),
        "passed": hits,
        "accuracy": round(hits / len(evaluated) * 100, 1) if evaluated else 0.0,
        "mae": round(sum(abs(r["detected_persons"] - r["gt_persons"]) for r in evaluated) / len(evaluated), 2)
        if evaluated
        else None,
        "rows": rows,
    }
    with open(os.path.join(out_dir, "batch_results.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"\n===== 总体准确率：{hits}/{len(evaluated)} = {summary['accuracy']}%  平均绝对误差 {summary['mae']} =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
