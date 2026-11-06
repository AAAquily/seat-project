# -*- coding: utf-8 -*-
"""框级诊断：把检测结果与人工标注（GT）逐框比对，定位「漏检 / 误检 / 重复 / 框不准」。

与 batch_test.py 的区别：
    batch_test 只比「人数」，本脚本比「框」——输出 precision / recall / 匹配 IoU，
    并把人漏检的具体位置、尺寸、置信度列出来，便于针对性补数据或调参。

用法：
    python scripts/diagnose_boxes.py                    # 28 张全量
    python scripts/diagnose_boxes.py --only real_02 real_08 20261104_0900_web_f0011
    python scripts/diagnose_boxes.py --conf 0.25 --weights runs/train/seat_v8/weights/best.pt
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from dynamic_seats import analyze_image  # noqa: E402

LABEL_DIRS = [
    "data/seat_dataset/labels/train",
    "data/seat_dataset/labels/val",
    "data/seat_dataset/labels/test",
]
IMAGE_DIRS = [
    "data/raw",
    "data/raw_real",
    "data/seat_dataset/images/train",
    "data/seat_dataset/images/val",
    "data/seat_dataset/images/test",
]
OUT_DIR = "output/diagnose"

WEB_IMAGES = [f"20261104_0900_web_f{i:04d}.jpg" for i in range(1, 20)]
REAL_IMAGES = [f"20261105_real_{i:02d}.jpg" for i in range(1, 10)]

# real_08 的基准是人工目测（数据集里没有该图的标签）
MANUAL_NO_LABEL = {"20261105_real_08.jpg"}


def find_image(name):
    for d in IMAGE_DIRS:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return path
    return None


def load_gt(name, width, height):
    """读取 YOLO 标签，转成像素框；返回 [(cls, [x1,y1,x2,y2])]。"""
    stem = os.path.splitext(name)[0]
    for d in LABEL_DIRS:
        path = os.path.join(d, stem + ".txt")
        if os.path.isfile(path):
            boxes = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, w, h = parts[0], *map(float, parts[1:5])
                    boxes.append(
                        (
                            int(cls),
                            [
                                (cx - w / 2) * width,
                                (cy - h / 2) * height,
                                (cx + w / 2) * width,
                                (cy + h / 2) * height,
                            ],
                        )
                    )
            return boxes
    return None


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(dets, gts, thr=0.5):
    """贪心匹配，返回 (匹配对, 未匹配 det 索引, 未匹配 gt 索引)。"""
    pairs = []
    used_d, used_g = set(), set()
    order = sorted(
        ((iou(d, g), di, gi) for di, d in enumerate(dets) for gi, g in enumerate(gts)),
        reverse=True,
    )
    for score, di, gi in order:
        if score < thr or di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        pairs.append((di, gi, score))
    return (
        pairs,
        [i for i in range(len(dets)) if i not in used_d],
        [i for i in range(len(gts)) if i not in used_g],
    )


def draw(image, persons, gt_persons, missed, extra, path):
    canvas = image.copy()
    for gt in gt_persons:
        cv2.rectangle(canvas, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 200, 0), 2)
    for p in persons:
        b = p["box"]
        cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
    for gt in missed:
        b = gt
        cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 255), 3)
        cv2.putText(canvas, "MISS", (int(b[0]), max(12, int(b[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 255), 2)
    cv2.putText(canvas, f"GT(green)={len(gt_persons)} DET(red)={len(persons)} MISS(purple)={len(missed)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(path, canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="", help="默认取 config 的 model.weights")
    parser.add_argument("--conf", type=float, default=-1, help="默认取 config")
    parser.add_argument("--only", nargs="*", default=[], help="只诊断指定文件名（不含扩展名可模糊匹配）")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--out-dir", default=OUT_DIR, help="输出目录（便于新旧模型分开存）")
    arguments = parser.parse_args()
    out_dir = arguments.out_dir

    config = yaml.safe_load(open("config/config.yaml", "r", encoding="utf-8"))
    weights = arguments.weights or config["model"]["weights"]
    device = config["model"].get("device", "cpu")
    seat_model = YOLO(weights)
    coco_model = YOLO("models/yolov8n.pt")

    os.makedirs(out_dir, exist_ok=True)
    names = WEB_IMAGES + REAL_IMAGES
    if arguments.only:
        names = [n for n in names if any(key in n for key in arguments.only)]

    rows = []
    for name in names:
        path = find_image(name)
        if path is None:
            print(f"[跳过] {name}")
            continue
        image = cv2.imread(path)
        height, width = image.shape[:2]
        gt_boxes = load_gt(name, width, height)
        if gt_boxes is None:
            print(f"[无标签] {name}（人工目测基准，跳过框级比对）")
            continue
        gt_persons = [b for cls, b in gt_boxes if cls == 0]
        gt_objects = [b for cls, b in gt_boxes if cls in (1, 2, 3, 4)]

        detection, _ = analyze_image(image, seat_model, coco_model, device=device)
        persons = detection["persons"]
        det_boxes = [p["box"] for p in persons]
        object_boxes = [o["box"] for o in detection["objects"]]

        pairs, missed_det, missed_gt = match(det_boxes, gt_persons, arguments.iou_thr)
        obj_pairs, obj_fp, obj_missed = match(object_boxes, gt_objects, arguments.iou_thr)
        mean_iou = float(np.mean([p[2] for p in pairs])) if pairs else 0.0

        missed_detail = []
        for gi in missed_gt:
            b = gt_persons[gi]
            missed_detail.append(
                {
                    "box": [round(v) for v in b],
                    "w": round(b[2] - b[0]),
                    "h": round(b[3] - b[1]),
                    "area_ratio": round((b[2] - b[0]) * (b[3] - b[1]) / (width * height) * 100, 3),
                }
            )

        stem = os.path.splitext(name)[0]
        draw(image, persons, gt_persons, [gt_persons[i] for i in missed_gt], [], os.path.join(out_dir, f"{stem}_框比对.jpg"))

        row = {
            "image": name,
            "gt_persons": len(gt_persons),
            "det_persons": len(persons),
            "tp": len(pairs),
            "fp": len(missed_det),
            "fn": len(missed_gt),
            "person_precision": round(len(pairs) / max(1, len(persons)), 3),
            "person_recall": round(len(pairs) / max(1, len(gt_persons)), 3),
            "mean_matched_iou": round(mean_iou, 3),
            "object_gt": len(gt_objects),
            "object_det": len(object_boxes),
            "object_tp": len(obj_pairs),
            "object_missed": len(obj_missed),
            "mode": detection["mode"],
            "missed": missed_detail,
        }
        rows.append(row)
        print(
            f"{name}  GT人={len(gt_persons):3d} 检出={len(persons):3d}  "
            f"TP={len(pairs):3d} FP={len(missed_det):3d} FN={len(missed_gt):3d}  "
            f"IoU均={mean_iou:.2f}  物 GT={len(gt_objects)} 检出={len(object_boxes)}  "
            f"模式={detection['mode']}"
        )

    if not rows:
        print("没有可诊断的图片（多数图无标签）")
        return 0

    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    summary = {
        "images": len(rows),
        "person_tp": total_tp,
        "person_fp": total_fp,
        "person_fn": total_fn,
        "person_precision": round(total_tp / max(1, total_tp + total_fp), 4),
        "person_recall": round(total_tp / max(1, total_tp + total_fn), 4),
        "person_f1": round(2 * total_tp / max(1, 2 * total_tp + total_fp + total_fn), 4),
        "mean_matched_iou": round(float(np.mean([r["mean_matched_iou"] for r in rows])), 4),
        "object_gt": sum(r["object_gt"] for r in rows),
        "object_det": sum(r["object_det"] for r in rows),
        "object_tp": sum(r["object_tp"] for r in rows),
        "rows": rows,
    }
    with open(os.path.join(out_dir, "box_diagnose.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        f"\n===== 框级汇总：P={summary['person_precision']:.3f} R={summary['person_recall']:.3f} "
        f"F1={summary['person_f1']:.3f} 匹配IoU={summary['mean_matched_iou']:.3f} "
        f"（TP{total_tp}/FP{total_fp}/FN{total_fn}）====="
    )
    print(f"物品：GT {summary['object_gt']} 检出 {summary['object_det']} 命中 {summary['object_tp']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
