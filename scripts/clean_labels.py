# -*- coding: utf-8 -*-
"""清理数据集标签：合并同类重复框（IoU>0.6）、删除 real 系列噪声小框。

规则（对应「标注框杂乱」的模型层根因）：
1. 同类框 IoU>0.6 → 合并为并集框（一人两框教模型输出碎片）；
2. real_* 标签中面积 < 0.0008 的物品/书/杯框 → 删除（大图近景下无对应实体，
   属于早期标注噪声，教模型输出密集小灰框；网图俯瞰小目标保留）；
3. 俯瞰图过扁人框保留（俯视合法）。

用法：python scripts/clean_labels.py [--apply]
不带 --apply 仅打印将发生的变更。
"""
import glob
import os
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIRS = [
    os.path.join(PROJECT, "data", "seat_dataset", "labels", "train"),
    os.path.join(PROJECT, "data", "seat_dataset", "labels", "val"),
    os.path.join(PROJECT, "data", "seat_dataset", "labels", "test"),
]
DUP_IOU = 0.60
TINY_AREA = 0.0008


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def union_box(a, b):
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    x2, y2 = max(ax2, bx2), max(ay2, by2)
    cls = a[0]
    return (cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def clean(boxes, is_real):
    boxes = [tuple(b) for b in boxes]
    changed = []

    # 1) 合并同类重复框（贪心：IoU 最大的对先并）
    merged = True
    while merged:
        merged = False
        best_pair, best_val = None, DUP_IOU
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] != boxes[j][0]:
                    continue
                v = iou(boxes[i], boxes[j])
                if v > best_val:
                    best_pair, best_val = (i, j), v
        if best_pair:
            i, j = best_pair
            changed.append(f"合并重复框 cls={boxes[i][0]} IoU={best_val:.2f}")
            boxes[i] = union_box(boxes[i], boxes[j])
            del boxes[j]
            merged = True

    # 2) real 系列的微小物品框直接删除
    if is_real:
        kept = []
        for b in boxes:
            if b[0] != 0 and b[3] * b[4] < TINY_AREA:
                changed.append(f"删噪声小框 cls={b[0]} wh={b[3]:.4f},{b[4]:.4f}")
                continue
            kept.append(b)
        boxes = kept

    return boxes, changed


def main():
    apply = "--apply" in sys.argv
    total_changes = 0
    for label_dir in LABEL_DIRS:
        for path in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
            name = os.path.basename(path)
            lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
            boxes = []
            for ln in lines:
                p = ln.split()
                if len(p) >= 5:
                    boxes.append((int(p[0]), *[float(x) for x in p[1:5]]))
            if not boxes:
                continue
            new_boxes, changed = clean(boxes, is_real="real_" in name)
            if not changed:
                continue
            total_changes += len(changed)
            print(f"{os.path.basename(label_dir)}/{name}: " + "; ".join(changed))
            if apply:
                with open(path, "w", encoding="utf-8") as f:
                    for b in new_boxes:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
    print(f"\n共 {total_changes} 处变更{'，已写入' if apply else '（dry-run，加 --apply 生效）'}")


if __name__ == "__main__":
    main()
