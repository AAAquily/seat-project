# -*- coding: utf-8 -*-
"""标签修正 v2（2026-11-06 框级审计结论）。

审计方法：scripts/diagnose_boxes.py 框级比对 + 重叠框对可视化人工复核
（output/diagnose/label_audit/ 每一对重叠框都放大看过）。

发现的问题与处理：
  1. 同一人被标两个框（半身框 + 全身框）→ 删半身框，保留全身框
     （real_01 还有一人三框：头 / 半身 / 全身，只留全身）；
  2. 一个框盖住两个人（漏拆）→ 删该框（两人已有各自的独立框）；
  3. 海报证件照被标成 person → 删（后续用海报负样本训练抑制该误检）。

删除清单全部按「像素坐标 ±4px」匹配，防止误删；原始标签备份到
data/seat_dataset/labels_backup_20261106_v2/。

用法：
    python scripts/fix_labels_v2.py            # 预览（不写文件）
    python scripts/fix_labels_v2.py --apply    # 应用（先备份）
"""

import argparse
import os
import shutil

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

LABEL_DIRS = ["data/seat_dataset/labels/train", "data/seat_dataset/labels/val", "data/seat_dataset/labels/test"]
BACKUP = "data/seat_dataset/labels_backup_20261106_v2"

# 每项：文件名（不含扩展名）、要删除的人框像素坐标 [x1,y1,x2,y2]、原因
REMOVALS = [
    # ---- real_01：一人三框（留全身 106x203），另有一个盖两人的松散大框 ----
    ("20261105_real_01", [1095, 451, 1162, 568], "同一人半身框（保留全身框）"),
    ("20261105_real_01", [1096, 452, 1162, 523], "同一人头部框（保留全身框）"),
    ("20261105_real_01", [43, 606, 696, 1273], "一个框盖住两个人（两人已有独立框）"),
    # ---- real_02：三组半身/全身双框 + 一个跨到椅子的过度延伸框 ----
    ("20261105_real_02", [1136, 572, 1230, 707], "同一人半身框（保留全身框）"),
    ("20261105_real_02", [640, 549, 744, 641], "同一人半身框（保留全身框）"),
    ("20261105_real_02", [1199, 618, 1515, 1249], "同一人过度延伸框（保留躯干框）"),
    ("20261105_real_02", [770, 558, 847, 760], "同一人窄框（保留全身框）"),
    ("20261105_real_02", [1595, 310, 1659, 436], "海报证件照误标为 person"),
    # ---- real_04 / real_05 / real_07：半身+全身双框 ----
    ("20261105_real_04", [1170, 530, 1259, 641], "同一人半身框（保留全身框）"),
    ("20261105_real_05", [931, 544, 990, 615], "同一人半身框（保留全身框）"),
    ("20261105_real_05", [831, 537, 890, 614], "同一人半身框（保留全身框）"),
    ("20261105_real_07", [1348, 601, 1504, 874], "同一人连背包的松散框（保留人体框）"),
    # ---- real_08：双框重复 + 盖两人的合并框 + 海报证件照 ----
    ("20261105_real_08", [136, 615, 447, 945], "一个框盖住两个人（两人已有独立框）"),
    ("20261105_real_08", [1517, 615, 1706, 776], "同一人半身框（保留全身框）"),
    ("20261105_real_08", [1404, 419, 1458, 488], "海报证件照误标为 person"),
    # ---- f0009：俯瞰图一处双框 ----
    ("20261104_0900_web_f0009", [1646, 1603, 1709, 1672], "同一人半身框（保留全身框）"),
]

TOL = 4.0  # 坐标匹配容差（像素）


def label_path(stem):
    for d in LABEL_DIRS:
        p = os.path.join(d, stem + ".txt")
        if os.path.isfile(p):
            return p
    return None


def image_size(stem):
    import cv2

    for d in ["data/seat_dataset/images/train", "data/seat_dataset/images/val", "data/seat_dataset/images/test"]:
        p = os.path.join(d, stem + ".jpg")
        if os.path.isfile(p):
            img = cv2.imread(p)
            return img.shape[1], img.shape[0]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真正写回标签文件（默认只预览）")
    args = parser.parse_args()

    # 按文件分组
    by_file = {}
    for stem, box, reason in REMOVALS:
        by_file.setdefault(stem, []).append((box, reason))

    if args.apply:
        os.makedirs(BACKUP, exist_ok=True)

    total_removed = 0
    for stem, items in sorted(by_file.items()):
        path = label_path(stem)
        if path is None:
            print(f"[跳过] {stem} 找不到标签")
            continue
        width, height = image_size(stem)
        lines = open(path, encoding="utf-8").read().splitlines()
        kept, removed = [], []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and int(parts[0]) == 0:
                cx, cy, bw, bh = map(float, parts[1:5])
                box = [(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height]
                hit = None
                for target, reason in items:
                    if all(abs(box[i] - target[i]) <= TOL for i in range(4)):
                        hit = (target, reason)
                        break
                if hit:
                    removed.append((box, hit[1]))
                    continue
            kept.append(line)

        status = "应用" if args.apply else "预览"
        print(f"[{status}] {stem}: 删 {len(removed)} / 共 {len(lines)} 行")
        for box, reason in removed:
            print(f"    - {[round(v) for v in box]}  {reason}")
        total_removed += len(removed)

        if args.apply:
            backup = os.path.join(BACKUP, os.path.basename(path))
            if not os.path.exists(backup):
                shutil.copy2(path, backup)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(kept) + ("\n" if kept else ""))

    print(f"\n合计删除 {total_removed} 个框（{'已写回，备份在 ' + BACKUP if args.apply else '预览模式，加 --apply 生效'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
