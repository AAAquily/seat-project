# -*- coding: utf-8 -*-
"""构建 v8 增强训练集（data/seat_dataset_v8）。

组成（全部来自 train 划分，不碰 val / test，杜绝评测泄漏）：
  1. 原 29 张 train 图 + 标签（已应用 fix_labels_v2 修正）；
  2. 2×2 瓦片裁剪 + 标签重映射：小人目标（密集教室、远桌）在瓦片里等效放大 2 倍，
     直接针对 f0011 / f0007 这类「小目标漏检」；
  3. 海报 / 照片墙负样本裁剪（空标签）：针对 real_02 / real_08 的证件照误检；
  4. 少量无目标瓦片作为背景负样本（上限 12 张）。

用法：
    python scripts/build_dataset_v8.py            # 生成（可重复执行，先清空输出目录）
"""

import os
import random
import shutil

import cv2

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)

SRC = "data/seat_dataset"
DST = "data/seat_dataset_v8"

# 2×2 瓦片：目标框中心在瓦片内才保留，框坐标裁到瓦片边界
TILE_GRID = 2
MIN_PERSON_IN_TILE = 1       # 瓦片至少含 1 个 person 框才收（纯背景瓦片单独限量）
MAX_EMPTY_TILES = 12         # 背景负样本瓦片上限

# 海报 / 照片墙负样本裁剪区（人工核对过：区域内没有真人）
# 坐标为 (文件名, [x1, y1, x2, y2])，来自 train 划分的图
NEGATIVE_CROPS = [
    ("20261105_real_01", [1430, 0, 1706, 420]),    # 右上「团队介绍」证件照墙
    ("20261105_real_01", [1250, 40, 1706, 300]),   # 蓝色海报板 + 证件照
    ("20261105_real_02", [1450, 0, 1706, 500]),    # 右墙证件照海报 + 白板
    ("20261105_real_02", [1560, 0, 1706, 380]),    # 证件照海报特写
]

random.seed(20261106)


def load_labels(stem):
    for split in ("train", "val", "test"):
        p = os.path.join(SRC, "labels", split, stem + ".txt")
        if os.path.isfile(p):
            rows = []
            for line in open(p, encoding="utf-8"):
                parts = line.split()
                if len(parts) >= 5:
                    rows.append([int(parts[0])] + [float(v) for v in parts[1:5]])
            return rows
    return []


def remap(rows, x0, y0, tw, th, sw, sh):
    """把归一化标签重映射到瓦片坐标系，框与瓦片相交 ≥ 60% 才保留。"""
    out = []
    for cls, cx, cy, bw, bh in rows:
        px1, py1 = (cx - bw / 2) * sw, (cy - bh / 2) * sh
        px2, py2 = px1 + bw * sw, py1 + bh * sh
        ix1, iy1 = max(px1, x0), max(py1, y0)
        ix2, iy2 = min(px2, x0 + tw), min(py2, y0 + th)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        if iw * ih < 0.6 * max(1e-6, (px2 - px1) * (py2 - py1)):
            continue
        ncx = ((ix1 + ix2) / 2 - x0) / tw
        ncy = ((iy1 + iy2) / 2 - y0) / th
        nbw = (ix2 - ix1) / tw
        nbh = (iy2 - iy1) / th
        if nbw <= 0.004 or nbh <= 0.004:
            continue
        out.append([cls, round(ncx, 6), round(ncy, 6), round(nbw, 6), round(nbh, 6)])
    return out


def write_label(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(" ".join(str(v) for v in row) + "\n")


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(DST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", split), exist_ok=True)

    stats = {"full": 0, "tiles": 0, "neg_crops": 0, "empty_tiles": 0}

    # 1) 原 train 全图
    train_dir = os.path.join(SRC, "images", "train")
    for name in sorted(os.listdir(train_dir)):
        stem, ext = os.path.splitext(name)
        shutil.copy2(os.path.join(train_dir, name), os.path.join(DST, "images", "train", name))
        rows = load_labels(stem)
        write_label(os.path.join(DST, "labels", "train", stem + ".txt"), rows)
        stats["full"] += 1

    # 2) 2×2 瓦片 + 4) 背景负样本瓦片
    empty_pool = []
    for name in sorted(os.listdir(train_dir)):
        stem, _ = os.path.splitext(name)
        img = cv2.imread(os.path.join(train_dir, name))
        if img is None:
            continue
        height, width = img.shape[:2]
        rows = load_labels(stem)
        tw, th = width // TILE_GRID, height // TILE_GRID
        for gy in range(TILE_GRID):
            for gx in range(TILE_GRID):
                x0, y0 = gx * tw, gy * th
                tile = img[y0 : y0 + th, x0 : x0 + tw]
                if tile.size == 0:
                    continue
                trows = remap(rows, x0, y0, tw, th, width, height)
                has_person = any(r[0] == 0 for r in trows)
                has_obj = any(r[0] in (1, 2, 3, 4) for r in trows)
                tile_name = f"{stem}_t{gy}{gx}.jpg"
                if has_person or has_obj:
                    cv2.imwrite(os.path.join(DST, "images", "train", tile_name), tile)
                    write_label(os.path.join(DST, "labels", "train", tile_name.replace(".jpg", ".txt")), trows)
                    stats["tiles"] += 1
                else:
                    empty_pool.append((tile, tile_name))

    for tile, tile_name in random.sample(empty_pool, min(MAX_EMPTY_TILES, len(empty_pool))):
        cv2.imwrite(os.path.join(DST, "images", "train", tile_name), tile)
        write_label(os.path.join(DST, "labels", "train", tile_name.replace(".jpg", ".txt")), [])
        stats["empty_tiles"] += 1

    # 3) 海报负样本
    for stem, (x1, y1, x2, y2) in NEGATIVE_CROPS:
        src_img = os.path.join(train_dir, stem + ".jpg")
        img = cv2.imread(src_img)
        if img is None:
            print(f"[跳过] 负样本源图不存在 {stem}")
            continue
        height, width = img.shape[:2]
        crop = img[y1:min(y2, height), x1:min(x2, width)]
        if crop.size == 0:
            continue
        rows = [r for r in remap(load_labels(stem), x1, y1, x2 - x1, y2 - y1, width, height) if r[0] == 0]
        if rows:
            print(f"[警告] 负样本裁剪区 {stem}{(x1, y1, x2, y2)} 内含 person 框，请核对区域！")
        name = f"{stem}_neg{x1}y{y1}.jpg"
        cv2.imwrite(os.path.join(DST, "images", "train", name), crop)
        write_label(os.path.join(DST, "labels", "train", name.replace(".jpg", ".txt")), [])
        stats["neg_crops"] += 1

    # val / test 原样复制（real_08 标签已修正）
    for split in ("val", "test"):
        for name in sorted(os.listdir(os.path.join(SRC, "images", split))):
            stem, ext = os.path.splitext(name)
            shutil.copy2(os.path.join(SRC, "images", split, name), os.path.join(DST, "images", split, name))
            write_label(os.path.join(DST, "labels", split, stem + ".txt"), load_labels(stem))

    # data.yaml（相对路径，yaml 所在目录即数据集根）
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as handle:
        handle.write(
            "train: images/train\nval: images/val\ntest: images/test\n"
            "nc: 6\nnames: ['person', 'book', 'bag', 'cup', 'item', 'stool']\n"
        )

    print("\n===== v8 数据集构建完成 =====")
    print(stats)
    for split in ("train", "val", "test"):
        n_img = len(os.listdir(os.path.join(DST, "images", split)))
        n_lbl = len(os.listdir(os.path.join(DST, "labels", split)))
        print(f"{split}: images={n_img} labels={n_lbl}")


if __name__ == "__main__":
    main()
