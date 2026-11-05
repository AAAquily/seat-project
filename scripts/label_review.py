"""给预标注结果生成带编号的预览图，便于人工核对后按编号删框。

用法：
    python scripts/label_review.py --image data/workspace/images/xxx.jpg \
        --label data/workspace/labels/xxx.txt --out data/prelabel_review/xxx.jpg

生成图中每个框左上角标注其行号（从 1 开始，与标签文件行一一对应）。
配合 --delete "2,5,7" 可直接删除指定行号的框。
"""

import argparse
import os

import cv2

CLASS_COLORS = {
    "person": (60, 60, 220),
    "book": (60, 180, 60),
    "bag": (200, 120, 40),
    "cup": (200, 60, 200),
    "item": (200, 200, 60),
}


def load_classes(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return ["person", "book", "bag", "cup", "item"]


def draw(image_path, label_path, classes):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        print("[无标签]", label_path)
        return img, []
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for i, ln in enumerate(f, 1):
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = (float(x) for x in p[1:])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            boxes.append((i, c, x1, y1, x2, y2))
    for i, c, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(classes[c], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f"#{i} {classes[c]}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = y1 - 6 if y1 - 6 > 12 else y2 + 18
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), color, -1)
        cv2.putText(img, tag, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img, boxes


def delete_lines(label_path, kill):
    kept = []
    with open(label_path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, ln in enumerate(lines, 1):
        if i not in kill:
            kept.append(ln)
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(kept)
    print(f"[删除] {label_path} 移除 {sorted(kill)}，剩 {len(kept)} 框")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="data/workspace/classes.txt")
    ap.add_argument("--delete", default="", help="逗号分隔的行号，如 2,5,7")
    args = ap.parse_args()

    classes = load_classes(args.classes)
    if args.delete:
        kill = {int(x) for x in args.delete.split(",") if x.strip()}
        delete_lines(args.label, kill)

    img, boxes = draw(args.image, args.label, classes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, img)
    print(f"[预览] {args.out}，共 {len(boxes)} 框")


if __name__ == "__main__":
    main()
