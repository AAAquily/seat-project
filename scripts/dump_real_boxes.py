# -*- coding: utf-8 -*-
"""导出检测框像素坐标：为 9 张真实图重建场景座位表提供种子。

对每张图分别用 v6（标定通道，conf 0.15）和 v8（conf 0.25）推理，
输出全部检测框（xyxy 像素坐标 + 类别 + 置信度），并画框存图供目检。
"""
import json
import os
import sys

import cv2
import yaml
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")

V6 = "runs/train/seat_v6/weights/best.pt"
V8 = "runs/train/seat_v8/weights/best.pt"
IMGSZ = 1280
OUT_DIR = "output/real9_boxes"
CLASSES = {0: "person", 1: "book", 2: "bag", 3: "cup", 4: "item", 5: "stool"}

os.makedirs(OUT_DIR, exist_ok=True)

with open("config/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

m_v6 = YOLO(V6)
m_v8 = YOLO(V8)

summary = {}
for i in range(1, 10):
    name = f"real_{i:02d}"
    path = f"data/raw_real/{name}.jpg"
    img = cv2.imread(path)
    h, w = img.shape[:2]
    entry = {"size": [w, h]}
    for tag, model, conf in (("v6", m_v6, 0.15), ("v8", m_v8, 0.25)):
        r = model.predict(img, conf=conf, iou=0.45, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        boxes = []
        vis = img.copy()
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            cls_id = int(b.cls[0])
            c = float(b.conf[0])
            boxes.append({"cls": CLASSES.get(cls_id, str(cls_id)), "conf": round(c, 3),
                          "box": [x1, y1, x2, y2]})
            color = (0, 0, 255) if cls_id == 0 else (255, 120, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{CLASSES.get(cls_id, cls_id)} {c:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        entry[tag] = boxes
        cv2.imwrite(f"{OUT_DIR}/{name}_{tag}.jpg", vis)
    summary[name] = entry
    with open(f"{OUT_DIR}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=1)
    print(f"{name} {w}x{h}: v6={len(entry['v6'])}框 v8={len(entry['v8'])}框")

with open(f"{OUT_DIR}/all.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("done ->", OUT_DIR)
