# -*- coding: utf-8 -*-
"""9 张真实图的座位判定证据明细：逐座打印人/物证据框，用于与人工标注对账。"""
import sys

import cv2
import yaml

sys.path.insert(0, "scripts")
sys.path.insert(0, __file__.rsplit("scripts", 1)[0] + "scripts")
sys.stdout.reconfigure(encoding="utf-8")

import judge_seats as J  # noqa: E402
from scene_match import match_scene, resolve_scene_seats  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMAGES = sys.argv[1:] or [f"data/raw_real/real_{i:02d}.jpg" for i in range(1, 10)]

cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
models = {}


def model_of(path):
    if path not in models:
        models[path] = YOLO(path)
    return models[path]


for path in IMAGES:
    img = cv2.imread(path)
    matched, ref, inl = match_scene(img, cfg, ".")
    seats = resolve_scene_seats(cfg, ref) if matched else cfg["seats"]
    weights = cfg["model"]["fixed_weights"] if matched else cfg["model"]["weights"]
    conf = cfg["model"]["scene_conf"] if matched else cfg["model"]["conf"]
    r = model_of(weights).predict(img, conf=conf, iou=0.45, imgsz=cfg["model"]["imgsz"],
                                 device="cpu", verbose=False)[0]
    dets = [{"class_id": int(b.cls[0]), "conf": float(b.conf[0]), "box": [int(v) for v in b.xyxy[0]]}
            for b in r.boxes]
    dets = J.filter_tiny_persons(
        dets, float(cfg["occupancy"].get("min_person_box_height", 0)),
        (img.shape[1], img.shape[0]))
    audit = {}
    res = J.judge_seats(dets, seats, cfg["occupancy"], cfg.get("classes"), image=img, audit=audit)
    print(f"\n===== {path}  参考={ref.split('/')[-1]} 内点={inl} 权重={weights.split('/')[-3]} conf={conf}")
    for it in res:
        pe = " ".join(f"{h['match']}@{h['conf']}{h['box']}" for h in it["person_evidence"])
        oe = " ".join(f"{h['conf']}{h['box']}" for h in it["evidence"])
        print(f"  {it['id']:4s} {it['state_cn']:6s} 人={it['person_count']} 物={it['object_count']}"
              f"  人证[{pe}]  物证[{oe}]")
    if audit.get("removed_spills"):
        print(f"  剔除碎片: {audit['removed_spills']}")
