# -*- coding: utf-8 -*-
"""动态座位识别：面向「任意场景自习室图片」的检测与判定模块。

背景：
    原 judge_seats.py 依赖 config.yaml 里手工标定的固定 ROI（对应 real_02 场景），
    换一张不同机位/不同房间的图片就会「全座无检测目标」。本模块不依赖固定 ROI，
    从检测结果里当场推断座位，适配任意拍摄场景。

检测流程（人数优先，2026-11-06）：
    1. 双模型融合：seat_v6（自训练，person + book/bag/cup/item/stool）
       与 COCO 预训练 yolov8n（person + chair），两路 person 框合并去重——
       seat_v6 对本数据域召回高但拥挤场景易碎片化多计，COCO 反之，互补后计数最稳；
    2. 远景自适应：全图推理后若「中位人框高度 < 6% 图高且人数 ≥ 30」，
       判定为远景小人场景（百人级图书馆俯瞰图），改用 1280px 切片 + conf 0.20
       重新推理（切片让小人占据更多像素，召回大幅提升），跨片 NMS 合并；
    3. 去重规则：交集面积 / 较小框面积 > 0.70 视为同一人（按置信度保留较高者），
       该阈值在「拥挤碎片合并」与「紧邻就座不误并」之间取平衡；

座位推断：
    - 座位实体 = COCO chair 检测 + seat_v6 stool 检测（红脚凳规则延续）；
    - 人坐某座位 = 人框与「椅子区+上方 0.8 椅高的落座区」有实质重叠，
      或人框中心落入该区；重叠率按椅子面积计（坐姿时椅子常被人挡住，按人框
      面积算会漏判）；
    - 有物无人 = 椅子无对应人，但其「桌面上方区」（椅子区向上 1.2 椅高）压到
      book/bag/cup/item 框；
    - 人数优先约定：全图人数 total_persons 统计所有去重后的人框，
      不要求每个人都被分配到座位（特殊角度椅子检测不到时人不丢）。

已知局限（28 图回归实测）：
    - 海报/照片墙上的证件照会被当成真人（如 real_08 右上「团队成员介绍」）；
    - 极密集俯瞰图中互相遮挡的人仍会少数（f0011：检出 9 / 实际 15）。
"""

import os

import cv2
import numpy as np

# COCO 类别号：person=0，chair=56
COCO_PERSON_ID = 0
COCO_CHAIR_ID = 56

# seat_v6 类别：0=person 1=book 2=bag 3=cup 4=item 5=stool
V6_PERSON_ID = 0
V6_OBJECT_IDS = (1, 2, 3, 4)
V6_STOOL_ID = 5

# 自适应触发：中位人框高 < 图高 6% 且人数 >= 30 → 远景小人场景，启用切片
TINY_PERSON_HEIGHT_RATIO = 0.06
TINY_PERSON_MIN_COUNT = 30
# 切片参数：1280px 窗口、25% 重叠、切片内 conf 放宽到 0.25
TILE_SIZE = 1280
TILE_OVERLAP = 0.25
TILE_CONF = 0.25
# 切片回退：切片人数 > 全图基准人数 × 1.25 时视为切片过计（木纹/椅背误检），回退全图
TILE_FALLBACK_RATIO = 1.25
# 常规全图推理 conf
BASE_CONF = 0.35
# COCO 通用模型的 person 置信度下限：COCO 对海报证件照会给出 0.2~0.6 的低置信度误检，
# 而业务模型（v8 起）已通过负样本学会拒识海报。该下限只过滤 COCO 来源的 person 框，
# 不影响物品/椅子检测，也不影响业务模型自身的 person 召回。
COCO_PERSON_FLOOR = 0.60
# 低于下限的 COCO person 框若位于「墙面区域」（画面上部 40% 以内）且框高不足图高 12%，
# 判为海报证件照并丢弃；位于画面中下部（桌椅区）的小框多为远景真人，仍然保留。
WALL_TOP_RATIO = 0.40
WALL_MAX_H_RATIO = 0.12
# 低分辨率小图（短边 < 800px）推理 conf：小图上采样后人框本就偏大，
# 高阈值会把中景的人整个滤掉，实测 0.25 更稳（f0019: 0.35→14 人 / 0.25→17 人）
SMALL_IMAGE_MIN_SIDE = 800
SMALL_IMAGE_CONF = 0.25
# 去重阈值：交集/较小框
DEDUP_CONTAIN = 0.70
# 跨切片 NMS 的 IoU 阈值
TILE_NMS_IOU = 0.45

# 座位推断几何参数（相对椅子框尺寸）
SIT_ZONE_UP = 0.8       # 落座区：椅子区向上扩 0.8 椅高（人的躯干在此）
SIT_MIN_CHAIR_COVER = 0.30  # 人框盖住椅子区 >= 30% 椅面积 → 坐这里
DESK_ZONE_UP = 1.2      # 桌面区：椅子区向上扩 1.2 椅高（判「有物无人」）
DESK_MIN_COVER = 0.20   # 物品框与桌面区交集 >= 20% 物品面积 → 归属该座位

STATE_OCCUPIED = "OCCUPIED"
STATE_OBJECT_ONLY = "OBJECT_ONLY"
STATE_EMPTY = "EMPTY"
STATE_LABELS_CN = {STATE_OCCUPIED: "有人", STATE_OBJECT_ONLY: "有物无人", STATE_EMPTY: "空置"}
STATE_COLORS = {  # BGR
    STATE_OCCUPIED: (40, 40, 220),
    STATE_OBJECT_ONLY: (0, 140, 255),
    STATE_EMPTY: (60, 170, 60),
}


# ---------------------------------------------------------------- 基础几何

def box_area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def dedup_boxes(dets, contain=DEDUP_CONTAIN):
    """按置信度降序去重：与已保留框交集/较小框 > contain 视为同人。"""
    kept = []
    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        duplicate = False
        for keep in kept:
            smaller = min(box_area(det["box"]), box_area(keep["box"]))
            if smaller > 0 and intersection(det["box"], keep["box"]) / smaller > contain:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def nms_merge(dets, iou=TILE_NMS_IOU):
    """标准 NMS：用于合并相邻切片检出的同一目标。"""
    kept = []
    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        duplicate = False
        for keep in kept:
            union = box_area(det["box"]) + box_area(keep["box"]) - intersection(det["box"], keep["box"])
            if union > 0 and intersection(det["box"], keep["box"]) / union > iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


# ---------------------------------------------------------------- 推理

def _predict_classes(model, source, conf, imgsz=1280, device="cpu"):
    """跑一次推理，返回 [{"class_id", "conf", "box"}]（像素坐标）。"""
    result = model.predict(source=source, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    out = []
    for box in result.boxes:
        xy = box.xyxy[0].tolist()
        out.append(
            {
                "class_id": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": [xy[0], xy[1], xy[2], xy[3]],
            }
        )
    return out


def _tile_offsets(height, width):
    """生成切片左上角坐标列表（含边界补片）。"""
    stride = int(TILE_SIZE * (1 - TILE_OVERLAP))
    xs = list(range(0, max(1, width - TILE_SIZE) + 1, stride)) or [0]
    ys = list(range(0, max(1, height - TILE_SIZE) + 1, stride)) or [0]
    if xs[-1] + TILE_SIZE < width:
        xs.append(max(0, width - TILE_SIZE))
    if ys[-1] + TILE_SIZE < height:
        ys.append(max(0, height - TILE_SIZE))
    return ys, xs


def _is_poster_like(det, image_height):
    """COCO 低置信度 person 框 + 墙面位置 + 小框 → 海报证件照。"""
    box = det["box"]
    box_h = box[3] - box[1]
    return (
        det["conf"] < COCO_PERSON_FLOOR
        and box[1] < WALL_TOP_RATIO * image_height
        and box_h < WALL_MAX_H_RATIO * image_height
    )


def detect_any_scene(image, seat_model, coco_model, device="cpu"):
    """自适应场景检测：返回统一结构的检测结果。

    返回 dict：
        persons   去重后的人框列表 [{"box","conf"}]
        objects   物品框（book/bag/cup/item）
        stools    凳子框（seat_v6 stool 类）
        chairs    椅子框（COCO chair）
        mode      "full" / "tiled"
    """
    v6_conf = BASE_CONF if min(image.shape[:2]) >= SMALL_IMAGE_MIN_SIDE else SMALL_IMAGE_CONF
    v6_full = _predict_classes(seat_model, image, v6_conf, device=device)
    coco_full = _predict_classes(coco_model, image, v6_conf, device=device)
    for det in coco_full:
        det["source"] = "coco"

    height, width = image.shape[:2]
    person_boxes = [
        d
        for d in v6_full + coco_full
        if d["class_id"] == COCO_PERSON_ID and not _is_poster_like(d, height)
    ]
    persons = dedup_boxes(person_boxes)

    # 远景小人场景 → 切片 + 低阈值重推
    if persons:
        median_h = float(np.median([d["box"][3] - d["box"][1] for d in persons]))
    else:
        median_h = float("inf")
    use_tiled = median_h < TINY_PERSON_HEIGHT_RATIO * height and len(persons) >= TINY_PERSON_MIN_COUNT

    if not use_tiled:
        objects = [d for d in v6_full if d["class_id"] in V6_OBJECT_IDS]
        stools = [d for d in v6_full if d["class_id"] == V6_STOOL_ID]
        chairs = [d for d in coco_full if d["class_id"] == COCO_CHAIR_ID]
        return {"persons": persons, "objects": objects, "stools": stools, "chairs": chairs, "mode": "full"}

    merged = []
    ys, xs = _tile_offsets(height, width)
    for y in ys:
        for x in xs:
            crop = image[y : min(height, y + TILE_SIZE), x : min(width, x + TILE_SIZE)]
            for det in _predict_classes(seat_model, crop, TILE_CONF, device=device) + _predict_classes(
                coco_model, crop, TILE_CONF, device=device
            ):
                box = det["box"]
                det["box"] = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
                merged.append(det)

    tiled_persons = dedup_boxes(
        nms_merge([d for d in merged if d["class_id"] == COCO_PERSON_ID])
    )
    if len(tiled_persons) > len(persons) * TILE_FALLBACK_RATIO:
        # 切片把桌面纹理/椅背误检成人导致人数暴增 → 不可信，回退全图结果
        objects = [d for d in v6_full if d["class_id"] in V6_OBJECT_IDS]
        stools = [d for d in v6_full if d["class_id"] == V6_STOOL_ID]
        chairs = [d for d in coco_full if d["class_id"] == COCO_CHAIR_ID]
        return {"persons": persons, "objects": objects, "stools": stools, "chairs": chairs, "mode": "full"}

    objects = nms_merge([d for d in merged if d["class_id"] in V6_OBJECT_IDS])
    stools = nms_merge([d for d in merged if d["class_id"] == V6_STOOL_ID])
    chairs = nms_merge([d for d in merged if d["class_id"] == COCO_CHAIR_ID])
    return {"persons": tiled_persons, "objects": objects, "stools": stools, "chairs": chairs, "mode": "tiled"}


# ---------------------------------------------------------------- 座位推断

def infer_dynamic_seats(detection: dict) -> list:
    """从检测结果推断座位状态，输出结构与 judge_seats.seats 对齐。"""
    chairs = [
        {"box": d["box"], "conf": d["conf"], "kind": "chair"}
        for d in detection["chairs"]
    ]
    chairs += [
        {"box": d["box"], "conf": d["conf"], "kind": "stool"}
        for d in detection["stools"]
    ]
    # 座位实体去重（chair 与 stool 重叠时保留一个）
    seats = []
    for seat in sorted(chairs, key=lambda s: s["conf"], reverse=True):
        duplicate = False
        for keep in seats:
            smaller = min(box_area(seat["box"]), box_area(keep["box"]))
            if smaller > 0 and intersection(seat["box"], keep["box"]) / smaller > 0.6:
                duplicate = True
                break
        if not duplicate:
            seats.append(seat)
    seats.sort(key=lambda s: (box_center(s["box"])[1], box_center(s["box"])[0]))

    persons = detection["persons"]
    objects = detection["objects"]

    # 每个座位的落座区 / 桌面区
    for seat in seats:
        x1, y1, x2, y2 = seat["box"]
        h = max(1.0, y2 - y1)
        seat["sit_zone"] = [x1, y1 - SIT_ZONE_UP * h, x2, y2]
        seat["desk_zone"] = [x1 - 0.25 * h, y1 - DESK_ZONE_UP * h, x2 + 0.25 * h, y1 + 0.1 * h]
        seat["persons"] = []
        seat["objects"] = []

    # 人 → 座位：盖住椅子区超阈值 或 中心落在落座区（第一级：椅子=座位）
    matched = [False] * len(persons)
    for pidx, person in enumerate(persons):
        pbox = person["box"]
        pcenter = box_center(pbox)
        best, best_score = None, 0.0
        for seat in seats:
            cover = intersection(pbox, seat["sit_zone"]) / max(1.0, box_area(seat["sit_zone"]))
            center_in = (
                seat["sit_zone"][0] <= pcenter[0] <= seat["sit_zone"][2]
                and seat["sit_zone"][1] <= pcenter[1] <= seat["sit_zone"][3]
            )
            score = max(cover, 0.45 if center_in else 0.0)
            if score > best_score:
                best, best_score = seat, score
        if best is not None and best_score >= SIT_MIN_CHAIR_COVER:
            best["persons"].append(pbox)
            matched[pidx] = True

    # 第二级（人数优先）：没匹配到任何椅子的人也各计一个「隐式座位」——
    # 受拍摄角度限制椅子不可见时，能检测到人的位置就是座位。
    # 但与某把椅子的落座区已有明显接触（≥15%）却未达归属阈值的人不生成，
    # 避免「空椅 + 旁边人位」对同一位置双计。
    implicit = []
    for pidx, person in enumerate(persons):
        if matched[pidx]:
            continue
        pbox = person["box"]
        touches_chair = any(
            intersection(pbox, seat["sit_zone"]) / max(1.0, box_area(seat["sit_zone"])) >= 0.15
            for seat in seats
        )
        if touches_chair:
            continue
        pbox = [float(v) for v in pbox]
        implicit.append({"box": pbox, "conf": person["conf"], "kind": "implicit",
                         "persons": [pbox], "objects": [],
                         "sit_zone": pbox, "desk_zone": pbox})
    seats += implicit
    seats.sort(key=lambda s: (box_center(s["box"])[1], box_center(s["box"])[0]))

    # 物品 → 座位（仅当该座位没人时才作为「有物无人」证据，但归属先记下）
    for obj in objects:
        obox = obj["box"]
        oarea = box_area(obox)
        best, best_cover = None, 0.0
        for seat in seats:
            cover = intersection(obox, seat["desk_zone"]) / oarea if oarea > 0 else 0.0
            if cover > best_cover:
                best, best_cover = seat, cover
        if best is not None and best_cover >= DESK_MIN_COVER:
            best["objects"].append(obj)

    results = []
    for index, seat in enumerate(seats, start=1):
        kind = {"stool": "凳", "implicit": "人位"}.get(seat["kind"], "座")
        if seat["persons"]:
            state = STATE_OCCUPIED
        elif seat["objects"]:
            state = STATE_OBJECT_ONLY
        else:
            state = STATE_EMPTY
        results.append(
            {
                "id": f"D{index:02d}",
                "name": f"动态{kind}{index}",
                "state": state,
                "state_cn": STATE_LABELS_CN[state],
                "person_count": len(seat["persons"]),
                "object_count": len(seat["objects"]),
                "roi": [int(v) for v in seat["box"]],
                "kind": seat["kind"],
                "person_evidence": [{"box": b} for b in seat["persons"]],
                "evidence": [{"box": o["box"]} for o in seat["objects"]],
            }
        )
    return results


def total_person_count(detection: dict) -> int:
    """全图人数（人数优先：未分配到座位的人也计入）。"""
    return len(detection["persons"])


# ---------------------------------------------------------------- 绘制

def draw_dynamic(image, seat_results, detection):
    """标注图（整齐版）：座位框=紧致证据框，只补画未入座的人框。"""
    canvas = image.copy()

    seat_person_boxes = []
    for seat in seat_results:
        seat_person_boxes += [h["box"] for h in seat.get("person_evidence", [])]

    # 未被座位吸收的人框（红）：过道行人等
    for person in detection["persons"]:
        pbox = person["box"]
        absorbed = False
        for other in seat_person_boxes:
            inter = intersection(pbox, other)
            smaller = min(box_area(pbox), box_area(other))
            if smaller > 0 and inter / smaller > 0.5:
                absorbed = True
                break
        if absorbed:
            continue
        x1, y1, x2, y2 = [int(v) for v in pbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 220), 2)
        _label(canvas, "人(未入座)", x1, y1 - 6 if y1 > 24 else y2 + 18, (40, 40, 220))

    # 座位框：有人→人框并集；有物无人→物品框并集；空置→椅子框本身
    for seat in seat_results:
        color = STATE_COLORS[seat["state"]]
        x1, y1, x2, y2 = seat["roi"]
        if seat["state"] == STATE_OCCUPIED and seat.get("person_evidence"):
            box = _boxes_union([h["box"] for h in seat["person_evidence"]])
        elif seat["state"] == STATE_OBJECT_ONLY and seat.get("evidence"):
            box = _boxes_union([h["box"] for h in seat["evidence"]])
        else:
            box = seat["roi"]
        bx1, by1, bx2, by2 = [int(v) for v in box]
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
        text = f"{seat['id']} {seat['state_cn']}"
        _label(canvas, text, bx1, by1 - 6 if by1 > 24 else by2 + 18, color)
    return canvas


def _boxes_union(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def _label(canvas, text, x, y, color):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- 单图入口

def analyze_image(image, seat_model, coco_model, device="cpu"):
    """完整单图分析：检测 → 座位推断。返回 (detection, seat_results)。"""
    detection = detect_any_scene(image, seat_model, coco_model, device=device)
    seat_results = infer_dynamic_seats(detection)
    return detection, seat_results


def main():
    """命令行入口：python scripts/dynamic_seats.py --image <图片路径> [--out-dir runs/judge]"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="任意场景自习室图片动态座位识别")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--out-dir", default="runs/judge", help="输出目录")
    arguments = parser.parse_args()

    from ultralytics import YOLO

    image = cv2.imread(arguments.image)
    if image is None:
        print(f"[错误] 图片读取失败：{arguments.image}")
        return 1

    seat_model = YOLO("runs/train/seat_v6/weights/best.pt")
    coco_model = YOLO("models/yolov8n.pt")
    detection, seat_results = analyze_image(image, seat_model, coco_model)

    os.makedirs(arguments.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(arguments.image))[0]
    cv2.imwrite(os.path.join(arguments.out_dir, f"{stem}_dynamic.jpg"), draw_dynamic(image, seat_results, detection))
    summary = {
        "image": arguments.image,
        "mode": detection["mode"],
        "total_persons": len(detection["persons"]),
        "seats": seat_results,
    }
    with open(os.path.join(arguments.out_dir, f"{stem}_dynamic.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[模式] {detection['mode']}｜全图人数 {len(detection['persons'])}｜座位 {len(seat_results)} 个")
    for seat in seat_results:
        print(f"  {seat['id']:4s} {seat['state_cn']:6s} 人={seat['person_count']} 物={seat['object_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
