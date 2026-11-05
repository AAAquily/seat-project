"""座位占用状态判定脚本：把 YOLO 检测结果映射到每个座位的三种状态。

三种状态：
    OCCUPIED    有人        —— 座位 ROI 内命中 person 框
    OBJECT_ONLY 有物无人    —— 无 person，但命中 book / bag / cup / item（占座的核心信号）
    EMPTY       空置        —— 两者都没有

用法示例：
    # 单图判定，输出标注图与 JSON
    python scripts/judge_seats.py --image data/video_demo/main_view.jpg

    # 换权重与置信度
    python scripts/judge_seats.py --image xxx.jpg --weights runs/train/seat_v1/weights/best.pt --conf 0.3

判定逻辑要点：
    1. 只统计「框与座位 ROI 的交集面积 / 框自身面积」——即框有多大部分落在座位里，
       比 IoU 更适合大小悬殊的场景（人框很大、座位框很小）；
    2. 物品还需满足最小面积比例，避免远端噪点被当成占座证据；
    3. 独占分配：每个检测框只归属重叠比例最大的那一个座位。座位 ROI 含椅子区后
       相邻座位会相互接触，不互斥会导致一个人被多个座位同时计数。
"""

import argparse
import json
import os

import cv2
import yaml
from ultralytics import YOLO

# 状态常量，供下游模块（时序跟踪、网页看板）统一引用
STATE_OCCUPIED = "OCCUPIED"
STATE_OBJECT_ONLY = "OBJECT_ONLY"
STATE_EMPTY = "EMPTY"

STATE_LABELS_CN = {
    STATE_OCCUPIED: "有人",
    STATE_OBJECT_ONLY: "有物无人",
    STATE_EMPTY: "空置",
}

# 绘制配色（BGR）：有人=红，有物无人=橙，空置=绿
STATE_COLORS = {
    STATE_OCCUPIED: (40, 40, 220),
    STATE_OBJECT_ONLY: (0, 140, 255),
    STATE_EMPTY: (60, 170, 60),
}


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="按座位 ROI 判定占用状态")

    parser.add_argument("--image", required=True, help="待判定的图片路径")
    parser.add_argument("--config", default="config/config.yaml", help="主配置路径")
    parser.add_argument("--weights", default="", help="权重路径，默认取配置文件中的 model.weights")
    parser.add_argument("--conf", type=float, default=-1, help="置信度阈值，默认取配置文件")
    parser.add_argument("--imgsz", type=int, default=0, help="推理尺寸，默认取配置文件")
    parser.add_argument("--out-json", default="runs/judge/seat_states.json", help="判定结果 JSON 输出路径")
    parser.add_argument("--out-image", default="runs/judge/seat_states.jpg", help="标注图输出路径")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """读取主配置。"""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_weights(config: dict, override: str) -> str:
    """确定权重路径：命令行优先；配置里的路径不存在时回退到预训练模型并提示。"""
    candidate = override or config["model"]["weights"]

    if os.path.isfile(candidate):
        return candidate

    fallback = os.path.join("models", "yolov8n.pt")
    print(f"[提示] 权重不存在：{candidate}，回退到 {fallback}")
    return fallback


def box_area(box: list) -> float:
    """计算矩形面积，box 为 [x1, y1, x2, y2]。"""
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height


def intersection_area(box_a: list, box_b: list) -> float:
    """计算两个矩形的交集面积。"""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    if right <= left or bottom <= top:
        return 0.0
    return (right - left) * (bottom - top)


def inside_ratio(detection_box: list, seat_box: list) -> float:
    """检测框有多少比例落在座位框内；返回 0～1。"""
    detection_area = box_area(detection_box)
    if detection_area <= 0:
        return 0.0
    return intersection_area(detection_box, seat_box) / detection_area


def dedup_person_hits(hits: list) -> list:
    """同一座位内的人物去重：检测器偶尔对同一人输出两个框（NMS 漏掉）。

    规则：按置信度从高到低保留；若候选框与已保留框的「交集面积 / 较小框面积」
    超过 0.5，视为同一个人，丢弃。
    """
    ordered = sorted(hits, key=lambda hit: hit["conf"], reverse=True)
    kept = []
    for hit in ordered:
        duplicate = False
        for existing in kept:
            overlap = intersection_area(hit["box"], existing["box"])
            smaller = min(box_area(hit["box"]), box_area(existing["box"]))
            if smaller > 0 and overlap / smaller > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(hit)
    return kept


def judge_seats(detections: list, seats: list, occupancy: dict) -> list:
    """逐座位判定状态。

    detections: [{"class_id": int, "conf": float, "box": [x1,y1,x2,y2]}]
    返回: [{"id", "name", "state", "person_count", "object_count", "evidence": [...]}]

    座位区域模型（与「座位定义规范」一致）：
        roi         椅子区 —— person 框按它判定；
        object_roi  可选，物品区（椅前桌面）—— 配置了它时物品框只按它判定，
                    用于把「桌面长期固定物」排除在占座信号之外。

    独占分配：先把每个检测框分给重叠比例最大的座位，再按座位聚合。
    """
    person_threshold = occupancy["person_iou_threshold"]
    object_threshold = occupancy["object_iou_threshold"]
    min_object_area_ratio = occupancy["min_object_area_ratio"]

    # 第一步：每个检测框找到归属座位（重叠比例最大者）。
    # 人只看椅子区；物品配置了物品区就只看物品区，否则退回椅子区。
    assigned = []  # [(seat_index, ratio), ...]，与 detections 一一对应
    for detection in detections:
        is_person = detection["class_id"] == 0
        best_index = -1
        best_ratio = 0.0
        for seat_index, seat in enumerate(seats):
            if is_person or not seat.get("object_roi"):
                ratio = inside_ratio(detection["box"], seat["roi"])
            else:
                ratio = inside_ratio(detection["box"], seat["object_roi"])
            if ratio > best_ratio:
                best_index = seat_index
                best_ratio = ratio
        assigned.append((best_index, best_ratio))

    # 第二步：按座位聚合命中结果
    hits_by_seat = {index: {"person": [], "object": []} for index in range(len(seats))}
    for detection, (seat_index, ratio) in zip(detections, assigned):
        if seat_index < 0 or ratio <= 0:
            continue

        if detection["class_id"] == 0:
            # 人：只要框的主体落在座位内即算坐在这里
            if ratio >= person_threshold:
                hits_by_seat[seat_index]["person"].append(
                    {
                        "ratio": round(ratio, 3),
                        "conf": round(detection["conf"], 3),
                        "box": list(detection["box"]),
                    }
                )
        else:
            # 物品：既要落在座位内，又要有足够面积，避免远端噪点误判
            seat = seats[seat_index]
            seat_box = seat["object_roi"] if seat.get("object_roi") else seat["roi"]
            seat_area = box_area(seat_box)
            object_area_ratio = box_area(detection["box"]) / seat_area if seat_area > 0 else 0.0
            if ratio >= object_threshold and object_area_ratio >= min_object_area_ratio:
                hits_by_seat[seat_index]["object"].append(
                    {
                        "class_id": detection["class_id"],
                        "ratio": round(ratio, 3),
                        "conf": round(detection["conf"], 3),
                    }
                )

    results = []
    for seat_index, seat in enumerate(seats):
        person_hits = dedup_person_hits(hits_by_seat[seat_index]["person"])
        object_hits = hits_by_seat[seat_index]["object"]

        if person_hits:
            state = STATE_OCCUPIED
        elif object_hits:
            state = STATE_OBJECT_ONLY
        else:
            state = STATE_EMPTY

        results.append(
            {
                "id": seat["id"],
                "name": seat.get("name", seat["id"]),
                "state": state,
                "state_cn": STATE_LABELS_CN[state],
                "person_count": len(person_hits),
                "object_count": len(object_hits),
                "evidence": object_hits,
            }
        )

    return results


def draw_result(image, seats: list, results: list, detections: list):
    """把座位状态与检测框画到图上，便于人工核对。"""
    canvas = image.copy()

    # 先画检测框：人=深红，物品=灰色，避免盖住座位标注
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = (40, 40, 220) if detection["class_id"] == 0 else (140, 140, 140)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

    state_by_id = {item["id"]: item for item in results}

    for seat in seats:
        x1, y1, x2, y2 = seat["roi"]
        info = state_by_id[seat["id"]]
        color = STATE_COLORS[info["state"]]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # 物品区用虚线细框画出，便于区分「判人的椅子区」和「判物的桌面区」
        if seat.get("object_roi"):
            ox1, oy1, ox2, oy2 = seat["object_roi"]
            for offset in range(0, max(ox2 - ox1, oy2 - oy1), 14):
                if offset < ox2 - ox1:
                    cv2.line(canvas, (ox1 + offset, oy1), (min(ox1 + offset + 7, ox2), oy1), color, 1)
                    cv2.line(canvas, (ox1 + offset, oy2), (min(ox1 + offset + 7, ox2), oy2), color, 1)
                if offset < oy2 - oy1:
                    cv2.line(canvas, (ox1, oy1 + offset), (ox1, min(oy1 + offset + 7, oy2)), color, 1)
                    cv2.line(canvas, (ox2, oy1 + offset), (ox2, min(oy1 + offset + 7, oy2)), color, 1)

        text = f"{seat['id']} {info['state_cn']}"
        text_y = y1 - 6 if y1 > 20 else y2 + 16
        cv2.putText(canvas, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return canvas


def main() -> int:
    """主流程：检测 → 判定 → 输出。"""
    arguments = parse_arguments()

    if not os.path.isfile(arguments.image):
        print(f"[错误] 图片不存在：{arguments.image}")
        return 1

    config = load_config(arguments.config)
    seats = config.get("seats") or []
    if not seats:
        print("[错误] 配置中 seats 为空，请先用 roi_tool.py 标定座位")
        return 1

    weights = resolve_weights(config, arguments.weights)
    conf = arguments.conf if arguments.conf >= 0 else config["model"]["conf"]
    imgsz = arguments.imgsz if arguments.imgsz > 0 else config["model"]["imgsz"]

    print(f"[推理] 权重={weights} conf={conf} imgsz={imgsz} 座位数={len(seats)}")

    model = YOLO(weights)
    prediction = model.predict(source=arguments.image, conf=conf, imgsz=imgsz, device=config["model"]["device"], verbose=False)
    result = prediction[0]

    detections = []
    for box in result.boxes:
        coordinates = box.xyxy[0].tolist()
        detections.append(
            {
                "class_id": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": [int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])],
            }
        )

    print(f"[检测] 共 {len(detections)} 个目标")

    seat_results = judge_seats(detections, seats, config["occupancy"])
    annotated = draw_result(result.orig_img, seats, seat_results, detections)

    os.makedirs(os.path.dirname(arguments.out_image) or ".", exist_ok=True)
    cv2.imwrite(arguments.out_image, annotated)

    os.makedirs(os.path.dirname(arguments.out_json) or ".", exist_ok=True)
    summary = {
        "image": arguments.image,
        "weights": weights,
        "totals": {
            "seats": len(seat_results),
            STATE_OCCUPIED: sum(1 for item in seat_results if item["state"] == STATE_OCCUPIED),
            STATE_OBJECT_ONLY: sum(1 for item in seat_results if item["state"] == STATE_OBJECT_ONLY),
            STATE_EMPTY: sum(1 for item in seat_results if item["state"] == STATE_EMPTY),
        },
        "seats": seat_results,
    }
    with open(arguments.out_json, "w", encoding="utf-8") as json_file:
        json.dump(summary, json_file, ensure_ascii=False, indent=2)

    # 控制台输出：逐座位状态一览
    print("\n座位状态：")
    for item in seat_results:
        print(f"  {item['id']:4s} {item['state_cn']:6s} 人={item['person_count']} 物={item['object_count']}")

    totals = summary["totals"]
    print(f"\n[汇总] 有人 {totals[STATE_OCCUPIED]} / 有物无人 {totals[STATE_OBJECT_ONLY]} / 空置 {totals[STATE_EMPTY]}")
    print(f"[输出] {arguments.out_json}")
    print(f"[输出] {arguments.out_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
