"""座位占用状态判定脚本：把 YOLO 检测结果映射到每个座位的四种状态。

四种状态：
    OCCUPIED    有人        —— 座位 ROI 内命中 person 框
    OBJECT_ONLY 有物无人    —— 无 person，但命中 book / bag / cup / item（占座的核心信号）
    EMPTY       空置        —— 两者都没有
    ABSENT      不在画面    —— 配置了 exists_check 的座位（如红脚凳），画面里检测不到
                            该座位实体时不计为座位，避免不同视角的照片产生幽灵座位

优先级约定（2026-11-06 调整）：
    1. 准确统计在座人数优先于座位归属的精确性——特殊拍摄角度导致椅子无法识别时，
       只要某个位置能检测到人，就判定该位置有人员；
    2. 人框归属采用「主体重叠 + 中心点兜底」双规则：主体落在座位内（ratio ≥ 阈值）
       为强命中；未强命中时若框中心点落在座位 ROI 内（允许 fallback_margin 像素的
       外扩容差）也计为命中，避免视角偏移造成漏计；
    3. 红脚凳等「实体凳」视为一个座位： stool 类检测命中其 ROI 即认为座位存在。

用法示例：
    # 单图判定，输出标注图与 JSON
    python scripts/judge_seats.py --image data/raw_real/real_03.jpg

    # 换权重与置信度
    python scripts/judge_seats.py --image xxx.jpg --weights runs/train/seat_v6/weights/best.pt --conf 0.3

判定逻辑要点：
    1. 只统计「框与座位 ROI 的交集面积 / 框自身面积」——即框有多大部分落在座位里，
       比 IoU 更适合大小悬殊的场景（人框很大、座位框很小）；
    2. 物品还需满足最小面积比例，避免远端噪点被当成占座证据；
    3. 独占分配：每个检测框只归属重叠比例最大的那一个座位。座位 ROI 含椅子区后
       相邻座位会相互接触，不互斥会导致一个人被多个座位同时计数；
    4. 全图在座人数（total_persons）对所有人框做跨座位去重后统计，
       用于核对「画面里到底有几个人」。
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
STATE_ABSENT = "ABSENT"

STATE_LABELS_CN = {
    STATE_OCCUPIED: "有人",
    STATE_OBJECT_ONLY: "有物无人",
    STATE_EMPTY: "空置",
    STATE_ABSENT: "不在画面",
}

# 绘制配色（BGR）：有人=红，有物无人=橙，空置=绿，不在画面=灰
STATE_COLORS = {
    STATE_OCCUPIED: (40, 40, 220),
    STATE_OBJECT_ONLY: (0, 140, 255),
    STATE_EMPTY: (60, 170, 60),
    STATE_ABSENT: (160, 160, 160),
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


def box_center(box: list) -> tuple:
    """返回框中心点 (cx, cy)。"""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def center_in_roi(center: tuple, roi: list, margin: float = 0.0) -> bool:
    """判断中心点是否落在 ROI 内；margin 为四边外扩容差（像素）。"""
    cx, cy = center
    return (roi[0] - margin <= cx <= roi[2] + margin) and (roi[1] - margin <= cy <= roi[3] + margin)


def center_margin_distance(center: tuple, roi: list) -> float:
    """中心点在 ROI 内时返回 0；否则返回到 ROI 的外扩距离（用于多座位竞争时取最近者）。"""
    cx, cy = center
    dx = max(roi[0] - cx, 0.0, cx - roi[2])
    dy = max(roi[1] - cy, 0.0, cy - roi[3])
    return max(dx, dy)


def stool_visible_by_color(image, roi: list, min_area_ratio: float = 0.04) -> bool:
    """HSV 红色色块兜底：判断 ROI 内是否出现红塑料凳实体。

    适用场景：训练集中 stool 样本太少、YOLO 检测不到时，用颜色特征做确定性判断。
    红塑料凳是高饱和度红色、近似方形的实心色块，与带靠背红椅（细长/被遮挡）区分度好。
    返回 True 表示 ROI 内存在凳子实体。
    """
    if image is None:
        return False

    x1, y1, x2, y2 = [int(v) for v in roi]
    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 红色在 HSV 色环两端各一段；饱和度和明度下限用于排除暗红木色
    mask = cv2.inRange(hsv, (0, 140, 90), (12, 255, 255)) | cv2.inRange(hsv, (170, 140, 90), (180, 255, 255))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    roi_area = (x2 - x1) * (y2 - y1)
    for index in range(1, count):
        _, _, w, h, area = stats[index]
        if area < min_area_ratio * roi_area:
            continue
        aspect = w / h if h > 0 else 0.0
        if 0.5 <= aspect <= 1.6:   # 凳子正视近似方形；细长条多为椅面边缘
            return True
    return False


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


def total_person_count(seat_results: list) -> int:
    """统计全图在座人数：各座位人数求和（座位内已去重、跨座位碎片已剔除）。

    注意：不同座位的两个人框可能互相重叠（邻座紧挨着坐），几何上无法与
    「同一人被切成两框」区分，因此不做跨座位几何去重——重复计数问题由
    上面的中心点兜底去重和边界溢出剔除两道工序保证。
    """
    return sum(item.get("person_count", 0) for item in seat_results)


def dedup_person_total(detections: list, contain: float = 0.5) -> int:
    """按人框直数全图人数（人数优先原则，固定座位表通道使用）。

    与 total_person_count 的区别：不经过座位 ROI 分配，站在过道、
    坐在未标定座位上的人也计入——只要画面里能检测到人就算一个人。
    同一人被切成多个碎片框时（交集/较小框 > contain）只计一次，
    以置信度最高的框为代表。
    """
    persons = sorted(
        (d for d in detections if d["class_id"] == 0),
        key=lambda d: d.get("conf", 0),
        reverse=True,
    )
    kept = []
    for det in persons:
        duplicate = False
        for keep in kept:
            overlap = intersection_area(det["box"], keep["box"])
            smaller = min(box_area(det["box"]), box_area(keep["box"]))
            if smaller > 0 and overlap / smaller > contain:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return len(kept)


def judge_seats(detections: list, seats: list, occupancy: dict, classes: dict = None, image=None) -> list:
    """逐座位判定状态。

    detections: [{"class_id": int, "conf": float, "box": [x1,y1,x2,y2]}]
    classes:    配置文件 classes 段（含 stool_id），用于识别「座位实体」检测；可为 None
    image:      原始图像（BGR）；exists_check 座位在 YOLO 检不到实体时用它做 HSV 兜底
    返回: [{"id", "name", "state", "state_cn", "person_count", "object_count",
            "person_evidence": [...], "evidence": [...]}]

    座位区域模型（与「座位定义规范」一致）：
        roi          椅子区 —— person 框按它判定；
        object_roi   可选，物品区（椅前桌面）—— 配置了它时物品框只按它判定；
        exists_check 可选，"stool" —— ROI 内检测不到 stool 实体时判 ABSENT（不计为座位）。

    独占分配：先把每个检测框分给重叠比例最大的座位，再按座位聚合。
    """
    person_threshold = occupancy["person_iou_threshold"]
    object_threshold = occupancy["object_iou_threshold"]
    min_object_area_ratio = occupancy["min_object_area_ratio"]

    # 人数优先相关参数：中心点兜底开关与外扩容差
    center_fallback = occupancy.get("person_center_fallback", True)
    fallback_margin = float(occupancy.get("person_center_fallback_margin", 60))
    stool_exist_ratio = float(occupancy.get("stool_exist_ratio", 0.30))
    stool_id = (classes or {}).get("stool_id")

    # 第零步：座位实体存在性检查——配置了 exists_check 的座位，画面里没有实体就判 ABSENT。
    # 优先用 YOLO 的 stool 检测；检不到时用 HSV 红色色块兜底（红塑料凳颜色特征明确）。
    stool_boxes = [d["box"] for d in detections if stool_id is not None and d["class_id"] == stool_id]
    seat_present = []
    for seat in seats:
        if seat.get("exists_check") == "stool":
            found = any(inside_ratio(box, seat["roi"]) >= stool_exist_ratio for box in stool_boxes)
            if not found:
                found = stool_visible_by_color(image, seat["roi"])
            seat_present.append(found)
        else:
            seat_present.append(True)

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
            # 人（强命中）：框的主体落在座位内即算坐在这里
            if ratio >= person_threshold:
                hits_by_seat[seat_index]["person"].append(
                    {
                        "ratio": round(ratio, 3),
                        "conf": round(detection["conf"], 3),
                        "box": list(detection["box"]),
                        "match": "body",
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
                        "box": list(detection["box"]),
                    }
                )

    # 第三步：人数优先的中心点兜底——主体重叠不足、但框中心落在某座位 ROI（含外扩）
    # 内的人框，也计为该座位有人；前提是没有其他座位对它构成强命中（避免邻座误吸）。
    if center_fallback:
        for detection, (seat_index, ratio) in zip(detections, assigned):
            if detection["class_id"] != 0 or ratio >= person_threshold:
                continue  # 已强命中或不是人框，跳过

            center = box_center(detection["box"])
            candidates = []
            for seat_index_other, seat in enumerate(seats):
                if not seat_present[seat_index_other]:
                    continue  # 座位不在画面中，不参与兜底
                if center_in_roi(center, seat["roi"], fallback_margin):
                    distance = center_margin_distance(center, seat["roi"])
                    candidates.append((distance, seat_index_other))

            if not candidates:
                continue

            # 跨座位去重：若该框与任何座位已命中的人框高度重叠（同人碎片），不再兜底计数，
            # 否则同一个人会被「主框 + 碎片框」计成两次
            duplicate_of_existing = False
            for other_index, hits in hits_by_seat.items():
                for existing in hits["person"]:
                    overlap = intersection_area(detection["box"], existing["box"])
                    smaller = min(box_area(detection["box"]), box_area(existing["box"]))
                    if smaller > 0 and overlap / smaller > 0.5:
                        duplicate_of_existing = True
                        break
                if duplicate_of_existing:
                    break
            if duplicate_of_existing:
                continue

            # 取外扩距离最小（最贴近）的座位
            candidates.sort()
            _, fallback_index = candidates[0]
            hits_by_seat[fallback_index]["person"].append(
                {
                    "ratio": round(ratio, 3),
                    "conf": round(detection["conf"], 3),
                    "box": list(detection["box"]),
                    "match": "center",
                }
            )

    # 第三步半：跨座位边界溢出剔除——人框在本座位重叠率低（< person_spill_own_ratio）
    # 且明显压进一个「已有人命中」的邻座 ROI（覆盖邻座面积 ≥ person_spill_neighbor_ratio）
    # 时，视为座位 ROI 边界切割人框产生的碎片，予以剔除，避免同一人被相邻座位重复计数。
    spill_own = float(occupancy.get("person_spill_own_ratio", 0.75))
    spill_neighbor = float(occupancy.get("person_spill_neighbor_ratio", 0.20))
    for seat_index, seat in enumerate(seats):
        if not seat_present[seat_index]:
            continue
        kept_hits = []
        for hit in hits_by_seat[seat_index]["person"]:
            own_ratio = hit["ratio"] if hit["match"] == "body" else inside_ratio(hit["box"], seat["roi"])
            if own_ratio >= spill_own:
                kept_hits.append(hit)
                continue
            spilled_into_occupied_neighbor = False
            for other_index, other_seat in enumerate(seats):
                if other_index == seat_index or not seat_present[other_index]:
                    continue
                if not hits_by_seat[other_index]["person"]:
                    continue  # 邻座本身没人，谈不上「溢入已占座位」
                # 用「覆盖邻座 ROI 面积的比例」衡量溢出程度：小框贴边不算溢出
                coverage = intersection_area(hit["box"], other_seat["roi"]) / box_area(other_seat["roi"])
                if box_area(other_seat["roi"]) > 0 and coverage >= spill_neighbor:
                    # 还需与邻座已命中的人框实际重叠（同一人被边界切开才会框挨框）；
                    # 只是压到邻座 ROI 空白角落的不算
                    for other_hit in hits_by_seat[other_index]["person"]:
                        overlap = intersection_area(hit["box"], other_hit["box"])
                        smaller = min(box_area(hit["box"]), box_area(other_hit["box"]))
                        if smaller > 0 and overlap / smaller >= spill_neighbor:
                            spilled_into_occupied_neighbor = True
                            break
                if spilled_into_occupied_neighbor:
                    break
            if not spilled_into_occupied_neighbor:
                kept_hits.append(hit)
        hits_by_seat[seat_index]["person"] = kept_hits

    results = []
    for seat_index, seat in enumerate(seats):
        person_hits = dedup_person_hits(hits_by_seat[seat_index]["person"])
        object_hits = hits_by_seat[seat_index]["object"]

        # exists_check 座位（如红脚凳）没有配套桌面，不适用「物品占座」逻辑；
        # 凳子旁的保温杯等杂物不应把空凳判成「有物无人」。
        if seat.get("exists_check"):
            object_hits = []

        # 不在画面中的座位直接判 ABSENT，不参与三态判定
        if not seat_present[seat_index]:
            state = STATE_ABSENT
        elif person_hits:
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
                "person_count": len(person_hits) if state != STATE_ABSENT else 0,
                "object_count": len(object_hits) if state != STATE_ABSENT else 0,
                "person_evidence": person_hits,
                "evidence": object_hits,
            }
        )

    return results


def _boxes_union(boxes: list) -> list:
    """一组 [x1,y1,x2,y2] 框的并集外包框。"""
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _clip_box(box: list, bounds: list) -> list:
    """把框裁剪到 bounds 内（防止证据框溢出座位区域）。"""
    return [
        max(box[0], bounds[0]), max(box[1], bounds[1]),
        min(box[2], bounds[2]), min(box[3], bounds[3]),
    ]


def _label(canvas, text: str, x: int, y: int, color):
    """带黑描边的状态标签，任意底色上都清晰。"""
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def draw_result(image, seats: list, results: list, detections: list, classes: dict = None):
    """标注图（整齐版）：只画「座位框 + 未归属人框」两层，不再叠加全部原始检测框。

    座位框按状态取紧致证据框：
      有人   → 座位内人框并集（裁剪到 ROI）——框贴合真实坐着的人；
      有物无人 → 座位内物品框并集（裁剪到 ROI）——框贴合占座物品；
      空置   → 座位 ROI 本身（本场景空座 ROI 已按椅子紧致标定）；
      不在画面 → 不画框，只留一个小标签。
    未被任何座位吸收的人框（过道行人等）画红框单独标出，保证「人数优先」可见。
    """
    canvas = image.copy()
    state_by_id = {item["id"]: item for item in results}

    seat_person_boxes = []
    seat_object_boxes = []
    for item in results:
        seat_person_boxes += [hit["box"] for hit in item.get("person_evidence", [])]
        seat_object_boxes += [hit["box"] for hit in item.get("evidence", [])]

    def _absorbed(box, pool):
        for other in pool:
            inter = intersection_area(box, other)
            smaller = min(box_area(box), box_area(other))
            if smaller > 0 and inter / smaller > 0.5:
                return True
        return False

    # 第一层：未被座位吸收的人框（红，过道/未标定位置的人）
    for detection in detections:
        if detection["class_id"] != 0:
            continue
        if _absorbed(detection["box"], seat_person_boxes):
            continue
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 40, 220), 2)
        _label(canvas, "人(未入座)", x1, y1 - 6 if y1 > 24 else y2 + 18, (40, 40, 220))

    # 第二层：座位框（证据驱动紧致框 + 状态标签）
    for seat in seats:
        info = state_by_id[seat["id"]]
        color = STATE_COLORS[info["state"]]
        x1, y1, x2, y2 = seat["roi"]

        if info["state"] == STATE_ABSENT:
            # 不在画面：细灰虚位框 + 小标签，弱化视觉
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (160, 160, 160), 1)
            _label(canvas, f"{seat['id']} 不在画面", x1, y1 - 6 if y1 > 24 else y2 + 18, (160, 160, 160))
            continue

        if info["state"] == STATE_OCCUPIED and info["person_evidence"]:
            box = _clip_box(_boxes_union([h["box"] for h in info["person_evidence"]]), seat["roi"])
        elif info["state"] == STATE_OBJECT_ONLY and info["evidence"]:
            box = _clip_box(_boxes_union([h["box"] for h in info["evidence"]]), seat["roi"])
        else:
            box = seat["roi"]

        bx1, by1, bx2, by2 = [int(v) for v in box]
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
        text = f"{seat['id']} {info['state_cn']}"
        _label(canvas, text, bx1, by1 - 6 if by1 > 24 else by2 + 18, color)

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

    seat_results = judge_seats(detections, seats, config["occupancy"], config.get("classes"), image=result.orig_img)
    annotated = draw_result(result.orig_img, seats, seat_results, detections, config.get("classes"))

    os.makedirs(os.path.dirname(arguments.out_image) or ".", exist_ok=True)
    cv2.imwrite(arguments.out_image, annotated)

    os.makedirs(os.path.dirname(arguments.out_json) or ".", exist_ok=True)
    persons = total_person_count(seat_results)
    summary = {
        "image": arguments.image,
        "weights": weights,
        "totals": {
            "seats": len(seat_results),
            "counted_seats": sum(1 for item in seat_results if item["state"] != STATE_ABSENT),
            "total_persons": persons,
            STATE_OCCUPIED: sum(1 for item in seat_results if item["state"] == STATE_OCCUPIED),
            STATE_OBJECT_ONLY: sum(1 for item in seat_results if item["state"] == STATE_OBJECT_ONLY),
            STATE_EMPTY: sum(1 for item in seat_results if item["state"] == STATE_EMPTY),
            STATE_ABSENT: sum(1 for item in seat_results if item["state"] == STATE_ABSENT),
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
    print(f"\n[汇总] 在座人数 {persons}｜有人 {totals[STATE_OCCUPIED]} / 有物无人 {totals[STATE_OBJECT_ONLY]}"
          f" / 空置 {totals[STATE_EMPTY]} / 不在画面 {totals[STATE_ABSENT]}")
    print(f"[输出] {arguments.out_json}")
    print(f"[输出] {arguments.out_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
