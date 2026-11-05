"""生成演示素材的座位 ROI（config/rois.json）—— 按「椅子 + 椅前桌面」模型。

座位定义（v2，与用户确认的口径一致）：
    1. 一个座位 = 一把椅子 + 椅子正前方一定范围的桌面；
    2. 一张大桌子旁有 N 把椅子，就有 N 个座位（座位数 = 贴桌椅子数）；
    3. 座位 ROI = 椅子所在区域 ∪ 该椅子正前方的桌面条带，判定模块在 ROI 内
       统计 person / 物品框。

几何模型：
    桌子按「远边 top / 近边 bottom / 左端 left / 右端 right」四个边描述，
    每把椅子登记为 (边, 沿边位置)。椅子框与桌面条带的尺寸按桌子尺寸的
    比例伸缩（近边椅子在画面里更大，比例取更大值），并用预览图人工核对。

坐标来源：
    在叠加原图坐标网格的放大图（data/video_demo/desk_*_grid.jpg）上逐桌读出，
    为人工测量值，允许 ±10px 误差；预览渲染后可再微调。

用法：
    python scripts/make_web_rois.py                 # 生成 rois.json + 预览图
    python scripts/make_web_rois.py --sync-config   # 同时把 seats 覆盖到 config.yaml
"""

import argparse
import json
import os

import cv2

SOURCE_IMAGE = "data/video_demo/main_view.jpg"
IMAGE_SIZE = [1920, 1410]
OUTPUT_PATH = "config/rois.json"
PREVIEW_PATH = "data/video_demo/roi_preview.jpg"

# ---------------------------------------------------------------------------
# 桌面矩形与椅子清单（在 desk_*_grid.jpg 网格图上人工读出）
# chairs 每项: (边, 沿边坐标)
#   top/bottom 边 → 该值为椅子中心的 x 像素
#   left/right 端 → 该值为椅子中心的 y 像素
# ---------------------------------------------------------------------------
DESKS = {
    "A": {
        "name": "A 桌（上中左，全空）",
        "rect": (548, 512, 800, 648),
        "chairs": [
            ("top", 570),
            ("top", 632),      # 远边 2 把，第 2 把有人（白衣）
            ("bottom", 620),
            ("bottom", 708),
            ("bottom", 786),   # 近边 3 把（坐标在 A_near_zoom.jpg 网格图上核准）
            ("left", 505),     # 左端 1 把
        ],
    },
    "B": {
        "name": "B 桌（中右，有人）",
        "rect": (1132, 735, 1398, 908),
        "chairs": [
            ("top", 1180),     # 远边左，有人（白衣）
            ("top", 1340),     # 远边右，空椅
            ("right", 820),    # 右端，有人（长发）
            ("bottom", 1258),  # 近边 1 把空椅
        ],
    },
    "C": {
        "name": "C 桌（中下，满桌白纸占座）",
        "rect": (768, 880, 1028, 1040),
        "chairs": [
            ("top", 855),
            ("top", 940),
            ("top", 1005),     # 远边 3 把
            ("right", 935),    # 右端 1 把
            ("bottom", 890),
            ("bottom", 985),   # 近边 2 把
        ],
    },
}

# 各边几何参数（相对桌子尺寸的比例）
# band  = 椅子区向桌外伸展的深度；strip = 桌面条带向桌内伸展的深度
# half  = 椅子框在沿边方向上的半宽
SIDE_PARAMS = {
    "top":    {"band": 0.42, "strip": 0.45, "half": 0.13},   # 相对桌高 band/strip，相对桌宽 half
    "bottom": {"band": 0.42, "strip": 0.50, "half": 0.13},   # band 压到椅子实际高度，防止桌后（画面近处）的人被算进来
    "left":   {"band": 0.38, "strip": 0.50, "half": 0.22},   # 相对桌宽 band/strip，相对桌高 half
    "right":  {"band": 0.38, "strip": 0.50, "half": 0.22},
}

# 编号顺序：远边（左→右）→ 右端（上→下）→ 近边（左→右）→ 左端（上→下）
SIDE_ORDER = ["top", "right", "bottom", "left"]


def clamp(value: float, low: int, high: int) -> int:
    """把坐标限制在画面范围内。"""
    return int(max(low, min(high, value)))


def build_seat_roi(desk_rect: tuple, side: str, position: int) -> dict:
    """按边与沿边位置生成一个座位的椅子框、桌面条带与合并 ROI。"""
    x1, y1, x2, y2 = desk_rect
    table_w = x2 - x1
    table_h = y2 - y1
    params = SIDE_PARAMS[side]

    if side in ("top", "bottom"):
        half_w = params["half"] * table_w
        chair_cx = position
        if side == "top":
            band_y1 = y1 - params["band"] * table_h
            band_y2 = y1
            strip_y1 = y1
            strip_y2 = y1 + params["strip"] * table_h
        else:
            band_y1 = y2
            band_y2 = y2 + params["band"] * table_h
            strip_y1 = y2 - params["strip"] * table_h
            strip_y2 = y2
        chair = [chair_cx - half_w, band_y1, chair_cx + half_w, band_y2]
        desk = [chair_cx - half_w, strip_y1, chair_cx + half_w, strip_y2]
    else:
        half_h = params["half"] * table_h
        chair_cy = position
        if side == "left":
            band_x1 = x1 - params["band"] * table_w
            band_x2 = x1
            strip_x1 = x1
            strip_x2 = x1 + params["strip"] * table_w
        else:
            band_x1 = x2
            band_x2 = x2 + params["band"] * table_w
            strip_x1 = x2 - params["strip"] * table_w
            strip_x2 = x2
        chair = [band_x1, chair_cy - half_h, band_x2, chair_cy + half_h]
        desk = [strip_x1, chair_cy - half_h, strip_x2, chair_cy + half_h]

    # 合并 ROI = 椅子框 ∪ 桌面条带（两框同宽，直接取外接矩形）
    roi = [
        clamp(min(chair[0], desk[0]), 0, IMAGE_SIZE[0]),
        clamp(min(chair[1], desk[1]), 0, IMAGE_SIZE[1]),
        clamp(max(chair[2], desk[2]), 0, IMAGE_SIZE[0]),
        clamp(max(chair[3], desk[3]), 0, IMAGE_SIZE[1]),
    ]
    return {
        "chair": [round(v) for v in chair],
        "desk": [round(v) for v in desk],
        "roi": roi,
    }


def build_all_seats() -> list:
    """逐桌生成座位列表，编号按 远边→右端→近边→左端 顺序。"""
    seats = []
    for letter, desk in DESKS.items():
        grouped = {side: [] for side in SIDE_ORDER}
        for side, position in desk["chairs"]:
            grouped[side].append(position)
        for side in SIDE_ORDER:
            grouped[side].sort()

        index = 0
        for side in SIDE_ORDER:
            for position in grouped[side]:
                index += 1
                seat_id = f"{letter}{index:02d}"
                geometry = build_seat_roi(desk["rect"], side, position)
                seats.append(
                    {
                        "id": seat_id,
                        "name": f"{letter}桌{index}号",
                        "side": side,
                        **geometry,
                    }
                )
    return seats


def render_preview(seats: list) -> None:
    """把座位框画到底图上，人工核对位置。"""
    image = cv2.imread(SOURCE_IMAGE)
    if image is None:
        print(f"[警告] 底图读取失败，跳过预览：{SOURCE_IMAGE}")
        return

    for seat in seats:
        cx1, cy1, cx2, cy2 = seat["chair"]
        dx1, dy1, dx2, dy2 = seat["desk"]
        rx1, ry1, rx2, ry2 = seat["roi"]

        # 桌面条带（蓝）与椅子区（黄）画半透明底层，合并 ROI 画绿色描边
        overlay = image.copy()
        cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (255, 180, 60), -1)
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (60, 200, 255), -1)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (80, 220, 80), 2)
        cv2.putText(
            image, seat["id"], (rx1 + 3, ry1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4,
        )
        cv2.putText(
            image, seat["id"], (rx1 + 3, ry1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 255, 120), 2,
        )

    os.makedirs(os.path.dirname(PREVIEW_PATH), exist_ok=True)
    cv2.imwrite(PREVIEW_PATH, image)
    print(f"[完成] 预览图：{PREVIEW_PATH}")


def sync_config(seats: list) -> None:
    """把 seats 列表覆盖到 config.yaml 的 seats 段（保留其余段落与注释）。"""
    config_path = "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        lines = config_file.readlines()

    start = end = None
    for index, line in enumerate(lines):
        if line.startswith("seats:"):
            start = index
            continue
        if start is not None and line and not line[0].isspace() and not line.startswith("#"):
            end = index
            break
    if start is None:
        print("[错误] config.yaml 中找不到 seats: 段")
        return
    if end is None:
        end = len(lines)

    block = ["# 由 scripts/make_web_rois.py 自动生成（椅子 + 椅前桌面模型），勿手工编辑\n"]
    block.append("seats:\n")
    for seat in seats:
        block.append(f"  - id: {seat['id']}\n")
        block.append(f"    name: {seat['name']}\n")
        block.append(f"    roi: {seat['roi']}\n")

    lines[start:end] = block
    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.writelines(lines)
    print(f"[完成] config.yaml seats 段已更新为 {len(seats)} 座")


def main() -> None:
    """生成 rois.json、渲染预览，可选同步 config.yaml。"""
    parser = argparse.ArgumentParser(description="按椅子模型生成座位 ROI")
    parser.add_argument("--sync-config", action="store_true", help="同时覆盖 config.yaml 的 seats 段")
    arguments = parser.parse_args()

    seats = build_all_seats()

    output_data = {
        "seat_model": "chair+desk",
        "definition": "一个座位 = 一把椅子 + 椅子正前方的桌面条带；座位数 = 贴桌椅子数",
        "source_image": SOURCE_IMAGE,
        "image_size": IMAGE_SIZE,
        "note": "演示素材 ROI；真实拍摄素材到位后需按同一模型重新标定",
        "seats": seats,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=2)

    counts = {}
    for seat in seats:
        letter = seat["id"][0]
        counts[letter] = counts.get(letter, 0) + 1
    print(f"[完成] 已写入 {len(seats)} 个座位：{OUTPUT_PATH}")
    for letter, number in counts.items():
        print(f"  {letter} 桌 {number} 座")
    for seat in seats:
        print(f"  {seat['id']} ({seat['side']:6s}) roi={seat['roi']}")

    render_preview(seats)
    if arguments.sync_config:
        sync_config(seats)


if __name__ == "__main__":
    main()
