"""生成网络演示素材的座位 ROI 初稿（config/rois.json）。

说明：
    1. 素材来源 —— data/raw 中 20261104_0900_web_f0010.jpg（图书馆中庭高机位俯拍图）；
    2. 试点范围 —— 中央 3 张独立方桌，每桌 4 座，共 12 个座位，与编号表 12 座试点对应；
    3. 坐标来源 —— 桌面外接矩形由人工在叠加网格的预览图上读出（1920 × 1410 像素系）；
    4. 后续维护 —— 真实拍摄素材到位后，用 roi_tool.py 重新标定即可，本文件仅作演示。
"""

import json
import os

# 底图路径与尺寸
SOURCE_IMAGE = "data/video_demo/main_view.jpg"
IMAGE_SIZE = [1920, 1410]

# 输出路径
OUTPUT_PATH = "config/rois.json"

# 三张试点桌的桌面外接矩形 (x1, y1, x2, y2)，按 A / B / C 桌命名
DESK_RECTS = {
    "A": (522, 486, 811, 643),    # 上中左桌：4 座全空
    "B": (1122, 744, 1438, 962),  # 中右桌：3 人围坐并有物品
    "C": (801, 823, 1127, 1052),  # 中下桌：满桌白纸无人，典型占座
}

# 座位编号顺序：1 = 远左，2 = 远右，3 = 近左，4 = 近右
SEAT_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]


def split_desk_into_seats(desk_rect: tuple) -> list:
    """把一张桌子矩形按四象限切成 4 个座位 ROI。"""
    x1, y1, x2, y2 = desk_rect
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    seat_rois = []
    for col_side, row_side in SEAT_ORDER:
        roi_x1 = x1 if col_side == 0 else center_x
        roi_x2 = center_x if col_side == 0 else x2
        roi_y1 = y1 if row_side == 0 else center_y
        roi_y2 = center_y if row_side == 0 else y2
        seat_rois.append([roi_x1, roi_y1, roi_x2, roi_y2])
    return seat_rois


def main() -> None:
    """组装 12 个座位框并写入 rois.json。"""
    seats = []

    for desk_letter, desk_rect in DESK_RECTS.items():
        seat_rois = split_desk_into_seats(desk_rect)
        for seat_index, roi in enumerate(seat_rois, start=1):
            seat_id = f"{desk_letter}{seat_index:02d}"
            seats.append({"id": seat_id, "name": f"{desk_letter}桌{seat_index}号", "roi": roi})

    output_data = {
        "source_image": SOURCE_IMAGE,
        "image_size": IMAGE_SIZE,
        "note": "网络演示素材的 ROI 初稿；真实拍摄素材到位后需用 roi_tool.py 重新标定",
        "seats": seats,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=2)

    print(f"[完成] 已写入 {len(seats)} 个座位框：{OUTPUT_PATH}")
    for seat in seats:
        print(f"  {seat['id']}  roi={seat['roi']}")


if __name__ == "__main__":
    main()
