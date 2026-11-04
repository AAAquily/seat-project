"""生成模拟自习室俯拍视频，用于在真实素材到位前跑通抽帧、ROI 标定与判定链路。

设计要点：
    1. 透视网格 —— 用双线性插值把 4 行 × 3 列的座位平面映射成梯形，模拟后墙高角度俯拍；
    2. 状态驱动 —— 每个座位在「空置 / 有人 / 有物无人」三种状态间轮换，覆盖项目全部判定分支；
    3. 完全可复现 —— 固定随机种子，同一组参数每次生成的画面逐像素一致。

局限说明（重要）：
    合成画面的纹理、光照、人物姿态与真实教室差异很大，只能验证工程流程
    （抽帧 → 标定 → 判定 → 防抖 → 看板），不能替代真实数据训练模型。
    真实素材到位后直接替换 data/raw 下的文件即可，后续脚本无需改动。

用法示例：
    python scripts/make_sim_video.py --seconds 90 --out data\\video_demo\\sim_cam1.mp4
"""

import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np

# ---------------- 画布与透视参数 ----------------

# 座位平面的四个角在画面中的位置：(x, y)，构成一个梯形，近大远小
CANVAS_NEAR_LEFT = (120, 690)
CANVAS_NEAR_RIGHT = (1160, 690)
CANVAS_FAR_LEFT = (330, 210)
CANVAS_FAR_RIGHT = (950, 210)

# 三种座位状态的标识
STATE_EMPTY = "empty"
STATE_PERSON = "person"
STATE_OBJECT = "object"
ALL_STATES = [STATE_EMPTY, STATE_PERSON, STATE_OBJECT]

# 绘制配色（BGR）
FLOOR_COLOR = (238, 234, 226)
DESK_TOP_COLOR = (168, 152, 132)
DESK_EDGE_COLOR = (128, 112, 96)
CLOTH_COLORS = [(58, 76, 128), (92, 92, 100), (140, 90, 60), (70, 110, 90), (120, 70, 110)]
SKIN_COLOR = (150, 170, 200)
BOOK_COLORS = [(70, 100, 150), (60, 130, 90), (150, 90, 70), (110, 80, 140)]
BAG_COLORS = [(45, 55, 70), (60, 45, 80), (50, 70, 60)]
CUP_COLORS = [(200, 200, 210), (120, 150, 190)]

# 亮度波动的幅度（灰度级），模拟日光灯下的缓慢明暗漂移
LIGHT_WAVE_AMPLITUDE = 3.0

# 噪声图缓存：预生成若干张轮流复用，避免逐帧重算数百万个随机数
NOISE_CACHE = []
NOISE_PATTERN_COUNT = 10
NOISE_STD = 2.2


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="生成模拟自习室俯拍视频")
    parser.add_argument("--seconds", type=float, default=90, help="视频时长（秒），默认 90")
    parser.add_argument("--fps", type=int, default=30, help="帧率，默认 30")
    parser.add_argument("--width", type=int, default=1280, help="画面宽度，默认 1280")
    parser.add_argument("--height", type=int, default=720, help="画面高度，默认 720")
    parser.add_argument("--rows", type=int, default=4, help="座位行数，默认 4")
    parser.add_argument("--cols", type=int, default=3, help="座位列数，默认 3")
    parser.add_argument("--hold-sec", type=float, default=12, help="每隔多少秒重排一次座位状态")
    parser.add_argument("--seed", type=int, default=20261104, help="随机种子，保证结果可复现")
    parser.add_argument("--out", default="data/video_demo/sim_cam1.mp4", help="输出视频路径")
    parser.add_argument("--layout-out", default="", help="同时输出座位角点 JSON，供 ROI 工具对齐")
    return parser.parse_args()


def scale_point(point: tuple, width: int, height: int) -> tuple:
    """把按 1280 × 720 设计的角点，按实际画布尺寸等比缩放。"""
    base_width, base_height = 1280, 720
    scaled_x = int(point[0] * width / base_width)
    scaled_y = int(point[1] * height / base_height)
    return scaled_x, scaled_y


def project_point(u: float, v: float, quad: dict) -> tuple:
    """把座位平面坐标 (u, v) 映射到画面像素坐标。

    u：从左到右，0 为最左、1 为最右；
    v：从远到近，0 为最远一排、1 为最近一排。
    """
    far_left = quad["far_left"]
    far_right = quad["far_right"]
    near_left = quad["near_left"]
    near_right = quad["near_right"]

    far_x = far_left[0] + (far_right[0] - far_left[0]) * u
    far_y = far_left[1] + (far_right[1] - far_left[1]) * u
    near_x = near_left[0] + (near_right[0] - near_left[0]) * u
    near_y = near_left[1] + (near_right[1] - near_left[1]) * u

    point_x = far_x + (near_x - far_x) * v
    point_y = far_y + (near_y - far_y) * v
    return int(point_x), int(point_y)


def build_seat_list(rows: int, cols: int) -> list:
    """按「行字母 + 两位列号」生成座位清单，行号从远到近依次为 A、B、C …"""
    seats = []
    for row_index in range(rows):
        for col_index in range(cols):
            seat_id = f"{chr(ord('A') + row_index)}{col_index + 1:02d}"
            seats.append(
                {
                    "id": seat_id,
                    "row": row_index,
                    "col": col_index,
                    "center_u": (col_index + 0.5) / cols,
                    "center_v": (row_index + 0.5) / rows,
                }
            )
    return seats


def draw_background(width: int, height: int, quad: dict) -> np.ndarray:
    """绘制地面与墙体，作为每帧的底图（不随后续绘制变化）。"""

    frame = np.full((height, width, 3), FLOOR_COLOR, dtype=np.uint8)

    # 地面区域：座位梯形向外扩展一圈，画出略深的地板色
    floor_points = np.array(
        [
            project_point(-0.12, 1.08, quad),
            project_point(1.12, 1.08, quad),
            project_point(1.12, -0.10, quad),
            project_point(-0.12, -0.10, quad),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [floor_points], (226, 220, 210))

    # 远端的墙面，帮助人眼（和标注者）判断场景朝向
    wall_points = np.array(
        [
            (0, 0),
            (width, 0),
            (width, int(height * 0.30)),
            (0, int(height * 0.30)),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [wall_points], (206, 200, 192))
    return frame


def draw_desk(frame: np.ndarray, seat: dict, rows: int, cols: int, quad: dict) -> None:
    """绘制一张课桌：桌面用透视四边形，前沿画一条深色边表示桌沿厚度。"""
    half_u = 0.42 / cols
    half_v = 0.38 / rows
    center_u = seat["center_u"]
    center_v = seat["center_v"]

    corners = [
        project_point(center_u - half_u, center_v - half_v, quad),
        project_point(center_u + half_u, center_v - half_v, quad),
        project_point(center_u + half_u, center_v + half_v, quad),
        project_point(center_u - half_u, center_v + half_v, quad),
    ]
    polygon = np.array(corners, dtype=np.int32)
    cv2.fillPoly(frame, [polygon], DESK_TOP_COLOR)
    cv2.polylines(frame, [polygon], isClosed=True, color=DESK_EDGE_COLOR, thickness=2)

    # 桌沿：把靠镜头一侧的边加粗，制造厚度感
    front_left = corners[3]
    front_right = corners[2]
    cv2.line(frame, front_left, front_right, DESK_EDGE_COLOR, thickness=4)


def draw_person(frame: np.ndarray, seat: dict, quad: dict, random_source: random.Random) -> None:
    """绘制一个俯视视角的人：肩膀椭圆 + 头部圆，颜色随机但同一座位保持不变。"""
    center_u = seat["center_u"]
    center_v = seat["center_v"]
    person_x, person_y = project_point(center_u, center_v - 0.05, quad)

    # 近处的人大、远处的人小，还原透视比例
    body_radius = int(16 + 22 * center_v)
    cloth_color = random_source.choice(CLOTH_COLORS)

    shoulder_axes = (int(body_radius * 1.5), int(body_radius * 0.9))
    cv2.ellipse(frame, (person_x, person_y), shoulder_axes, 0, 0, 360, cloth_color, -1)
    head_radius = max(6, int(body_radius * 0.55))
    cv2.circle(frame, (person_x, person_y - int(body_radius * 0.15)), head_radius, SKIN_COLOR, -1)
    cv2.circle(frame, (person_x, person_y - int(body_radius * 0.15)), head_radius, (110, 110, 110), 1)


def draw_object(frame: np.ndarray, seat: dict, quad: dict, random_source: random.Random) -> None:
    """绘制遗留物品：书本（叠放矩形）、背包（圆角矩形）、水杯（小圆柱），三者随机组合。"""
    center_u = seat["center_u"]
    center_v = seat["center_v"]
    base_x, base_y = project_point(center_u, center_v + 0.02, quad)
    unit = int(6 + 14 * center_v)

    object_kind = random_source.choice(["book", "bag", "cup", "book_cup", "bag_cup"])

    if "book" in object_kind:
        book_color = random_source.choice(BOOK_COLORS)
        for layer in range(2):
            offset_y = base_y - layer * max(4, unit // 2)
            cv2.rectangle(
                frame,
                (base_x - unit * 2, offset_y - unit),
                (base_x + unit * 2, offset_y + unit // 2),
                book_color,
                -1,
            )
            cv2.rectangle(
                frame,
                (base_x - unit * 2, offset_y - unit),
                (base_x + unit * 2, offset_y + unit // 2),
                (40, 40, 40),
                1,
            )

    if "bag" in object_kind:
        bag_color = random_source.choice(BAG_COLORS)
        bag_center = (base_x + int(unit * 3.2), base_y - int(unit * 0.4))
        cv2.ellipse(frame, bag_center, (int(unit * 1.6), int(unit * 1.2)), 0, 0, 360, bag_color, -1)
        cv2.ellipse(frame, bag_center, (int(unit * 1.6), int(unit * 1.2)), 0, 0, 360, (30, 30, 30), 1)

    if "cup" in object_kind:
        cup_color = random_source.choice(CUP_COLORS)
        cup_center = (base_x - int(unit * 2.8), base_y + int(unit * 0.6))
        cv2.circle(frame, cup_center, max(4, unit // 2), cup_color, -1)
        cv2.circle(frame, cup_center, max(4, unit // 2), (90, 90, 90), 1)


def assign_seat_states(seats: list, seed: int, epoch_index: int, rows: int, cols: int) -> dict:
    """为当前时间段重新分配座位状态，保证三种状态都有出现，便于覆盖判定分支。"""
    random_source = random.Random(seed + epoch_index * 7919)
    states = {}

    for seat in seats:
        states[seat["id"]] = random_source.choice(ALL_STATES)

    # 强制保证每个 epoch 至少有一个空座、一个有人、一个有物无人，避免样本偏斜
    seat_ids = [seat["id"] for seat in seats]
    states[random_source.choice(seat_ids)] = STATE_EMPTY
    states[random_source.choice(seat_ids)] = STATE_PERSON
    states[random_source.choice(seat_ids)] = STATE_OBJECT
    return states


def add_light_and_noise(frame: np.ndarray, frame_index: int) -> np.ndarray:
    """叠加缓慢变化的整体亮度与传感器噪声，模拟真实灯光波动与画面颗粒。

    噪声图预生成后轮换复用：逐帧重算数百万个随机数会让 90 秒视频渲染慢十倍以上。
    """
    brightness_shift = int(round(LIGHT_WAVE_AMPLITUDE * math.sin(frame_index / 47.0)))
    adjusted = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_shift)

    noise_pattern = NOISE_CACHE[frame_index % NOISE_PATTERN_COUNT]
    noisy = cv2.add(adjusted, noise_pattern, dtype=cv2.CV_8U)
    return noisy


def build_noise_cache(shape: tuple, seed: int) -> None:
    """预生成一组噪声图，形状与画面一致，供逐帧轮换使用。"""
    NOISE_CACHE.clear()

    random_generator = np.random.default_rng(seed)
    for _ in range(NOISE_PATTERN_COUNT):
        pattern = random_generator.normal(0, NOISE_STD, shape).astype(np.int16)
        NOISE_CACHE.append(pattern)


def build_base_frame(width: int, height: int, quad: dict, seats: list, rows: int, cols: int) -> np.ndarray:
    """渲染静态底图：地面 + 墙体 + 全部课桌。桌子全程不动，只画一次即可。"""
    base_frame = draw_background(width, height, quad)

    for seat in seats:
        draw_desk(base_frame, seat, rows, cols, quad)
    return base_frame


def render_frame(
    base_frame: np.ndarray,
    seats: list,
    states: dict,
    quad: dict,
    frame_index: int,
    second_now: float,
    seed: int,
) -> np.ndarray:
    """渲染单帧画面：在底图上由远及近叠加人与物品，保证近处遮挡远处。"""
    frame = base_frame.copy()

    # 由远及近绘制，近排后画才能盖住远排，符合真实遮挡关系
    for seat in sorted(seats, key=lambda item: item["row"]):
        state = states[seat["id"]]
        if state == STATE_EMPTY:
            continue

        seat_random = random.Random(f"{seed}-{seat['id']}")
        if state == STATE_PERSON:
            draw_person(frame, seat, quad, seat_random)
        else:
            draw_object(frame, seat, quad, seat_random)

    frame = add_light_and_noise(frame, frame_index)

    # 左上角时间戳：既方便人工核对，也避免抽帧脚本把连续画面判为完全重复
    timestamp_text = f"SIM  T+{int(second_now):03d}s  F{frame_index:04d}"
    cv2.putText(frame, timestamp_text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 60, 60), 2)
    return frame


def export_layout(path: str, quad: dict, rows: int, cols: int, width: int, height: int) -> None:
    """输出座位平面四角坐标，供 ROI 标定工具自动对齐网格。"""
    layout = {
        "image_size": [width, height],
        "rows": rows,
        "cols": cols,
        "quad": quad,
    }
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as layout_file:
        json.dump(layout, layout_file, ensure_ascii=False, indent=2)


def make_video(arguments: argparse.Namespace) -> int:
    """生成视频，返回写入的帧数。"""
    width = arguments.width
    height = arguments.height
    rows = arguments.rows
    cols = arguments.cols

    quad = {
        "near_left": scale_point(CANVAS_NEAR_LEFT, width, height),
        "near_right": scale_point(CANVAS_NEAR_RIGHT, width, height),
        "far_left": scale_point(CANVAS_FAR_LEFT, width, height),
        "far_right": scale_point(CANVAS_FAR_RIGHT, width, height),
    }

    seats = build_seat_list(rows, cols)
    base_frame = build_base_frame(width, height, quad, seats, rows, cols)
    build_noise_cache((height, width, 3), arguments.seed)

    output_dir = os.path.dirname(os.path.abspath(arguments.out))
    os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(arguments.out, fourcc, arguments.fps, (width, height))
    if not writer.isOpened():
        print(f"[错误] 无法创建视频文件：{arguments.out}")
        return -1

    total_frames = int(arguments.seconds * arguments.fps)
    print(f"[信息] 画布 {width} × {height}，{rows} 行 × {cols} 列 = {len(seats)} 个座位")
    print(f"[信息] 时长 {arguments.seconds} 秒，帧率 {arguments.fps}，共 {total_frames} 帧")
    print(f"[信息] 状态每 {arguments.hold_sec} 秒重排一次")

    states = {}
    current_epoch = -1

    for frame_index in range(total_frames):
        second_now = frame_index / arguments.fps
        epoch_index = int(second_now // arguments.hold_sec)
        if epoch_index != current_epoch:
            current_epoch = epoch_index
            states = assign_seat_states(seats, arguments.seed, epoch_index, rows, cols)

        frame = render_frame(
            base_frame,
            seats,
            states,
            quad,
            frame_index,
            second_now,
            arguments.seed,
        )
        writer.write(frame)

        if frame_index % (arguments.fps * 10) == 0:
            print(f"[进度] 已生成 {frame_index} / {total_frames} 帧")

    writer.release()
    print(f"[完成] 视频已保存：{os.path.abspath(arguments.out)}")

    if arguments.layout_out:
        export_layout(arguments.layout_out, quad, rows, cols, width, height)
        print(f"[完成] 布局文件已保存：{os.path.abspath(arguments.layout_out)}")

    # 打印最终一帧的座位状态，便于对照检查
    final_states = "，".join(f"{seat_id}={state}" for seat_id, state in sorted(states.items()))
    print(f"[信息] 末段座位状态：{final_states}")
    return total_frames


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()
    result = make_video(arguments)
    if result < 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
