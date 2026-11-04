"""座位 ROI 标定工具：在教室全景图上框出每个座位，导出坐标供判定模块使用。

设计要点：
    1. 交互拖框 —— 鼠标左键拖出矩形，按 A01、A02 … 顺序自动编号，无需手动填写；
    2. 一键回退 —— 按 u 撤销上一个框、按 c 清空重来，标错不用重启程序；
    3. 可续编辑 —— 用 --load 读入已有 rois.json 继续微调，不必从头再标；
    4. 双重入口 —— 支持纯手动拖框，也支持 --auto-grid 按机位布局自动铺格子。

快捷键：
    鼠标左键拖动 = 画一个座位框      u = 撤销上一个框
    s = 保存并退出                    c = 清空全部
    ESC / q = 不保存直接退出

用法示例：
    # 手动标定
    python scripts/roi_tool.py --image data\\raw\\20261104_0900_sim1_f0001.jpg

    # 读入已有标定继续微调
    python scripts/roi_tool.py --image demo.jpg --load config\\rois.json

    # 仿真视频配套：按布局 JSON 自动铺 4 × 3 网格
    python scripts/roi_tool.py --image demo.jpg --auto-grid \\
        --layout data\\video_demo\\sim_cam1.layout.json

    # 无界面预览：把已标定的 ROI 画到底图上另存，供快速核对
    python scripts/roi_tool.py --image demo.jpg --load config\\rois.json \\
        --preview out\\roi_preview.jpg
"""

import argparse
import json
import os
import sys

import cv2

# 显示窗口的最大宽度；原图更宽时按比例缩小显示，鼠标坐标再换算回原图
MAX_DISPLAY_WIDTH = 1280

# 交互窗口标题（cv2 窗口标题不支持中文，统一用英文）
WINDOW_TITLE = "Seat ROI Tool"

# 交互状态：当前正在拖拽的框的起点与终点（原图坐标系）
drag_start = None
drag_end = None


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="座位 ROI 标定工具")
    parser.add_argument("--image", required=True, help="底图路径（教室全景图）")
    parser.add_argument("--layout", default="", help="机位布局 JSON，配合 --auto-grid 使用")
    parser.add_argument("--load", default="", help="已有 rois.json 路径，读入后继续编辑")
    parser.add_argument("--auto-grid", action="store_true", help="启动时按布局自动铺设座位网格")
    parser.add_argument("--rows", type=int, default=4, help="座位行数，默认 4")
    parser.add_argument("--cols", type=int, default=3, help="座位列数，默认 3")
    parser.add_argument("--preview", default="", help="预览模式：把 ROI 画到底图另存后退出")
    parser.add_argument("--out", default="config/rois.json", help="标定结果保存路径")
    return parser.parse_args()


def load_base_image(image_path: str):
    """加载底图，失败返回 None。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 图片读取失败：{image_path}")
        return None
    return image


def load_layout_quad(layout_path: str):
    """读取机位布局 JSON，取出座位平面四角坐标。"""
    if not os.path.isfile(layout_path):
        print(f"[错误] 找不到布局文件：{layout_path}")
        return None

    with open(layout_path, "r", encoding="utf-8") as layout_file:
        layout = json.load(layout_file)

    quad = layout.get("quad")
    if not quad:
        print("[错误] 布局文件中缺少 quad 字段")
        return None
    return quad


def load_rois(rois_path: str):
    """读入已有标定文件，返回座位框列表。"""
    if not os.path.isfile(rois_path):
        print(f"[错误] 找不到标定文件：{rois_path}")
        return None

    with open(rois_path, "r", encoding="utf-8") as rois_file:
        data = json.load(rois_file)

    seat_boxes = data.get("seats", [])
    print(f"[信息] 已读入 {len(seat_boxes)} 个座位框：{rois_path}")
    return seat_boxes


def project_point(u: float, v: float, quad: dict) -> tuple:
    """把座位平面坐标 (u, v) 映射为画面像素坐标，与视频生成端几何一致。

    u：从左到右取 0 到 1；v：从远到近取 0 到 1。
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


def build_grid_boxes(quad: dict, rows: int, cols: int) -> list:
    """在透视四边形内铺设 rows × cols 的座位框，编号规则与手册一致。"""
    seat_boxes = []

    for row_index in range(rows):
        for col_index in range(cols):
            seat_id = f"{chr(ord('A') + row_index)}{col_index + 1:02d}"
            center_u = (col_index + 0.5) / cols
            center_v = (row_index + 0.5) / rows
            half_u = 0.42 / cols
            half_v = 0.38 / rows

            # 座位四角投影到画面后取外接矩形，保证 ROI 覆盖整张桌面
            corners = [
                project_point(center_u - half_u, center_v - half_v, quad),
                project_point(center_u + half_u, center_v - half_v, quad),
                project_point(center_u + half_u, center_v + half_v, quad),
                project_point(center_u - half_u, center_v + half_v, quad),
            ]
            xs = [corner[0] for corner in corners]
            ys = [corner[1] for corner in corners]
            seat_boxes.append({"id": seat_id, "roi": [min(xs), min(ys), max(xs), max(ys)]})

    print(f"[信息] 自动铺设 {rows} 行 × {cols} 列 = {len(seat_boxes)} 个座位框")
    return seat_boxes


def make_seat_id(existing_count: int, index_offset: int = 0) -> str:
    """按已有框数量顺序生成编号：第 0 个是 A01，第 12 个是 B01，以此类推。"""
    sequence = existing_count + index_offset
    row_letter = chr(ord("A") + sequence // 12)
    seat_number = sequence % 12 + 1
    return f"{row_letter}{seat_number:02d}"


def save_rois(image_path: str, image_shape: tuple, seat_boxes: list, out_path: str) -> None:
    """把标定结果写入 JSON，并在控制台打印可直接粘贴进 config.yaml 的片段。"""
    height, width = image_shape[:2]
    output_data = {
        "source_image": image_path,
        "image_size": [width, height],
        "seats": seat_boxes,
    }

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_file:
        json.dump(output_data, out_file, ensure_ascii=False, indent=2)

    print(f"[完成] 已保存 {len(seat_boxes)} 个座位框：{os.path.abspath(out_path)}")
    print("[提示] 以下片段可直接粘贴到 config/config.yaml 的 seats 段：")
    print("seats:")
    for seat in seat_boxes:
        x1, y1, x2, y2 = seat["roi"]
        print(f"  - id: {seat['id']}")

        print(f"    roi: [{x1}, {y1}, {x2}, {y2}]")


def render_overlay(image, seat_boxes: list, display_scale: float):
    """把已有框、编号与正在拖拽的框画到底图副本上，并缩放到显示尺寸。"""
    canvas = image.copy()

    for seat in seat_boxes:
        x1, y1, x2, y2 = seat["roi"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            canvas,
            seat["id"],
            (x1 + 4, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 0),
            2,
        )

    if drag_start is not None and drag_end is not None:
        cv2.rectangle(canvas, drag_start, drag_end, (0, 120, 255), 2)

    if display_scale < 1.0:
        display_width = int(canvas.shape[1] * display_scale)
        display_height = int(canvas.shape[0] * display_scale)
        canvas = cv2.resize(canvas, (display_width, display_height))
    return canvas


def run_interactive(image, image_path: str, seat_boxes: list, display_scale: float, out_path: str) -> None:
    """打开窗口进入交互标定循环。"""
    global drag_start, drag_end

    mouse_param = {"boxes": seat_boxes, "display_scale": display_scale}
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse, mouse_param)

    print("[操作] 左键拖动画框 | u 撤销 | c 清空 | s 保存退出 | ESC 直接退出")
    while True:
        canvas = render_overlay(image, seat_boxes, display_scale)
        cv2.imshow(WINDOW_TITLE, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and seat_boxes:
            removed = seat_boxes.pop()
            print(f"[撤销] 已移除最后一个框")
        elif key == ord("c"):
            seat_boxes.clear()
            print("[清空] 已移除全部框")
        elif key == ord("s"):
            save_rois(image_path, image.shape, seat_boxes, out_path)
            break
        elif key == 27 or key == ord("q"):
            print("[退出] 未保存，直接关闭")
            break

    cv2.destroyAllWindows()


def on_mouse(event: int, x: int, y: int, flags: int, param: dict) -> None:
    """鼠标回调：换算回原图坐标，记录拖拽框；松开左键时完成一个座位框。"""
    global drag_start, drag_end

    scale = param["display_scale"]
    real_x = int(x / scale)
    real_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (real_x, real_y)
        drag_end = (real_x, real_y)
    elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
        drag_end = (real_x, real_y)
    elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
        drag_end = (real_x, real_y)
        x_low, x_high = sorted([drag_start[0], drag_end[0]])
        y_low, y_high = sorted([drag_start[1], drag_end[1]])

        # 过滤误触产生的极小框
        if x_high - x_low > 8 and y_high - y_low > 8:
            new_box = {
                "id": make_seat_id(len(param["boxes"])),
                "roi": [x_low, y_low, x_high, y_high],
            }
            param["boxes"].append(new_box)
            print(f"[新增] {new_box['id']}  roi={new_box['roi']}")

        drag_start = None
        drag_end = None


def run_preview(image, seat_boxes: list, display_scale: float, preview_path: str) -> None:
    """预览模式：把 ROI 叠加图另存，不打开窗口。"""
    canvas = render_overlay(image, seat_boxes, 1.0)

    preview_dir = os.path.dirname(os.path.abspath(preview_path))
    os.makedirs(preview_dir, exist_ok=True)
    cv2.imwrite(preview_path, canvas)
    print(f"[完成] 预览图已保存：{os.path.abspath(preview_path)}（共 {len(seat_boxes)} 框）")


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()

    image = load_base_image(arguments.image)
    if image is None:
        return 1

    seat_boxes = []
    if arguments.load:
        seat_boxes = load_rois(arguments.load) or []
    elif arguments.auto_grid:
        quad = load_layout_quad(arguments.layout) if arguments.layout else None
        if arguments.layout and quad is None:
            return 1
        if quad is None:
            print("[错误] 自动铺格需要 --layout 布局文件")
            return 1
        seat_boxes = build_grid_boxes(quad, arguments.rows, arguments.cols)

    display_scale = min(1.0, MAX_DISPLAY_WIDTH / image.shape[1])

    if arguments.preview:
        run_preview(image, seat_boxes, display_scale, arguments.preview)
        return 0

    run_interactive(image, arguments.image, seat_boxes, display_scale, arguments.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
