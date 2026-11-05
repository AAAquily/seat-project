"""时序跟踪主程序：持续读取画面，逐帧判定座位状态，累计离座时长并落库。

数据流：帧源 → YOLOv8 检测 → 座位三态判定 → 状态机（防抖/计时/豁免）→ SQLite 与事件导出

三种帧源：
    --source images   图片序列（单张或目录），按文件名顺序读取，可循环多轮
    --source video    视频文件，按 --frame-step 跳帧采样
    --source camera   本机摄像头或 RTSP 流

时间模拟（演示 30 分钟占座阈值的关键）：
    --minutes-per-frame 指定每个处理帧代表多少分钟，默认 1.0。
    例如单张图片循环 35 轮 = 虚拟时间前进 35 分钟，可快速验证占座判定，无需真等半小时。

用法示例：
    # 用主机位图循环 35 轮（虚拟 35 分钟），验证 C 桌占座能否触发
    python scripts/track_stream.py --source images --images data/video_demo/main_view.jpg --rounds 35

    # 用真实视频（每帧代表 1 分钟压缩演示）
    python scripts/track_stream.py --source video --video data/video_demo/sim_cam1.mp4 --frame-step 30

    # 接摄像头实时运行（每帧代表 1 分钟仅用于演示，生产环境应改为 --minutes-per-frame 1）
    python scripts/track_stream.py --source camera --camera 0
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

import cv2
import yaml
from ultralytics import YOLO

# 保证从任意工作目录启动都能导入项目内模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from app import db  # noqa: E402
from app.state_machine import (  # noqa: E402
    OUTCOME_ONGOING,
    STATE_EMPTY,
    STATE_OBJECT_ONLY,
    STATE_OCCUPIED,
    SeatTracker,
)
from judge_seats import draw_result, judge_seats as judge_seat_states  # noqa: E402

STATE_LABELS_CN = {
    STATE_OCCUPIED: "有人",
    STATE_OBJECT_ONLY: "有物无人",
    STATE_EMPTY: "空置",
}


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="座位状态时序跟踪")

    parser.add_argument("--source", choices=["images", "video", "camera"], default="images", help="帧源类型")
    parser.add_argument("--images", default="data/video_demo/main_view.jpg", help="图片路径或目录")
    parser.add_argument("--video", default="", help="视频文件路径")
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号，默认 0")
    parser.add_argument("--frame-step", type=int, default=30, help="视频每隔多少帧取一帧，默认 30")
    parser.add_argument("--rounds", type=int, default=1, help="图片序列循环轮数，默认 1")
    parser.add_argument("--config", default="config/config.yaml", help="主配置路径")
    parser.add_argument("--seats-key", default="seats", help="使用哪套座位表：seats / video_seats")
    parser.add_argument("--db", default="", help="数据库路径，默认取配置 storage.db_path")
    parser.add_argument("--start-time", default="", help="虚拟起始时间 YYYY-MM-DDTHH:MM，默认今天 09:00")
    parser.add_argument("--minutes-per-frame", type=float, default=1.0, help="每帧代表的虚拟分钟数，默认 1.0")
    parser.add_argument("--save-every", type=int, default=0, help="每隔多少帧保存一张标注图，0 表示不保存")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限制")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """读取主配置。"""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def collect_image_paths(images_argument: str) -> list:
    """收集图片路径：目录则按文件名排序取全部 jpg/png，单文件则只取该文件。"""
    if os.path.isdir(images_argument):
        names = sorted(os.listdir(images_argument))
        return [
            os.path.join(images_argument, name)
            for name in names
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if os.path.isfile(images_argument):
        return [images_argument]

    return []


def iter_frames(arguments: argparse.Namespace):
    """按帧源产出 (BGR 图像, 帧描述) 二元组，供主循环消费。"""
    if arguments.source == "images":
        paths = collect_image_paths(arguments.images)
        if not paths:
            print(f"[错误] 未找到图片：{arguments.images}")
            return

        for round_index in range(max(1, arguments.rounds)):
            for path in paths:
                image = cv2.imread(path)
                if image is None:
                    print(f"[跳过] 图片读取失败：{path}")
                    continue
                yield image, f"{os.path.basename(path)}#{round_index + 1}"
        return

    if arguments.source == "video":
        if not arguments.video or not os.path.isfile(arguments.video):
            print(f"[错误] 视频不存在：{arguments.video}")
            return

        capture = cv2.VideoCapture(arguments.video)
        frame_index = 0
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break

            if frame_index % max(1, arguments.frame_step) == 0:
                yield frame, f"video_frame{frame_index}"
            frame_index += 1
        capture.release()
        return

    # camera 分支：本机摄像头或 RTSP
    capture = cv2.VideoCapture(arguments.camera)
    if not capture.isOpened():
        print(f"[错误] 摄像头打开失败：{arguments.camera}")
        return

    while True:
        success, frame = capture.read()
        if not success:
            break
        yield frame, "camera"
    capture.release()


def detect_frame(model: YOLO, image, conf: float, imgsz: int, device: str) -> list:
    """对单帧推理，返回统一结构的检测列表。"""
    prediction = model.predict(source=image, conf=conf, imgsz=imgsz, device=device, verbose=False)
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
    return detections, result.orig_img


def export_events_csv(events: list, export_dir: str, started_at: datetime) -> str:
    """把本次运行产生的占座事件导出为 CSV，便于交作业时附在报告里。"""
    os.makedirs(export_dir, exist_ok=True)
    file_name = f"occupancy_events_{started_at.strftime('%Y%m%d_%H%M%S')}.csv"
    file_path = os.path.join(export_dir, file_name)

    with open(file_path, "w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["座位", "开始时间", "结束时间", "持续分钟", "结局", "阈值分钟"])
        for event in events:
            writer.writerow(
                [
                    event.seat_id,
                    event.start_ts.strftime("%Y-%m-%d %H:%M"),
                    event.end_ts.strftime("%Y-%m-%d %H:%M") if event.end_ts else "",
                    event.duration_minutes,
                    event.outcome,
                    event.threshold_minutes,
                ]
            )
    return file_path


def main() -> int:
    """主流程。"""
    arguments = parse_arguments()
    config = load_config(arguments.config)

    seats = config.get(arguments.seats_key) or config.get("seats") or []
    if not seats:
        print("[错误] 配置中 seats 为空，请先标定座位 ROI")
        return 1

    db_path = arguments.db or config["storage"]["db_path"]
    connection = db.connect(db_path)
    db.upsert_seats(connection, seats)
    dangling = db.close_dangling_events(connection)
    if dangling:
        print(f"[清理] 上次运行遗留 {dangling} 条未结事件，已标记为 INTERRUPTED")

    weights = config["model"]["weights"]
    if not os.path.isfile(weights):
        print(f"[提示] 权重不存在：{weights}，回退 models/yolov8n.pt")
        weights = os.path.join("models", "yolov8n.pt")

    model = YOLO(weights)
    tracker = SeatTracker(
        seat_ids=[seat["id"] for seat in seats],
        leave_minutes=config["occupancy"]["object_leave_minutes"],
        confirm_frames=config["occupancy"]["empty_confirm_frames"],
        exempt_config=config.get("exempt_time"),
    )

    if arguments.start_time:
        clock = datetime.strptime(arguments.start_time, "%Y-%m-%dT%H:%M")
    else:
        clock = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    step = timedelta(minutes=arguments.minutes_per_frame)
    print(f"[启动] 权重={weights} 数据库={db_path}")
    print(f"[启动] 起始时间={clock:%Y-%m-%d %H:%M} 每帧={arguments.minutes_per_frame} 分钟 阈值={config['occupancy']['object_leave_minutes']} 分钟")

    run_dir = os.path.join("runs", "track", clock.strftime("%Y%m%d_%H%M%S"))
    all_events = []
    processed = 0
    seen_events_by_seat = {}
    last_update = None

    for frame_index, (image, frame_label) in enumerate(iter_frames(arguments)):
        if arguments.max_frames and processed >= arguments.max_frames:
            break

        moment = clock + step * frame_index
        conf = config["model"].get("video_conf", config["model"]["conf"]) if arguments.seats_key != "seats" else config["model"]["conf"]
        detections, canvas = detect_frame(model, image, conf, config["model"]["imgsz"], config["model"]["device"])
        seat_results = judge_seat_states(detections, seats, config["occupancy"])

        raw_states = {item["id"]: item["state"] for item in seat_results}
        details = {
            item["id"]: {"person_count": item["person_count"], "object_count": item["object_count"]}
            for item in seat_results
        }

        update = tracker.update(raw_states, moment, details)
        last_update = update

        # 逐座位落库：状态、计数、已累计的离座分钟
        for snapshot in update.snapshots:
            db.insert_observation(
                connection,
                moment,
                snapshot.seat_id,
                snapshot.state,
                snapshot.raw_state,
                snapshot.person_count,
                snapshot.object_count,
                snapshot.leave_minutes,
                snapshot.violation,
            )
        connection.commit()

        # 新开的占座事件立即写库并记住主键，结束时回填
        for event in update.started_events:
            event.event_id = db.open_event(connection, event.seat_id, event.start_ts, event.threshold_minutes)
            seen_events_by_seat[event.seat_id] = event
            all_events.append(event)
            print(f"  [占座] {moment:%H:%M} {event.seat_id} 有物无人已满 {event.threshold_minutes:.0f} 分钟")

        for event in update.closed_events:
            if event.event_id is not None:
                db.close_event(connection, event.event_id, event.end_ts, event.duration_minutes, event.outcome)
            print(f"  [结束] {moment:%H:%M} {event.seat_id} 占座结束（{event.outcome}，共 {event.duration_minutes} 分钟）")

        for change in update.changes:
            print(f"  [变化] {moment:%H:%M} {change.seat_id} {STATE_LABELS_CN[change.old_state]} → {STATE_LABELS_CN[change.new_state]}")

        if arguments.save_every and frame_index % arguments.save_every == 0:
            os.makedirs(run_dir, exist_ok=True)
            annotated = draw_result(canvas, seats, seat_results, detections)
            cv2.imwrite(os.path.join(run_dir, f"frame_{frame_index:04d}.jpg"), annotated)

        processed += 1

    # 收尾：数据流正常结束，未结事件标记为 ONGOING（占座仍在持续），不视为异常中断
    last_moment = clock + step * max(0, processed - 1)
    for event in tracker.close_all(last_moment, OUTCOME_ONGOING):
        if event.event_id is not None:
            db.close_event(connection, event.event_id, event.end_ts, event.duration_minutes, event.outcome)
        print(f"  [收尾] {event.seat_id} 事件标记为 {event.outcome}（{event.duration_minutes} 分钟）")

    final_snapshots = last_update.snapshots if last_update else []

    print("\n================ 运行结束 ================")
    print(f"处理帧数：{processed}　虚拟时长：{arguments.minutes_per_frame * max(0, processed - 1):.0f} 分钟")

    print("\n座位最终状态：")
    for snapshot in final_snapshots:
        flag = "  ← 占座" if snapshot.violation else ""
        print(f"  {snapshot.seat_id:4s} {STATE_LABELS_CN[snapshot.state]:6s} 离座累计 {snapshot.leave_minutes:5.1f} 分钟{flag}")

    if all_events:
        print("\n占座事件：")
        for event in all_events:
            status = f"{event.duration_minutes} 分钟 / {event.outcome}" if event.end_ts else "进行中"
            print(f"  {event.seat_id}: {event.start_ts:%H:%M} 起，{status}")
        export_dir = config["storage"]["export_dir"]
        csv_path = export_events_csv(all_events, export_dir, clock)
        print(f"\n[导出] {csv_path}")
    else:
        print("\n未触发占座事件（累计时长未达阈值，或用例本身无「有物无人」）")

    print(f"[数据库] {os.path.abspath(db_path)}")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
