"""FastAPI 服务：提供座位状态接口与网页看板。

接口一览：
    GET  /               网页看板（app/static/index.html）
    GET  /api/health     健康检查，返回模型与数据库就绪状态
    GET  /api/seats      当前 12 座状态（取每个座位最新一条观测）
    GET  /api/events     占座事件列表，默认最近 20 条
    GET  /api/summary    当日统计（各状态帧数与占座次数）
    POST /api/detect     用配置里的演示底图跑一次检测并落库，返回最新状态
    GET  /api/frame      最近一次检测的标注图

启动：
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import os
import sys
from datetime import datetime

import cv2
import yaml
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# 从任意工作目录启动都能导入项目内模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from app import db  # noqa: E402
from app.detector import detect_image  # noqa: E402
from judge_seats import draw_result, judge_seats as judge_seat_states  # noqa: E402

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
STATIC_DIR = os.path.join(PROJECT_ROOT, "app", "static")

STATE_LABELS_CN = {
    "OCCUPIED": "有人",
    "OBJECT_ONLY": "有物无人",
    "EMPTY": "空置",
}

with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)

app = FastAPI(title="自习室座位占用监测", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_connection():
    """每次请求新建连接：sqlite3 连接不能跨线程复用。"""
    return db.connect(os.path.join(PROJECT_ROOT, CONFIG["storage"]["db_path"]))


def resolve_weights() -> str:
    """取权重路径，缺失时回退到预训练模型。"""
    weights = CONFIG["model"]["weights"]
    if os.path.isfile(os.path.join(PROJECT_ROOT, weights)):
        return os.path.join(PROJECT_ROOT, weights)
    return os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")


def build_counts(seats: list) -> dict:
    """统计各状态座位数。"""
    counts = {"OCCUPIED": 0, "OBJECT_ONLY": 0, "EMPTY": 0, "VIOLATION": 0}
    for seat in seats:
        state = seat["state"]
        if state in counts:
            counts[state] += 1
        if seat["violation"]:
            counts["VIOLATION"] += 1
    return counts


def load_current_seats() -> dict:
    """组装当前状态：座位静态信息 + 最新观测 + 进行中的占座事件。"""
    connection = get_connection()
    try:
        latest_rows = db.fetch_latest_snapshots(connection)
        ongoing_rows = db.fetch_ongoing_violations(connection)

        latest_by_seat = {row["seat_id"]: row for row in latest_rows}
        ongoing_by_seat = {row["seat_id"]: row for row in ongoing_rows}

        seats = []
        for seat in CONFIG["seats"]:
            seat_id = seat["id"]
            observation = latest_by_seat.get(seat_id)
            violation_row = ongoing_by_seat.get(seat_id)

            if observation is None:
                state = "EMPTY"
                leave_minutes = 0.0
                person_count = 0
                object_count = 0
                updated_at = ""
            else:
                state = observation["state"]
                leave_minutes = float(observation["leave_minutes"] or 0)
                person_count = int(observation["person_count"] or 0)
                object_count = int(observation["object_count"] or 0)
                updated_at = observation["ts"]

            seats.append(
                {
                    "id": seat_id,
                    "name": seat.get("name", seat_id),
                    "roi": seat["roi"],
                    "state": state,
                    "state_cn": STATE_LABELS_CN.get(state, state),
                    "person_count": person_count,
                    "object_count": object_count,
                    "leave_minutes": round(leave_minutes, 1),
                    "violation": violation_row is not None,
                    "violation_since": violation_row["start_ts"] if violation_row else "",
                    "updated_at": updated_at,
                }
            )

        latest_ts = max((row["ts"] for row in latest_rows), default="")
        return {
            "updated_at": latest_ts,
            "server_time": datetime.now().isoformat(timespec="seconds"),
            "counts": build_counts(seats),
            "threshold_minutes": CONFIG["occupancy"]["object_leave_minutes"],
            "seats": seats,
        }
    finally:
        connection.close()


@app.get("/")
def index():
    """返回看板页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    """健康检查：数据库可写、权重存在。"""
    weights = resolve_weights()
    connection = get_connection()
    try:
        seat_total = connection.execute("SELECT COUNT(*) FROM seats").fetchone()[0]
    finally:
        connection.close()

    return {
        "status": "ok",
        "weights": os.path.relpath(weights, PROJECT_ROOT),
        "weights_exists": os.path.isfile(weights),
        "seats_in_db": seat_total,
        "threshold_minutes": CONFIG["occupancy"]["object_leave_minutes"],
    }


@app.get("/api/seats")
def api_seats():
    """当前座位状态。"""
    return load_current_seats()


@app.get("/api/events")
def api_events(limit: int = Query(20, ge=1, le=200), seat_id: str = Query("", max_length=8)):
    """占座事件列表，按开始时间倒序。"""
    connection = get_connection()
    try:
        rows = db.fetch_events(connection, limit=limit, seat_id=seat_id or None)
        events = [
            {
                "seat_id": row["seat_id"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"] or "",
                "duration_minutes": float(row["duration_minutes"] or 0),
                "outcome": row["outcome"] or "ONGOING",
                "threshold_minutes": float(row["threshold_minutes"] or 0),
            }
            for row in rows
        ]
    finally:
        connection.close()

    return {"count": len(events), "events": events}


@app.get("/api/summary")
def api_summary(day: str = Query("", max_length=10)):
    """当日统计：各状态观测帧数与占座次数。"""
    day_prefix = day or datetime.now().strftime("%Y-%m-%d")

    connection = get_connection()
    try:
        rows = db.summarize_day(connection, day_prefix)
        summary = [
            {
                "seat_id": row["seat_id"],
                "frames": int(row["frames"]),
                "occupied_frames": int(row["occupied_frames"] or 0),
                "object_frames": int(row["object_frames"] or 0),
                "empty_frames": int(row["empty_frames"] or 0),
                "violation_count": int(row["violation_count"] or 0),
            }
            for row in rows
        ]
    finally:
        connection.close()

    return {"day": day_prefix, "seats": summary}


@app.post("/api/detect")
def api_detect():
    """用演示底图跑一次检测：判定 → 写观测 → 输出标注图。"""
    image_path = os.path.join(PROJECT_ROOT, CONFIG["demo"]["image"])
    if not os.path.isfile(image_path):
        return JSONResponse(status_code=400, content={"error": f"演示底图不存在：{CONFIG['demo']['image']}"})

    frame = cv2.imread(image_path)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "演示底图读取失败"})

    detections, canvas = detect_image(
        frame,
        resolve_weights(),
        CONFIG["model"]["conf"],
        CONFIG["model"]["imgsz"],
        CONFIG["model"]["device"],
    )
    seat_results = judge_seat_states(detections, CONFIG["seats"], CONFIG["occupancy"])

    # 落库：每座位一条观测，时间取当前时刻
    moment = datetime.now()
    connection = get_connection()
    try:
        for item in seat_results:
            db.insert_observation(
                connection,
                moment,
                item["id"],
                item["state"],
                item["state"],
                item["person_count"],
                item["object_count"],
                0.0,
                False,
            )
        connection.commit()
    finally:
        connection.close()

    # 标注图供 /api/frame 展示
    annotated_path = os.path.join(PROJECT_ROOT, CONFIG["demo"]["annotated_image"])
    os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
    annotated = draw_result(canvas, CONFIG["seats"], seat_results, detections)
    cv2.imwrite(annotated_path, annotated)

    payload = load_current_seats()
    payload["detections"] = len(detections)
    payload["source"] = CONFIG["demo"]["image"]
    return payload


@app.get("/api/frame")
def api_frame():
    """返回最近一次检测的标注图；尚未检测过时回退到演示底图。"""
    annotated_path = os.path.join(PROJECT_ROOT, CONFIG["demo"]["annotated_image"])
    if os.path.isfile(annotated_path):
        return FileResponse(annotated_path, media_type="image/jpeg")

    fallback = os.path.join(PROJECT_ROOT, CONFIG["demo"]["image"])
    if os.path.isfile(fallback):
        return FileResponse(fallback, media_type="image/jpeg")

    return JSONResponse(status_code=404, content={"error": "尚无可用图像"})
