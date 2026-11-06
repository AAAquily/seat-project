"""FastAPI 服务：提供座位状态接口与网页看板。

接口一览：
    GET  /                        网页看板（app/static/index.html）
    GET  /api/health              健康检查，返回模型与数据库就绪状态
    GET  /api/seats               当前 12 座状态（取每个座位最新一条观测）
    GET  /api/events              占座事件列表，默认最近 20 条
    GET  /api/summary             当日统计（各状态帧数与占座次数）
    POST /api/detect              用配置里的演示底图跑一次检测并落库，返回最新状态
    GET  /api/frame               最近一次检测的标注图
    POST /api/analyze/image       上传图片 → 单帧识别，返回座位状态与标注图（不落库）
    POST /api/analyze/video       上传视频 → 建后台任务，逐帧识别并可轮询进度
    GET  /api/jobs/{job_id}       查询视频任务进度、实时座位状态、状态变化时间线
    GET  /api/jobs/{job_id}/frame 取任务当前处理帧的标注图
    POST /api/jobs/{job_id}/cancel 取消正在处理的视频任务

启动：
    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

import base64
import os
import sys
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# 从任意工作目录启动都能导入项目内模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from app import db  # noqa: E402
from app import video_job  # noqa: E402
from app.detector import detect_image, get_model  # noqa: E402
from judge_seats import (  # noqa: E402
    dedup_person_total,
    draw_result,
    judge_seats as judge_seat_states,
    total_person_count,
)
from dynamic_seats import (  # noqa: E402
    analyze_image as analyze_dynamic_scene,
    draw_dynamic,
)
from scene_match import match_scene  # noqa: E402

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
STATIC_DIR = os.path.join(PROJECT_ROOT, "app", "static")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

# 上传约束：图片走内存直接解码，视频落盘后再处理，两者都设上限防止误传大文件
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_VIDEO_BYTES = 300 * 1024 * 1024
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

STATE_LABELS_CN = {
    "OCCUPIED": "有人",
    "OBJECT_ONLY": "有物无人",
    "EMPTY": "空置",
    "ABSENT": "不在画面",
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
    """统计各状态座位数；ABSENT（不在画面）不计入任何占用状态。"""
    counts = {"OCCUPIED": 0, "OBJECT_ONLY": 0, "EMPTY": 0, "VIOLATION": 0, "ABSENT": 0}
    for seat in seats:
        state = seat["state"]
        if state in counts:
            counts[state] += 1
        if seat["violation"]:
            counts["VIOLATION"] += 1
    return counts


def build_uploaded_view(seat_results: list, seat_defs: list = None) -> dict:
    """把单帧判定结果转成看板可直接渲染的结构（与 /api/seats 对齐）。

    seat_defs 为空时（动态座位模式），直接使用结果里自带的 roi / name。
    """
    roi_by_seat = {seat["id"]: seat["roi"] for seat in seat_defs or []}
    name_by_seat = {seat["id"]: seat.get("name", seat["id"]) for seat in seat_defs or []}

    seats = []
    for item in seat_results:
        seats.append(
            {
                "id": item["id"],
                "name": name_by_seat.get(item["id"], item.get("name", item["id"])),
                "roi": roi_by_seat.get(item["id"], item.get("roi", [0, 0, 0, 0])),
                "state": item["state"],
                "state_cn": item.get("state_cn", STATE_LABELS_CN.get(item["state"], item["state"])),
                "person_count": item.get("person_count", 0),
                "object_count": item.get("object_count", 0),
                # 单张图片没有时间维度，离座时长固定为 0
                "leave_minutes": 0.0,
                "violation": False,
                "violation_since": "",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "server_time": datetime.now().isoformat(timespec="seconds"),
        "counts": build_counts(seats),
        # 全图在座人数：跨座位去重后的人框数，用于核对画面里的真实人数
        "total_persons": sum(item.get("person_count", 0) for item in seat_results),
        "threshold_minutes": CONFIG["occupancy"]["object_leave_minutes"],
        "seats": seats,
    }


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> tuple:
    """把图像编码成 JPEG 字节，返回 (是否成功, 字节)。"""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return ok, buffer.tobytes() if ok else b""


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
    seat_results = judge_seat_states(detections, CONFIG["seats"], CONFIG["occupancy"], CONFIG.get("classes"), image=frame)

    # 落库：每座位一条观测，时间取当前时刻
    moment = datetime.now()
    connection = get_connection()
    try:
        for item in seat_results:
            if item["state"] == "ABSENT":
                continue  # 座位不在画面中：不写观测，避免污染时序统计
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
    annotated = draw_result(canvas, CONFIG["seats"], seat_results, detections, CONFIG.get("classes"))
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


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """上传图片 → 单帧识别 → 返回 12 座状态 + 标注图（base64）。

    这条路径只做「看一眼」，不写数据库：看板的历史数据应来自持续采集，
    上传测试图若落库会把实时状态冲掉。
    """
    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_IMAGE_EXT:
        return JSONResponse(
            status_code=400,
            content={"error": f"不支持的图片格式「{extension or '未知'}」，请上传 JPG / PNG / WebP / BMP"},
        )

    payload = await file.read()
    if not payload:
        return JSONResponse(status_code=400, content={"error": "上传内容为空"})
    if len(payload) > MAX_IMAGE_BYTES:
        return JSONResponse(
            status_code=400,
            content={"error": f"图片过大（{len(payload) / 1024 / 1024:.1f} MB），请压缩到 25 MB 以内"},
        )

    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(status_code=400, content={"error": "图片解码失败，文件可能已损坏"})

    started_at = time.perf_counter()

    # 场景路由：与参考图特征匹配成功 → 固定座位表（标定场景，精度最高）；
    # 匹配不上 → 动态座位识别（不依赖 ROI，任意场景可用）。
    matched, reference, inliers = match_scene(frame, CONFIG, PROJECT_ROOT)
    if matched:
        detections, canvas = detect_image(
            frame,
            resolve_weights(),
            CONFIG["model"].get("scene_conf", CONFIG["model"]["conf"]),
            CONFIG["model"]["imgsz"],
            CONFIG["model"]["device"],
        )
        seat_results = judge_seat_states(detections, CONFIG["seats"], CONFIG["occupancy"], CONFIG.get("classes"), image=frame)
        annotated = draw_result(canvas, CONFIG["seats"], seat_results, detections, CONFIG.get("classes"))
        view = build_uploaded_view(seat_results, CONFIG["seats"])
        view["scene_mode"] = "calibrated"
        view["matched_reference"] = reference
        view["match_inliers"] = inliers
        view["total_persons"] = dedup_person_total(detections)
        view["detections"] = len(detections)
    else:
        coco_model = get_model(os.path.join(PROJECT_ROOT, "models", "yolov8n.pt"))
        seat_model = get_model(resolve_weights())
        detection, seat_results = analyze_dynamic_scene(frame, seat_model, coco_model, device=CONFIG["model"]["device"])
        annotated = draw_dynamic(frame, seat_results, detection)
        view = build_uploaded_view(seat_results)
        view["scene_mode"] = detection["mode"]
        view["total_persons"] = len(detection["persons"])
        view["detections"] = len(detection["persons"]) + len(detection["objects"]) + len(detection["chairs"])

    encoded_ok, jpeg = encode_jpeg(annotated, 85)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    view["mode"] = "image"
    view["source"] = filename
    view["image_size"] = [int(frame.shape[1]), int(frame.shape[0])]
    view["elapsed_ms"] = elapsed_ms
    view["annotated"] = (
        "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii") if encoded_ok else ""
    )
    return view


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    time_scale: float = Form(60.0),
    sample_interval: float = Form(1.0),
):
    """上传视频 → 建后台任务逐帧识别，返回 job_id 供前端轮询进度。

    time_scale 是时间倍率：视频内 1 秒按几秒计入占座计时。
    默认 60 表示 40 秒素材即可演示满 30 分钟的占座判定；设 1 则按视频真实时长。
    """
    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_VIDEO_EXT:
        return JSONResponse(
            status_code=400,
            content={"error": f"不支持的视频格式「{extension or '未知'}」，请上传 MP4 / AVI / MOV / MKV / WebM"},
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    target_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{extension}")

    total_bytes = 0
    try:
        with open(target_path, "wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_BYTES:
                    handle.close()
                    os.remove(target_path)
                    return JSONResponse(
                        status_code=400,
                        content={"error": "视频超过 300 MB，请截取片段或压缩后再上传"},
                    )
                handle.write(chunk)
    except Exception as error:  # noqa: BLE001
        if os.path.isfile(target_path):
            os.remove(target_path)
        return JSONResponse(status_code=500, content={"error": f"保存上传文件失败：{error}"})

    if total_bytes == 0:
        os.remove(target_path)
        return JSONResponse(status_code=400, content={"error": "上传内容为空"})

    job = video_job.create_job(filename, target_path, time_scale, sample_interval)
    video_job.start_job(job, CONFIG, resolve_weights())

    return {
        "job_id": job.job_id,
        "filename": filename,
        "size_mb": round(total_bytes / 1024 / 1024, 2),
        "time_scale": job.time_scale,
        "sample_interval": job.sample_interval,
        "threshold_minutes": CONFIG["occupancy"]["object_leave_minutes"],
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    """查询视频任务：进度、当前座位状态、状态变化时间线、占座事件。"""
    job = video_job.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或服务已重启"})

    snapshot = job.snapshot()
    snapshot["threshold_minutes"] = CONFIG["occupancy"]["object_leave_minutes"]
    snapshot["mode"] = "video"
    return snapshot


@app.get("/api/jobs/{job_id}/frame")
def job_frame(job_id: str):
    """取任务当前帧的标注图；尚未处理出第一帧时返回 204。"""
    job = video_job.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或服务已重启"})

    jpeg = job.current_frame()
    if not jpeg:
        return Response(status_code=204)

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    """取消正在处理的视频任务。"""
    job = video_job.get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或服务已重启"})

    job.cancel()
    return {"job_id": job_id, "status": "cancelling"}

