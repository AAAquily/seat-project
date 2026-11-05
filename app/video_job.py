"""上传视频的识别任务：把「一次上传」封装成可查询进度的后台任务。

三个设计要点：

1. 线程内串行推理
   YOLO 权重是进程级单例，多线程同时 predict 会争用模型内部状态，
   因此用一把全局锁把推理串行化；HTTP 层只读任务快照，不会被推理阻塞。

2. 时间轴映射（time_scale）
   真实占座阈值是 30 分钟，而上传的视频往往只有几十秒。为了让短视频也能演示
   「放下东西离开 → 满 30 分钟判占座」的完整链路，这里引入时间倍率：
   视频内 1 秒按 time_scale 秒计入状态机。默认 60（1 秒≈1 分钟），
   40 秒素材即可走完 40 分钟的判定；设成 1 就是按视频真实时长计时。

3. 按秒采样
   CPU 上逐帧推理太慢，默认每 1 秒视频取 1 帧（stride = fps × 采样间隔），
   既不会漏掉状态切换，又能把 90 秒素材压到约 90 次推理。

状态机、判定规则、绘图全部复用已有模块，这里只负责「串起来 + 记录进度」。
"""

import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import cv2

# 允许从任意工作目录导入项目内模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for extra_path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from app.detector import detect_image  # noqa: E402
from app.state_machine import (  # noqa: E402
    OUTCOME_CLEARED,
    OUTCOME_ONGOING,
    OUTCOME_RESOLVED,
    SeatTracker,
)
from judge_seats import draw_result, judge_seats as judge_seat_states  # noqa: E402

# 推理锁：同一个模型对象不能被两个线程同时前向
INFERENCE_LOCK = threading.Lock()

STATE_LABELS_CN = {
    "OCCUPIED": "有人",
    "OBJECT_ONLY": "有物无人",
    "EMPTY": "空置",
}

OUTCOME_LABELS_CN = {
    OUTCOME_ONGOING: "仍在占座",
    OUTCOME_RESOLVED: "人已回来",
    OUTCOME_CLEARED: "物品清走",
}

# 时间线最多保留的条数，避免长视频把内存撑大
MAX_TIMELINE = 200


@dataclass
class VideoJob:
    """一次视频识别任务的完整状态，HTTP 层只读取它。"""

    job_id: str
    filename: str
    video_path: str
    time_scale: float = 60.0
    sample_interval: float = 1.0

    status: str = "pending"          # pending / running / done / error / cancelled
    message: str = ""
    progress: float = 0.0

    frames_total: int = 0
    frames_done: int = 0
    video_duration: float = 0.0
    video_seconds: float = 0.0       # 当前处理到视频的第几秒
    simulated_seconds: float = 0.0   # 换算到状态机的时间（video_seconds × time_scale）

    seats: List[dict] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    timeline: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)

    detections: int = 0
    frame_seq: int = 0
    frame_jpeg: bytes = b""

    elapsed_seconds: float = 0.0
    created_at: str = ""

    cancel_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------- 供 HTTP 层读取 ----------

    def snapshot(self) -> dict:
        """返回可 JSON 序列化的任务快照。"""
        with self._lock:
            return {
                "job_id": self.job_id,
                "filename": self.filename,
                "status": self.status,
                "message": self.message,
                "progress": round(self.progress, 4),
                "frames_total": self.frames_total,
                "frames_done": self.frames_done,
                "video_duration": round(self.video_duration, 2),
                "video_seconds": round(self.video_seconds, 2),
                "simulated_seconds": round(self.simulated_seconds, 2),
                "time_scale": self.time_scale,
                "sample_interval": self.sample_interval,
                "detections": self.detections,
                "frame_seq": self.frame_seq,
                "elapsed_seconds": round(self.elapsed_seconds, 2),
                "created_at": self.created_at,
                "counts": dict(self.counts),
                "seats": list(self.seats),
                "timeline": list(self.timeline),
                "events": list(self.events),
            }

    def current_frame(self) -> bytes:
        with self._lock:
            return self.frame_jpeg

    def cancel(self) -> None:
        with self._lock:
            self.cancel_requested = True

    # ---------- 供处理线程写入 ----------

    def publish(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def push_timeline(self, entry: dict) -> None:
        with self._lock:
            self.timeline.append(entry)
            if len(self.timeline) > MAX_TIMELINE:
                del self.timeline[0 : len(self.timeline) - MAX_TIMELINE]


# 任务表：job_id -> VideoJob
_JOBS: Dict[str, VideoJob] = {}
_JOBS_LOCK = threading.Lock()


def create_job(filename: str, video_path: str, time_scale: float, sample_interval: float) -> VideoJob:
    """登记一个新任务。"""
    job = VideoJob(
        job_id=uuid.uuid4().hex[:12],
        filename=filename,
        video_path=video_path,
        time_scale=max(1.0, float(time_scale)),
        sample_interval=max(0.2, float(sample_interval)),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> Optional[VideoJob]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def list_jobs() -> List[VideoJob]:
    with _JOBS_LOCK:
        return list(_JOBS.values())


def _build_seats(snapshots: list, seat_config: list) -> tuple:
    """把状态机快照转成看板用的座位列表与状态计数。"""
    name_by_id = {seat["id"]: seat.get("name", seat["id"]) for seat in seat_config}
    roi_by_id = {seat["id"]: seat["roi"] for seat in seat_config}

    seats = []
    counts = {"OCCUPIED": 0, "OBJECT_ONLY": 0, "EMPTY": 0, "VIOLATION": 0}

    for snapshot in snapshots:
        state = snapshot.state
        if state in counts:
            counts[state] += 1
        if snapshot.violation:
            counts["VIOLATION"] += 1

        seats.append(
            {
                "id": snapshot.seat_id,
                "name": name_by_id.get(snapshot.seat_id, snapshot.seat_id),
                "roi": roi_by_id.get(snapshot.seat_id, [0, 0, 0, 0]),
                "state": state,
                "state_cn": STATE_LABELS_CN.get(state, state),
                "person_count": snapshot.person_count,
                "object_count": snapshot.object_count,
                "leave_minutes": snapshot.leave_minutes,
                "violation": snapshot.violation,
            }
        )

    return seats, counts


def _event_duration_minutes(event) -> float:
    """事件持续时间 = 结束时刻 - 触发时刻。

    状态机给的 event.duration_minutes 是「该座位本次离座总时长」，
    其中包含触发阈值之前的那段缓冲（例如离座 55 分钟才收尾，其中前 30 分钟只是计时）。
    看板上「持续时长」一列若直接用总时长，会和「开始/结束时间」对不上，
    因此这里按时间戳重算，另把总时长单独作为 leave_minutes_total 返回。
    """
    if not event.end_ts:
        return 0.0
    return round((event.end_ts - event.start_ts).total_seconds() / 60.0, 1)


def _find_event_index(job: VideoJob, seat_id: str) -> int:
    """找到该座位仍未收尾的事件下标，没有则返回 -1。"""
    for index, event in enumerate(job.events):
        if event["seat_id"] == seat_id and not event["end"]:
            return index
    return -1


def run_job(job: VideoJob, config: dict, weights: str) -> None:
    """处理线程主体：读帧 → 检测 → 判定 → 状态机 → 发布进度。"""
    started_at = time.perf_counter()
    capture = None

    try:
        capture = cv2.VideoCapture(job.video_path)
        if not capture.isOpened():
            job.publish(status="error", message="视频无法打开，可能是编码不受支持（建议转成 H.264 的 mp4）")
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if total_frames > 0 else 0.0

        # 按「每秒 sample_interval 帧」换算步长；至少 1 帧
        stride = max(1, int(round(fps * job.sample_interval)))

        seat_config = config.get("video_seats") or config["seats"]
        seat_ids = [seat["id"] for seat in seat_config]
        occupancy = config["occupancy"]
        tracker = SeatTracker(
            seat_ids,
            leave_minutes=occupancy["object_leave_minutes"],
            confirm_frames=occupancy["empty_confirm_frames"],
            exempt_config=config.get("exempt_time"),
        )

        # 视频第 0 秒映射到当前整点（秒归零），保证时间戳好看且豁免窗口判断正常
        clock_start = datetime.now().replace(second=0, microsecond=0)

        job.publish(
            status="running",
            frames_total=total_frames,
            video_duration=duration,
            message=f"共 {total_frames} 帧 / {duration:.1f} 秒，采样步长 {stride} 帧",
        )

        frame_index = 0
        last_clock = clock_start

        while True:
            if job.cancel_requested:
                job.publish(status="cancelled", message="已取消")
                break

            success, frame = capture.read()
            if not success:
                break

            if frame_index % stride == 0:
                video_seconds = frame_index / fps
                # 视频时间 → 状态机时间
                clock = clock_start + timedelta(seconds=video_seconds * job.time_scale)
                last_clock = clock

                with INFERENCE_LOCK:
                    detections, canvas = detect_image(
                        frame,
                        weights,
                        config["model"].get("video_conf", config["model"]["conf"]),
                        config["model"]["imgsz"],
                        config["model"]["device"],
                    )

                seat_results = judge_seat_states(detections, seat_config, occupancy)
                raw_states = {item["id"]: item["state"] for item in seat_results}
                details = {
                    item["id"]: {
                        "person_count": item["person_count"],
                        "object_count": item["object_count"],
                    }
                    for item in seat_results
                }

                update = tracker.update(raw_states, clock, details)
                seats, counts = _build_seats(update.snapshots, seat_config)

                # 状态切换写进时间线
                for change in update.changes:
                    job.push_timeline(
                        {
                            "video_seconds": round(video_seconds, 1),
                            "clock": clock.strftime("%H:%M:%S"),
                            "seat_id": change.seat_id,
                            "from": change.old_state,
                            "to": change.new_state,
                            "from_cn": STATE_LABELS_CN.get(change.old_state, change.old_state),
                            "to_cn": STATE_LABELS_CN.get(change.new_state, change.new_state),
                        }
                    )

                # 开出的占座事件
                for event in update.started_events:
                    job.events.append(
                        {
                            "seat_id": event.seat_id,
                            "start": clock.strftime("%H:%M:%S"),
                            "end": "",
                            "duration_minutes": 0.0,
                            "leave_minutes_total": 0.0,
                            "threshold_minutes": event.threshold_minutes,
                            "outcome": OUTCOME_ONGOING,
                            "outcome_cn": OUTCOME_LABELS_CN[OUTCOME_ONGOING],
                        }
                    )

                # 收尾的占座事件
                for event in update.closed_events:
                    index = _find_event_index(job, event.seat_id)
                    if index >= 0:
                        job.events[index].update(
                            {
                                "end": clock.strftime("%H:%M:%S"),
                                "duration_minutes": _event_duration_minutes(event),
                                "leave_minutes_total": event.duration_minutes,
                                "outcome": event.outcome,
                                "outcome_cn": OUTCOME_LABELS_CN.get(event.outcome, event.outcome),
                            }
                        )

                # 标注图：draw_result 内部会 copy，不会改到 canvas 共享内存
                annotated = draw_result(canvas, seat_config, seat_results, detections)
                encoded_ok, buffer = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )

                job.publish(
                    seats=seats,
                    counts=counts,
                    detections=len(detections),
                    video_seconds=video_seconds,
                    simulated_seconds=video_seconds * job.time_scale,
                    frame_seq=job.frame_seq + 1,
                    frame_jpeg=buffer.tobytes() if encoded_ok else job.frame_jpeg,
                    progress=(frame_index / total_frames) if total_frames > 0 else 0.0,
                    frames_done=job.frames_done + 1,
                    elapsed_seconds=time.perf_counter() - started_at,
                )

            frame_index += 1

        if job.status == "running":
            # 正常读完：未结事件按 ONGOING 收尾（占座仍在持续）
            for event in tracker.close_all(last_clock, OUTCOME_ONGOING):
                index = _find_event_index(job, event.seat_id)
                if index >= 0:
                    job.events[index].update(
                        {
                            "end": last_clock.strftime("%H:%M:%S"),
                            "duration_minutes": _event_duration_minutes(event),
                            "leave_minutes_total": event.duration_minutes,
                            "outcome": event.outcome,
                            "outcome_cn": OUTCOME_LABELS_CN.get(event.outcome, event.outcome),
                        }
                    )

            job.publish(
                status="done",
                progress=1.0,
                elapsed_seconds=time.perf_counter() - started_at,
                message=f"处理完成：{job.frames_done} 帧，检出 {len(job.events)} 起占座",
            )

    except Exception as error:  # noqa: BLE001 - 任务线程要把异常转成状态，不能让线程静默死掉
        job.publish(status="error", message=f"处理失败：{type(error).__name__}: {error}")
    finally:
        if capture is not None:
            capture.release()
        # 上传的视频处理完即删，避免 data/uploads 越堆越大
        try:
            if os.path.isfile(job.video_path):
                os.remove(job.video_path)
        except OSError:
            pass


def start_job(job: VideoJob, config: dict, weights: str) -> None:
    """在后台线程中启动任务。"""
    thread = threading.Thread(target=run_job, args=(job, config, weights), daemon=True)
    thread.start()
