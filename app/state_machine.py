"""座位状态时序机：把逐帧的原始判定收敛成稳定状态，并产出占座事件。

它解决三个逐帧判定解决不了的问题：
    1. 防抖 —— 单帧误检（例如椅子被认成人）不应改变座位状态，需连续 N 帧一致才确认；
    2. 计时 —— 「有物无人」必须持续满 30 分钟才算占座，中途人回来或物品被清走即清零；
    3. 豁免 —— 午休、闭馆整理等时段的时长不计入占座计时。

状态的字符串取值与 scripts/judge_seats.py 保持一致，作为模块间的数据契约：
    "OCCUPIED" 有人 / "OBJECT_ONLY" 有物无人 / "EMPTY" 空置
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

STATE_OCCUPIED = "OCCUPIED"
STATE_OBJECT_ONLY = "OBJECT_ONLY"
STATE_EMPTY = "EMPTY"

# 事件结局：占座事件结束时记录，便于统计「真的走了」还是「只是去接水」
OUTCOME_RESOLVED = "RESOLVED"      # 占座期间本人回来
OUTCOME_CLEARED = "CLEARED"        # 占座期间物品被清走（座位变空）
OUTCOME_ONGOING = "ONGOING"        # 数据流正常结束，事件仍持续（下次运行可续算）
OUTCOME_INTERRUPTED = "INTERRUPTED"  # 程序异常中断，事件未正常收尾

# 星期缩写映射，与 config.yaml 中 days: [mon, tue, ...] 对应
WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass
class SeatSnapshot:
    """某一时刻单个座位的完整状态，供看板与数据库使用。"""

    seat_id: str
    state: str
    raw_state: str
    person_count: int
    object_count: int
    leave_minutes: float
    violation: bool


@dataclass
class StateChange:
    """一次被确认的状态切换。"""

    seat_id: str
    old_state: str
    new_state: str
    moment: datetime


@dataclass
class OccupancyEvent:
    """一条占座事件：从「有物无人满 30 分钟」开始，到人回来或物品被清走结束。"""

    seat_id: str
    start_ts: datetime
    threshold_minutes: float
    end_ts: Optional[datetime] = None
    outcome: str = ""
    duration_minutes: float = 0.0
    event_id: Optional[int] = None

    def close(self, moment: datetime, outcome: str, duration_minutes: float) -> None:
        """收尾事件：写入结束时间、结局与实际持续时长。"""
        self.end_ts = moment
        self.outcome = outcome
        self.duration_minutes = round(duration_minutes, 1)


@dataclass
class TrackerUpdate:
    """一次 update 的产出，驱动层据此写库与打印。"""

    snapshots: List[SeatSnapshot] = field(default_factory=list)
    changes: List[StateChange] = field(default_factory=list)
    started_events: List[OccupancyEvent] = field(default_factory=list)
    closed_events: List[OccupancyEvent] = field(default_factory=list)


def parse_hhmm(text: str) -> int:
    """把 "12:30" 解析成当天第几分钟，便于区间比较。"""
    hour_text, minute_text = text.split(":")
    return int(hour_text) * 60 + int(minute_text)


class ExemptSchedule:
    """豁免时段判断：命中窗口内的时间不计入占座时长。"""

    def __init__(self, exempt_config: Optional[dict]):
        self.enabled = bool(exempt_config and exempt_config.get("enabled"))
        self.windows = []

        if not self.enabled:
            return

        for window in exempt_config.get("windows", []):
            start_minute = parse_hhmm(window["start"])
            end_minute = parse_hhmm(window["end"])
            days = [day.lower() for day in window.get("days", WEEKDAY_KEYS)]
            self.windows.append(
                {
                    "name": window.get("name", "豁免时段"),
                    "start": start_minute,
                    "end": end_minute,
                    "days": set(days),
                }
            )

    def is_exempt(self, moment: datetime) -> bool:
        """判断某时刻是否处于豁免窗口内。"""
        if not self.enabled:
            return False

        weekday = WEEKDAY_KEYS[moment.weekday()]
        minute_of_day = moment.hour * 60 + moment.minute

        for window in self.windows:
            if weekday not in window["days"]:
                continue
            if window["start"] <= minute_of_day < window["end"]:
                return True
        return False


class SeatTracker:
    """逐座位维护防抖状态、离座累计时长与占座事件。"""

    def __init__(
        self,
        seat_ids: List[str],
        leave_minutes: float,
        confirm_frames: int,
        exempt_config: Optional[dict] = None,
    ):
        if confirm_frames < 1:
            confirm_frames = 1

        self.seat_ids = list(seat_ids)
        self.leave_minutes = float(leave_minutes)
        self.confirm_frames = int(confirm_frames)
        self.exempt = ExemptSchedule(exempt_config)

        # 每个座位的运行态
        self.confirmed_state: Dict[str, str] = {seat_id: STATE_EMPTY for seat_id in self.seat_ids}
        self.raw_state: Dict[str, str] = {seat_id: STATE_EMPTY for seat_id in self.seat_ids}
        self.pending_state: Dict[str, Optional[str]] = {seat_id: None for seat_id in self.seat_ids}
        self.pending_count: Dict[str, int] = {seat_id: 0 for seat_id in self.seat_ids}
        self.leave_seconds: Dict[str, float] = {seat_id: 0.0 for seat_id in self.seat_ids}
        self.violation: Dict[str, bool] = {seat_id: False for seat_id in self.seat_ids}
        self.open_event: Dict[str, Optional[OccupancyEvent]] = {seat_id: None for seat_id in self.seat_ids}
        self.last_tick: Optional[datetime] = None

    def update(self, raw_states: Dict[str, str], now: datetime, details: Optional[Dict[str, dict]] = None) -> TrackerUpdate:
        """推进一帧：raw_states 为 {seat_id: 原始状态}，now 为当前（可为虚拟）时间。

        details 可选，形如 {seat_id: {"person_count": 1, "object_count": 2}}，
        仅用于把检测计数带进快照，不参与状态判定。
        """
        details = details or {}

        delta_seconds = 0.0
        if self.last_tick is not None:
            delta_seconds = max(0.0, (now - self.last_tick).total_seconds())
        self.last_tick = now

        update = TrackerUpdate()
        # 豁免时段内的时间整体跳过，避免把午休两小时算成占座
        countable_seconds = 0.0 if self.exempt.is_exempt(now) else delta_seconds

        for seat_id in self.seat_ids:
            raw = raw_states.get(seat_id, self.confirmed_state[seat_id])
            self.raw_state[seat_id] = raw
            self._advance_debounce(seat_id, raw, now, update)
            self._advance_timer(seat_id, now, countable_seconds, update)
            update.snapshots.append(self._snapshot(seat_id, details.get(seat_id, {})))

        return update

    def _advance_debounce(self, seat_id: str, raw: str, now: datetime, update: TrackerUpdate) -> None:
        """防抖：连续 confirm_frames 帧与当前确认状态不同，才切换。"""
        if raw == self.confirmed_state[seat_id]:
            self.pending_state[seat_id] = None
            self.pending_count[seat_id] = 0
            return

        if self.pending_state[seat_id] == raw:
            self.pending_count[seat_id] += 1
        else:
            self.pending_state[seat_id] = raw
            self.pending_count[seat_id] = 1

        if self.pending_count[seat_id] < self.confirm_frames:
            return

        old_state = self.confirmed_state[seat_id]
        self.confirmed_state[seat_id] = raw
        self.pending_state[seat_id] = None
        self.pending_count[seat_id] = 0
        update.changes.append(StateChange(seat_id, old_state, raw, now))
        self._on_state_changed(seat_id, old_state, raw, now, update)

    def _on_state_changed(self, seat_id: str, old_state: str, new_state: str, now: datetime, update: TrackerUpdate) -> None:
        """状态切换的副作用：离开「有物无人」时结案，进入时重置计时。"""
        if old_state != STATE_OBJECT_ONLY:
            if new_state == STATE_OBJECT_ONLY:
                self.leave_seconds[seat_id] = 0.0
                self.violation[seat_id] = False
            return

        accumulated_minutes = self.leave_seconds[seat_id] / 60.0
        event = self.open_event[seat_id]

        if event is not None:
            outcome = OUTCOME_RESOLVED if new_state == STATE_OCCUPIED else OUTCOME_CLEARED
            event.close(now, outcome, accumulated_minutes)
            update.closed_events.append(event)
            self.open_event[seat_id] = None

        # 离开「有物无人」后，无论是否曾构成占座，计时都归零
        self.leave_seconds[seat_id] = 0.0
        self.violation[seat_id] = False

    def _advance_timer(self, seat_id: str, now: datetime, countable_seconds: float, update: TrackerUpdate) -> None:
        """累加离座时长，跨过阈值时开出占座事件。"""
        if self.confirmed_state[seat_id] != STATE_OBJECT_ONLY:
            return

        self.leave_seconds[seat_id] += countable_seconds

        if self.violation[seat_id]:
            return

        if self.leave_seconds[seat_id] < self.leave_minutes * 60.0:
            return

        self.violation[seat_id] = True
        event = OccupancyEvent(
            seat_id=seat_id,
            start_ts=now,
            threshold_minutes=self.leave_minutes,
        )
        self.open_event[seat_id] = event
        update.started_events.append(event)

    def _snapshot(self, seat_id: str, detail: Optional[dict] = None) -> SeatSnapshot:
        """生成对外快照，检测计数来自驱动层传入的细节。"""
        detail = detail or {}
        return SeatSnapshot(
            seat_id=seat_id,
            state=self.confirmed_state[seat_id],
            raw_state=self.raw_state[seat_id],
            person_count=int(detail.get("person_count", 0)),
            object_count=int(detail.get("object_count", 0)),
            leave_minutes=round(self.leave_seconds[seat_id] / 60.0, 1),
            violation=self.violation[seat_id],
        )

    def remaining_minutes(self, seat_id: str) -> float:
        """距离触发占座还剩多少分钟，供看板显示倒计时。"""
        remaining = self.leave_minutes - self.leave_seconds[seat_id] / 60.0
        return round(max(0.0, remaining), 1)

    def close_all(self, now: datetime, outcome: str = OUTCOME_INTERRUPTED) -> List[OccupancyEvent]:
        """数据流结束时收尾所有未结事件。

        正常读完数据流传 OUTCOME_ONGOING（事件仍在持续），程序异常退出保留默认的
        OUTCOME_INTERRUPTED，便于事后区分「没跑完」和「占座未结束」。
        """
        closed = []
        for seat_id in self.seat_ids:
            event = self.open_event[seat_id]
            if event is None:
                continue

            event.close(now, outcome, self.leave_seconds[seat_id] / 60.0)
            closed.append(event)
            self.open_event[seat_id] = None
        return closed
