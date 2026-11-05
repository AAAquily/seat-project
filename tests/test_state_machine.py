"""状态机单元测试：不依赖模型与数据库，纯逻辑校验。

覆盖四条核心规则：
    1. 防抖 —— 连续 confirm_frames 帧一致才确认状态
    2. 阈值 —— 离座累计满 object_leave_minutes 才触发占座事件
    3. 结案 —— 人回来记 RESOLVED，物品被清走记 CLEARED
    4. 豁免 —— 豁免时段内的时间不计入离座累计

运行：python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.state_machine import (  # noqa: E402
    OUTCOME_CLEARED,
    OUTCOME_RESOLVED,
    STATE_EMPTY,
    STATE_OBJECT_ONLY,
    STATE_OCCUPIED,
    OccupancyEvent,
    SeatTracker,
    StateChange,
    TrackerUpdate,
)

START = datetime(2026, 11, 5, 10, 0, 0)


@dataclass
class FeedResult:
    """一段连续帧的聚合结果：事件可能发生在区间中任意一帧，必须全程收集。"""

    started: List[OccupancyEvent] = field(default_factory=list)
    closed: List[OccupancyEvent] = field(default_factory=list)
    changes: List[StateChange] = field(default_factory=list)
    last: TrackerUpdate = None


def build_tracker(leave_minutes: float = 30, confirm_frames: int = 3, exempt_config: dict = None) -> SeatTracker:
    """构造只含 A01 一个座位的状态机，简化断言。"""
    return SeatTracker(
        seat_ids=["A01"],
        leave_minutes=leave_minutes,
        confirm_frames=confirm_frames,
        exempt_config=exempt_config,
    )


def feed(tracker: SeatTracker, state: str, minutes: int, start: datetime) -> FeedResult:
    """以每分钟一帧的节奏连续喂入同一状态，返回整段区间的聚合结果。"""
    result = FeedResult()

    for offset in range(minutes):
        update = tracker.update({"A01": state}, start + timedelta(minutes=offset))
        result.started.extend(update.started_events)
        result.closed.extend(update.closed_events)
        result.changes.extend(update.changes)
        result.last = update

    return result


class DebounceTest(unittest.TestCase):
    """防抖规则。"""

    def test_single_frame_does_not_change_state(self):
        tracker = build_tracker(confirm_frames=3)
        update = tracker.update({"A01": STATE_OCCUPIED}, START)
        self.assertEqual(update.snapshots[0].state, STATE_EMPTY)
        self.assertEqual(len(update.changes), 0)

    def test_three_consecutive_frames_confirm_state(self):
        tracker = build_tracker(confirm_frames=3)
        feed(tracker, STATE_OCCUPIED, 3, START)
        self.assertEqual(tracker.confirmed_state["A01"], STATE_OCCUPIED)

    def test_interrupted_frames_reset_counter(self):
        tracker = build_tracker(confirm_frames=3)
        tracker.update({"A01": STATE_OCCUPIED}, START)
        tracker.update({"A01": STATE_OBJECT_ONLY}, START + timedelta(minutes=1))
        tracker.update({"A01": STATE_OCCUPIED}, START + timedelta(minutes=2))
        # 三次不同状态交替出现，计数被打断，不应确认任何状态
        self.assertEqual(tracker.confirmed_state["A01"], STATE_EMPTY)


class ThresholdTest(unittest.TestCase):
    """离座累计与阈值。"""

    def test_no_event_below_threshold(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        result = feed(tracker, STATE_OBJECT_ONLY, 25, START)
        self.assertEqual(len(result.started), 0)

    def test_event_starts_after_threshold(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        result = feed(tracker, STATE_OBJECT_ONLY, 31, START)
        self.assertEqual(len(result.started), 1)
        self.assertEqual(result.started[0].seat_id, "A01")
        self.assertTrue(tracker.violation["A01"])

    def test_timer_counts_only_object_only_state(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        feed(tracker, STATE_OCCUPIED, 20, START)
        result = feed(tracker, STATE_OBJECT_ONLY, 10, START + timedelta(minutes=20))
        # 有人时段不计入离座累计，故此刻不应触发
        self.assertEqual(len(result.started), 0)
        self.assertFalse(tracker.violation["A01"])

    def test_person_counts_as_not_left(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        feed(tracker, STATE_OBJECT_ONLY, 20, START)
        result = feed(tracker, STATE_OCCUPIED, 1, START + timedelta(minutes=20))
        # 人回来后离开计时立刻归零
        self.assertEqual(result.last.snapshots[0].leave_minutes, 0.0)


class CloseEventTest(unittest.TestCase):
    """事件结案。"""

    def test_person_return_resolves_event(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        feed(tracker, STATE_OBJECT_ONLY, 35, START)
        result = feed(tracker, STATE_OCCUPIED, 2, START + timedelta(minutes=35))
        self.assertEqual(len(result.closed), 1)
        self.assertEqual(result.closed[0].outcome, OUTCOME_RESOLVED)
        self.assertGreaterEqual(result.closed[0].duration_minutes, 30)

    def test_cleared_closes_event(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        feed(tracker, STATE_OBJECT_ONLY, 35, START)
        result = feed(tracker, STATE_EMPTY, 2, START + timedelta(minutes=35))
        self.assertEqual(len(result.closed), 1)
        self.assertEqual(result.closed[0].outcome, OUTCOME_CLEARED)

    def test_no_close_event_without_violation(self):
        tracker = build_tracker(leave_minutes=30, confirm_frames=1)
        feed(tracker, STATE_OBJECT_ONLY, 10, START)
        result = feed(tracker, STATE_OCCUPIED, 2, START + timedelta(minutes=10))
        # 未满阈值的短暂离座不产生事件
        self.assertEqual(len(result.closed), 0)


class ExemptWindowTest(unittest.TestCase):
    """豁免时段。"""

    def build_exempt_config(self) -> dict:
        return {
            "enabled": True,
            "windows": [
                {
                    "name": "午休",
                    "start": "12:00",
                    "end": "13:30",
                    "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                }
            ],
        }

    def test_exempt_hours_not_counted(self):
        # 阈值设 100 分钟：11:00 起放置，扣掉 12:00-13:30 的 90 分钟后仅约 60 分钟，不应触发
        tracker = build_tracker(leave_minutes=100, confirm_frames=1, exempt_config=self.build_exempt_config())
        start = datetime(2026, 11, 5, 11, 0, 0)
        result = feed(tracker, STATE_OBJECT_ONLY, 150, start)
        self.assertFalse(tracker.violation["A01"])
        self.assertEqual(len(result.started), 0)

    def test_without_exempt_would_trigger(self):
        tracker = build_tracker(leave_minutes=100, confirm_frames=1, exempt_config=None)
        start = datetime(2026, 11, 5, 11, 0, 0)
        result = feed(tracker, STATE_OBJECT_ONLY, 150, start)
        self.assertTrue(tracker.violation["A01"])
        self.assertEqual(len(result.started), 1)

    def test_weekday_filter(self):
        # 只豁免周一：2026-11-05 是周四，不应生效
        config = {
            "enabled": True,
            "windows": [{"name": "仅周一", "start": "12:00", "end": "13:30", "days": ["mon"]}],
        }
        tracker = build_tracker(leave_minutes=100, confirm_frames=1, exempt_config=config)
        start = datetime(2026, 11, 5, 11, 0, 0)
        result = feed(tracker, STATE_OBJECT_ONLY, 150, start)
        self.assertEqual(len(result.started), 1)


if __name__ == "__main__":
    unittest.main()
