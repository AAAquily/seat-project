"""SQLite 存储层：座位基础信息、逐帧观测记录、占座事件。

三张表的分工：
    seats            座位静态信息（编号、名称、ROI 坐标），启动时按配置覆盖更新
    observations     逐帧观测流水，用于回放与统计某座位一天的状态变化
    occupancy_events 占座事件，一次「有物无人满 30 分钟」对应一条，含结束结局

时间统一以 ISO 8601 字符串存储（如 2026-11-05T09:00:00），便于 SQL 排序与按日筛选。
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Optional

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS seats (
        id      TEXT PRIMARY KEY,
        name    TEXT,
        x1      INTEGER,
        y1      INTEGER,
        x2      INTEGER,
        y2      INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts           TEXT    NOT NULL,
        seat_id      TEXT    NOT NULL,
        state        TEXT    NOT NULL,
        raw_state    TEXT,
        person_count INTEGER DEFAULT 0,
        object_count INTEGER DEFAULT 0,
        leave_minutes REAL   DEFAULT 0,
        violation    INTEGER DEFAULT 0,
        UNIQUE (ts, seat_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS occupancy_events (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        seat_id        TEXT    NOT NULL,
        start_ts       TEXT    NOT NULL,
        end_ts         TEXT,
        duration_minutes REAL  DEFAULT 0,
        outcome        TEXT,
        threshold_minutes REAL DEFAULT 30
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_observations_seat_ts ON observations (seat_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_seat ON occupancy_events (seat_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_start ON occupancy_events (start_ts)",
]


def connect(db_path: str) -> sqlite3.Connection:
    """打开（不存在则创建）数据库，并确保表结构就绪。"""
    directory = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(directory, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
    connection.commit()
    return connection


def upsert_seats(connection: sqlite3.Connection, seats: List[dict]) -> None:
    """把配置里的座位信息写入数据库；已存在则更新名称与坐标。"""
    for seat in seats:
        roi = seat["roi"]
        connection.execute(
            """
            INSERT INTO seats (id, name, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                x1 = excluded.x1,
                y1 = excluded.y1,
                x2 = excluded.x2,
                y2 = excluded.y2
            """,
            (seat["id"], seat.get("name", seat["id"]), roi[0], roi[1], roi[2], roi[3]),
        )
    connection.commit()


def insert_observation(
    connection: sqlite3.Connection,
    moment: datetime,
    seat_id: str,
    state: str,
    raw_state: str,
    person_count: int,
    object_count: int,
    leave_minutes: float,
    violation: bool,
) -> None:
    """写入一条逐帧观测；同一时刻同一座位重复写入时忽略（幂等）。"""
    connection.execute(
        """
        INSERT OR IGNORE INTO observations
            (ts, seat_id, state, raw_state, person_count, object_count, leave_minutes, violation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            moment.isoformat(timespec="seconds"),
            seat_id,
            state,
            raw_state,
            person_count,
            object_count,
            leave_minutes,
            1 if violation else 0,
        ),
    )


def open_event(connection: sqlite3.Connection, seat_id: str, start: datetime, threshold_minutes: float) -> int:
    """新建一条进行中的占座事件，返回自增主键。"""
    cursor = connection.execute(
        """
        INSERT INTO occupancy_events (seat_id, start_ts, threshold_minutes)
        VALUES (?, ?, ?)
        """,
        (seat_id, start.isoformat(timespec="seconds"), threshold_minutes),
    )
    connection.commit()
    return int(cursor.lastrowid)


def close_event(
    connection: sqlite3.Connection,
    event_id: int,
    end: datetime,
    duration_minutes: float,
    outcome: str,
) -> None:
    """结束一条占座事件：写入结束时间、持续时长与结局。"""
    connection.execute(
        """
        UPDATE occupancy_events
           SET end_ts = ?, duration_minutes = ?, outcome = ?
         WHERE id = ?
        """,
        (end.isoformat(timespec="seconds"), round(duration_minutes, 1), outcome, event_id),
    )
    connection.commit()


def close_dangling_events(connection: sqlite3.Connection) -> int:
    """把上次运行遗留的未结事件标记为中断，返回处理条数。

    程序被强制关闭时事件来不及收尾，若不清理，下次统计会把它们当成仍在进行的长占座。
    """
    cursor = connection.execute(
        """
        UPDATE occupancy_events
           SET end_ts = COALESCE(end_ts, start_ts),
               outcome = COALESCE(NULLIF(outcome, ''), 'INTERRUPTED')
         WHERE end_ts IS NULL
        """
    )
    connection.commit()
    return cursor.rowcount


def fetch_events(connection: sqlite3.Connection, limit: int = 50, seat_id: Optional[str] = None) -> List[sqlite3.Row]:
    """按开始时间倒序查询占座事件，可按座位过滤。"""
    if seat_id:
        cursor = connection.execute(
            "SELECT * FROM occupancy_events WHERE seat_id = ? ORDER BY start_ts DESC LIMIT ?",
            (seat_id, limit),
        )
    else:
        cursor = connection.execute(
            "SELECT * FROM occupancy_events ORDER BY start_ts DESC LIMIT ?",
            (limit,),
        )
    return cursor.fetchall()


def summarize_day(connection: sqlite3.Connection, day_prefix: str) -> List[sqlite3.Row]:
    """按座位统计某天（YYYY-MM-DD）的观测帧与占座次数，用于看板趋势图。"""
    cursor = connection.execute(
        """
        SELECT o.seat_id,
               COUNT(*)                                             AS frames,
               SUM(CASE WHEN o.state = 'OCCUPIED'    THEN 1 ELSE 0 END) AS occupied_frames,
               SUM(CASE WHEN o.state = 'OBJECT_ONLY' THEN 1 ELSE 0 END) AS object_frames,
               SUM(CASE WHEN o.state = 'EMPTY'       THEN 1 ELSE 0 END) AS empty_frames,
               (SELECT COUNT(*) FROM occupancy_events e
                 WHERE e.seat_id = o.seat_id AND e.start_ts LIKE ?) AS violation_count
          FROM observations o
         WHERE o.ts LIKE ?
         GROUP BY o.seat_id
         ORDER BY o.seat_id
        """,
        (day_prefix + "%", day_prefix + "%"),
    )
    return cursor.fetchall()
