"""从固定机位录像中按固定时间间隔抽取图像帧，用于数据集采集。

设计要点：
    1. 按秒抽帧，而非逐帧保存 —— 相邻帧高度相似，逐帧保存只会制造重复样本；
    2. 内置相似度过滤 —— 与上一张保留帧差异过小的帧直接丢弃，减少人工筛选量；
    3. 文件名统一为 日期_时段_机位_帧号.jpg，与标注流程、数据集划分脚本的命名约定一致。

用法示例：
    python scripts/extract_frames.py --video D:\\videos\\20261104_0900_cam1.mp4 \\
        --out data\\raw --date 20261104 --slot 0900 --cam cam1 --every 5
"""

import argparse
import os
import sys

import cv2
import numpy as np

# 相似度判定时使用的缩略图边长（像素），越小判定越快，64 已足够区分场景变化
SIMILARITY_THUMB_SIZE = 64

# 灰度平均绝对差低于该阈值，视为与上一张保留帧几乎相同，予以丢弃
DEFAULT_SIMILARITY_THRESHOLD = 2.0


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从录像中按固定间隔抽帧，自动过滤重复画面"
    )
    parser.add_argument("--video", required=True, help="输入视频文件路径")
    parser.add_argument("--out", required=True, help="抽帧输出目录，如 data\\raw")
    parser.add_argument("--date", required=True, help="采集日期，格式 YYYYMMDD")
    parser.add_argument("--slot", required=True, help="时段，如 0900 / 1400 / 1900")
    parser.add_argument("--cam", default="cam1", help="机位标识，如 cam1")
    parser.add_argument("--every", type=float, default=5.0, help="抽帧间隔（秒），默认 5")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="相似度阈值，越大过滤越严格；设为 0 表示关闭过滤",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="最多保存多少张，0 表示不限")
    parser.add_argument("--quality", type=int, default=92, help="JPEG 保存质量 1—100")
    return parser.parse_args()


def build_output_path(out_dir: str, date: str, slot: str, cam: str, frame_index: int) -> str:
    """按统一规则拼接输出文件名：日期_时段_机位_帧号.jpg。"""
    file_name = f"{date}_{slot}_{cam}_f{frame_index:04d}.jpg"
    return os.path.join(out_dir, file_name)


def make_thumbnail(frame: np.ndarray) -> np.ndarray:
    """把画面转成小尺寸灰度图，用于快速比较两帧差异。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (SIMILARITY_THUMB_SIZE, SIMILARITY_THUMB_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return small.astype(np.float32)


def is_too_similar(previous_thumb: np.ndarray, current_thumb: np.ndarray, threshold: float) -> bool:
    """判断当前帧与上一张保留帧是否过于相似。"""
    if previous_thumb is None or threshold <= 0:
        return False
    difference = np.abs(current_thumb - previous_thumb)
    mean_difference = float(difference.mean())
    return mean_difference < threshold


def extract_frames(arguments: argparse.Namespace) -> int:
    """执行抽帧，返回实际保存的帧数。"""
    if not os.path.isfile(arguments.video):
        print(f"[错误] 找不到视频文件：{arguments.video}")
        return -1

    os.makedirs(arguments.out, exist_ok=True)

    capture = cv2.VideoCapture(arguments.video)
    if not capture.isOpened():
        print(f"[错误] 视频无法打开（格式可能不受支持）：{arguments.video}")
        return -1

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        print("[错误] 读取视频帧率失败，文件可能已损坏")
        capture.release()
        return -1

    frame_step = max(1, int(round(fps * arguments.every)))
    duration_sec = total_frames / fps if total_frames > 0 else 0
    print(f"[信息] 视频帧率 {fps:.1f} fps，总帧数 {total_frames}，时长约 {duration_sec:.1f} 秒")
    print(f"[信息] 抽帧间隔 {arguments.every} 秒，相当于每 {frame_step} 帧取 1 张")

    saved_count = 0
    skipped_count = 0
    frame_index = 0
    previous_thumb = None

    while True:
        success, frame = capture.read()
        if not success:
            break

        is_sample_point = frame_index % frame_step == 0
        if is_sample_point:
            current_thumb = make_thumbnail(frame)
            if is_too_similar(previous_thumb, current_thumb, arguments.similarity_threshold):
                skipped_count += 1
            else:
                saved_count += 1
                output_path = build_output_path(
                    arguments.out,
                    arguments.date,
                    arguments.slot,
                    arguments.cam,
                    saved_count,
                )
                cv2.imwrite(
                    output_path,
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), arguments.quality],
                )
                previous_thumb = current_thumb

                if arguments.max_frames > 0 and saved_count >= arguments.max_frames:
                    print(f"[信息] 已达上限 {arguments.max_frames} 张，提前结束")
                    break

        frame_index += 1

    capture.release()

    print(f"[完成] 共保存 {saved_count} 张，因过于相似跳过 {skipped_count} 张")
    print(f"[完成] 输出目录：{os.path.abspath(arguments.out)}")

    if saved_count == 0:
        print("[警告] 一张都没有保存，请检查视频内容或降低 --similarity-threshold")

    return saved_count


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()
    result = extract_frames(arguments)
    if result < 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
