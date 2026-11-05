"""用一张真实场景图生成「监控风格」演示视频，用于验证上传视频的识别链路。

为什么需要它：
    项目里原有的 sim_cam1.mp4 是程序绘制的几何示意图，画面里的桌椅位置与
    真实照片标定的 ROI 完全对不上，跑出来检测为 0，无法验证真实链路。
    本脚本以一张真实机位图为底，叠加监控视频应有的细节（亮度波动、轻微抖动、
    时间戳水印），既保证画面能被模型识别，也让演示看起来像真实录像。

用法：
    python scripts/make_demo_video_from_image.py \
        --image data/video_demo/main_view.jpg \
        --out data/video_demo/demo_scene.mp4 \
        --seconds 60 --fps 10
"""

import argparse
import os

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由真实场景图生成监控风格演示视频")
    parser.add_argument("--image", required=True, help="底图路径")
    parser.add_argument("--out", required=True, help="输出视频路径")
    parser.add_argument("--seconds", type=float, default=60.0, help="视频时长（秒），默认 60")
    parser.add_argument("--fps", type=int, default=10, help="帧率，默认 10（静止画面无需高帧率）")
    parser.add_argument("--start-hour", type=int, default=9, help="时间戳起始小时，默认 9")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    base = cv2.imread(arguments.image)
    if base is None:
        print(f"[错误] 底图读取失败：{arguments.image}")
        return 1

    height, width = base.shape[:2]
    # mp4 编码要求宽高为偶数
    if width % 2 or height % 2:
        width -= width % 2
        height -= height % 2
        base = base[:height, :width]

    os.makedirs(os.path.dirname(os.path.abspath(arguments.out)), exist_ok=True)
    writer = cv2.VideoWriter(
        arguments.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        arguments.fps,
        (width, height),
    )
    if not writer.isOpened():
        print("[错误] 视频写入器打开失败，检查输出路径与编码器")
        return 1

    total_frames = int(arguments.seconds * arguments.fps)
    # 预生成几张抖动矩阵，避免每帧重复构造
    shifts = [(0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1)]

    for index in range(total_frames):
        frame = base.copy()

        # 摄像机轻微抖动：整像素平移，模拟支架微震
        shift_x, shift_y = shifts[index % len(shifts)]
        if shift_x or shift_y:
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            frame = cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

        # 灯光缓慢波动 + 轻微传感器噪声
        brightness = 3.0 * np.sin(index / 25.0)
        frame = np.clip(frame.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

        noise = np.random.default_rng(index).normal(0, 1.8, (height, width, 3))
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

        # 左上角时间戳：秒数随帧推进，便于核对播放进度
        elapsed_seconds = index / arguments.fps
        minute_total = int(arguments.start_hour * 60 + elapsed_seconds // 60)
        second = int(elapsed_seconds % 60)
        stamp = f"CAM-01  {minute_total // 60:02d}:{minute_total % 60:02d}:{second:02d}"
        cv2.rectangle(frame, (16, 16), (330, 58), (32, 32, 32), -1)
        cv2.putText(frame, stamp, (26, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

        writer.write(frame)

        if index % 50 == 0:
            print(f"  已写 {index}/{total_frames} 帧", flush=True)

    writer.release()

    size_mb = os.path.getsize(arguments.out) / 1024 / 1024
    print(f"完成：{arguments.out}")
    print(f"  时长 {arguments.seconds:.0f} 秒 / {arguments.fps} fps / {total_frames} 帧 / {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
