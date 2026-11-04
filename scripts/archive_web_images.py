"""网络素材归档脚本：把外部图片统一转为 JPEG 并按项目命名规范入库。

设计要点：
    1. 命名统一 —— 输出为 日期_时段_web_帧号.jpg，与抽帧脚本的命名规则保持一致；
    2. 格式统一 —— PNG / WEBP / 大尺寸 JPEG 全部转为质量 92 的 JPEG，方便后续标注与训练；
    3. 来源可溯 —— 同时生成登记表 CSV，记录「新文件名 ↔ 原文件名」的对应关系；
    4. 只复制不删除 —— 源目录文件保持原样，归档出错可随时重跑。

用法示例：
    python scripts/archive_web_images.py --src D:\\image --out data\\raw \\
        --date 20261104 --slot 0900
"""

import argparse
import csv
import os
import sys

from PIL import Image

# 支持的图片扩展名
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# 输出 JPEG 的保存质量
JPEG_QUALITY = 92


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="把外部图片归档为项目标准命名")
    parser.add_argument("--src", required=True, help="源图片目录")
    parser.add_argument("--out", required=True, help="归档输出目录，如 data\\raw")
    parser.add_argument("--date", required=True, help="素材日期，格式 YYYYMMDD")
    parser.add_argument("--slot", required=True, help="时段标记，如 0900")
    parser.add_argument("--cam", default="web", help="素材来源标识，默认 web")
    return parser.parse_args()


def list_image_files(directory: str) -> list:
    """列出源目录下所有受支持的图片（按文件名排序，保证编号稳定）。"""
    file_paths = []
    for entry_name in sorted(os.listdir(directory)):
        if entry_name.lower().endswith(SUPPORTED_EXTENSIONS):
            file_paths.append(os.path.join(directory, entry_name))
    return file_paths


def convert_to_jpeg(source_path: str, target_path: str) -> tuple:
    """把任意格式图片转为 JPEG，返回 (宽度, 高度)。"""
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        rgb_image.save(target_path, "JPEG", quality=JPEG_QUALITY)
        return rgb_image.size


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()

    if not os.path.isdir(arguments.src):
        print(f"[错误] 源目录不存在：{arguments.src}")
        return 1

    os.makedirs(arguments.out, exist_ok=True)

    image_files = list_image_files(arguments.src)
    if not image_files:
        print(f"[错误] 源目录下没有受支持的图片：{arguments.src}")
        return 1

    print(f"[信息] 共发现 {len(image_files)} 个图片文件，开始归档")

    manifest_rows = []
    for index, source_path in enumerate(image_files, start=1):
        target_name = f"{arguments.date}_{arguments.slot}_{arguments.cam}_f{index:04d}.jpg"
        target_path = os.path.join(arguments.out, target_name)

        try:
            width, height = convert_to_jpeg(source_path, target_path)
        except Exception as error:
            print(f"[失败] {os.path.basename(source_path)}：{error}")
            continue

        manifest_rows.append(
            {
                "new_name": target_name,
                "source_name": os.path.basename(source_path),
                "width": width,
                "height": height,
            }
        )
        print(f"[归档] {target_name}  <-  {os.path.basename(source_path)}  ({width}×{height})")

    manifest_path = os.path.join(arguments.out, "web_images_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["new_name", "source_name", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[完成] 归档 {len(manifest_rows)} 张，输出目录：{os.path.abspath(arguments.out)}")
    print(f"[完成] 来源登记表：{os.path.abspath(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
