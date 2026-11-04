"""素材清点工具：扫描图片目录，输出尺寸/格式清单，并拼接一张缩略总览图。

用途：
    1. 拍摄或下载素材后，快速确认数量、分辨率、格式是否可用（webp / 异常尺寸要提前发现）；
    2. 生成缩略总览图，一眼看完所有候选画面，避免逐个打开浪费时间。

用法示例：
    python scripts/inventory_images.py --dir D:\\image --sheet data\\video_demo\\contact_sheet.jpg
"""

import argparse
import os
import sys

from PIL import Image, ImageDraw

# 支持的图片扩展名
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# 缩略总览图的排布参数
THUMB_WIDTH = 380
THUMB_HEIGHT = 285
COLUMNS = 5
PADDING = 8
LABEL_HEIGHT = 24


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="清点图片素材并生成缩略总览图")
    parser.add_argument("--dir", required=True, help="素材所在目录")
    parser.add_argument("--sheet", default="", help="缩略总览图输出路径，留空则不生成")
    return parser.parse_args()


def list_image_files(directory: str) -> list:
    """列出目录下所有受支持的图片文件（按文件名排序，保证结果稳定）。"""
    file_names = []
    for entry_name in sorted(os.listdir(directory)):
        if entry_name.lower().endswith(SUPPORTED_EXTENSIONS):
            file_names.append(os.path.join(directory, entry_name))
    return file_names


def inspect_image(path: str) -> dict:
    """读取单张图片的基本信息，读取失败的记为不可用。"""
    info = {"path": path, "name": os.path.basename(path), "ok": False}
    try:
        with Image.open(path) as image:
            info["width"], info["height"] = image.size
            info["format"] = image.format
            info["mode"] = image.mode
            info["ok"] = True
    except Exception as error:
        info["error"] = str(error)
    return info


def build_contact_sheet(infos: list, output_path: str) -> None:
    """把所有可用图片拼成一张缩略总览图，每格左上角标注序号与文件名。"""
    usable_infos = [info for info in infos if info["ok"]]
    if not usable_infos:
        print("[警告] 没有可用图片，跳过总览图生成")
        return

    row_count = (len(usable_infos) + COLUMNS - 1) // COLUMNS
    cell_width = THUMB_WIDTH + PADDING
    cell_height = THUMB_HEIGHT + LABEL_HEIGHT + PADDING

    sheet_width = COLUMNS * cell_width + PADDING
    sheet_height = row_count * cell_height + PADDING
    sheet = Image.new("RGB", (sheet_width, sheet_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for index, info in enumerate(usable_infos):
        grid_x = index % COLUMNS
        grid_y = index // COLUMNS
        cell_left = PADDING + grid_x * cell_width
        cell_top = PADDING + grid_y * cell_height

        with Image.open(info["path"]) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((THUMB_WIDTH, THUMB_HEIGHT), Image.LANCZOS)
            paste_x = cell_left + (THUMB_WIDTH - thumbnail.width) // 2
            paste_y = cell_top + (THUMB_HEIGHT - thumbnail.height) // 2
            sheet.paste(thumbnail, (paste_x, paste_y))

        label = f"#{index + 1:02d} {info['name'][:26]}"
        draw.text((cell_left + 2, cell_top + THUMB_HEIGHT + 4), label, fill=(30, 30, 30))

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    sheet.save(output_path, quality=88)
    print(f"[完成] 总览图已保存：{os.path.abspath(output_path)}")


def main() -> int:
    """程序入口。"""
    arguments = parse_arguments()

    if not os.path.isdir(arguments.dir):
        print(f"[错误] 目录不存在：{arguments.dir}")
        return 1

    image_files = list_image_files(arguments.dir)
    if not image_files:
        print(f"[错误] 目录下没有受支持的图片：{arguments.dir}")
        return 1

    infos = [inspect_image(path) for path in image_files]

    print(f"[信息] 共发现 {len(infos)} 个图片文件")
    print(f"{'序号':<4}{'分辨率':<14}{'格式':<8}{'状态':<8}文件名")
    print("-" * 78)

    for index, info in enumerate(infos):
        if info["ok"]:
            resolution = f"{info['width']}×{info['height']}"
            print(f"{index + 1:<6}{resolution:<14}{info['format']:<8}{'可用':<8}{info['name']}")
        else:
            print(f"{index + 1:<6}{'--':<14}{'--':<8}{'不可用':<8}{info['name']}")
            print(f"      错误：{info['error']}")

    usable_count = sum(1 for info in infos if info["ok"])
    print("-" * 78)
    print(f"[汇总] 可用 {usable_count} 张，不可用 {len(infos) - usable_count} 张")

    if arguments.sheet:
        build_contact_sheet(infos, arguments.sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
