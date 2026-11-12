"""对比多个权重版本在 9 张真实图上的座位级判定结果。

用法：
    python scripts/compare_versions.py output/real9_refix output/real9_v9 output/real9_v9b

真值口径来自 docs/真实9图按标注标准调试报告.md 第四节（人工标注）。
座位级一致率用「各状态计数之差 / 2」估算误判座位数——因为真值只到
每张图的「有人 / 有物无人 / 空置」个数，不逐座列 id。
"""

import json
import os
import sys

# 人工标注真值：(有人, 有物无人, 空置)
TRUTH = {
    "real_01": (6, 0, 7),
    "real_02": (9, 3, 1),
    "real_03": (6, 2, 3),
    "real_04": (7, 0, 1),
    "real_05": (9, 1, 3),
    "real_06": (8, 2, 4),
    "real_07": (7, 2, 5),
    "real_08": (4, 3, 3),
    "real_09": (9, 0, 1),
}


def load(result_dir):
    """读入某结果目录下 9 张图的 json，返回 {图片名: (有人, 有物, 空置, 座位总数)}。"""
    table = {}
    for name in TRUTH:
        path = os.path.join(result_dir, name + ".json")
        if not os.path.isfile(path):
            table[name] = None
            continue
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        totals = data["totals"]
        table[name] = (
            totals["OCCUPIED"],
            totals["OBJECT_ONLY"],
            totals["EMPTY"],
            totals["seats"],
        )
    return table


def score(table):
    """统计误判座位数与一致率。"""
    wrong = 0
    seats = 0
    detail = {}
    for name, truth in TRUTH.items():
        row = table.get(name)
        if row is None:
            detail[name] = None
            continue
        got = row[:3]
        seats += row[3]
        diff = sum(abs(a - b) for a, b in zip(got, truth)) // 2
        wrong += diff
        detail[name] = diff
    rate = (seats - wrong) / seats if seats else 0.0
    return wrong, seats, rate, detail


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("用法: python scripts/compare_versions.py <结果目录> [结果目录 ...]")
        return 1

    tables = {d: load(d) for d in dirs}
    header = f"{'图':<9}" + "".join(f"{os.path.basename(d.rstrip('/')):>22}" for d in dirs)
    print(header)
    print("-" * len(header))
    for name in TRUTH:
        cells = []
        for d in dirs:
            row = tables[d].get(name)
            cells.append("—" if row is None else f"{row[0]}/{row[1]}/{row[2]}")
        line = f"{name:<9}" + "".join(f"{c:>22}" for c in cells)
        print(line)

    print("-" * len(header))
    print(f"{'真值':<9}" + "".join(f"{'%d/%d/%d' % TRUTH[n]:>22}" for n in TRUTH))
    for d in dirs:
        wrong, seats, rate, detail = score(tables[d])
        bad = {k: v for k, v in detail.items() if v}
        print(f"[{os.path.basename(d.rstrip('/'))}] 误判 {wrong} 座 / {seats} = 一致率 {rate*100:.1f}%  明细 {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
