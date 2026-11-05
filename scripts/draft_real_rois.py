"""渲染 ROI 草稿到 real_02 底图，用于人工核对与迭代。

用法：
    python scripts/draft_real_rois.py            # 只画 ROI
    python scripts/draft_real_rois.py --det     # 叠加当前模型检测框
"""

import argparse
import cv2

# 底图与尺寸
BASE = "data/raw_real/real_02.jpg"  # 1706 x 1279

# ROI 草稿：id -> (x1, y1, x2, y2)，椅子 + 椅前桌面
DRAFT = {
    "S01": (30, 620, 470, 1250),    # 左木桌·白衣人（有人）
    "S02": (130, 480, 300, 900),    # 左木桌·黑发人（有人）
    "S03": (330, 480, 480, 830),    # 左木桌·黄衣人（有人）
    "S04": (600, 420, 730, 780),    # 后桌·白衣坐者（有人）
    "S05": (750, 440, 845, 780),    # 后桌·黑衣坐者（有人）
    "S06": (845, 560, 920, 850),    # 后墙·空红椅1（空置）
    "S07": (905, 560, 995, 860),    # 后墙·空红椅2（空置）
    "S08": (1080, 520, 1260, 810),  # 右墙·遮挡白衣坐者+白纸桌（有人）
    "S09": (1175, 590, 1470, 1250), # 右墙·黄衣女（有人）
    "S10": (1420, 550, 1706, 1279), # 右远端·粉衣人（有人）
    "S11": (350, 780, 720, 1279),   # 前灰桌·空红椅（空置）
    "S12": (700, 890, 1000, 1279),  # 右下·空红椅（空置）
    "S13": (1000, 940, 1260, 1279), # 右下·椅挂白帆布包（占座）
}

COLOR_OCC = (0, 0, 255)   # 预期有人
COLOR_EMP = (0, 200, 0)   # 预期空/占


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(BASE)
    for sid, (x1, y1, x2, y2) in DRAFT.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(img, sid, (x1 + 4, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    if args.det:
        from ultralytics import YOLO
        model = YOLO("runs/train/seat_v2/weights/best.pt")
        r = model.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
        for b in r.boxes:
            cls = r.names[int(b.cls)]
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            color = (0, 0, 255) if cls == "person" else (0, 200, 200)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{cls} {float(b.conf):.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite("output/baseline_real/roi_draft.jpg", img)
    print("saved output/baseline_real/roi_draft.jpg")


if __name__ == "__main__":
    main()
