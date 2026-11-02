# seat-project 自习室座位占用识别系统

高校课程项目：用摄像头 + YOLOv8 自动识别自习室每个座位的状态，分为**有人、有物无人、空置**三类，
并在「有物无人」持续超过设定时长时判定为**占座**，通过网页看板实时展示。

## 一、快速开始

```bash
# 1. 激活环境（每次开工先执行）
conda activate seat

# 2. 进入项目目录
cd /d D:\seat-project

# 3. 用自带的官方模型跑一次检测，确认环境可用
yolo predict model=models/yolov8n.pt source=data\raw\test.jpg
```

## 二、目录结构

```
seat-project/
├── app/              系统代码（识别服务 + 网页看板）
│   └── static/       前端页面、CSS、JS
├── config/
│   └── config.yaml   主配置：座位 ROI、判定阈值、豁免时段
├── data/             数据（不进 Git）
│   ├── raw/          原始采集图片
│   ├── labels/       YOLO 格式标注
│   └── images/       划分后的 train/val/test
├── models/           模型权重（yolov8n.pt 等，不进 Git）
├── scripts/          数据划分、预标注、评估等脚本
├── tests/            测试脚本与抽检记录
├── docs/             文档
│   └── evidence/     过程留证（截图、指标、命令输出）
├── logs/             运行日志（不进 Git）
├── runs/             YOLO 训练/推理输出（不进 Git）
├── requirements.txt  Python 依赖清单
├── start.bat         一键启动脚本（Windows）
└── README.md         本文件
```

## 三、座位状态判定规则

| 状态 | 判定条件 |
|---|---|
| 有人 | 座位 ROI 内检出人体框，且重叠率达到阈值 |
| 有物无人 | 无人体框，但检出书本/背包/水杯等物品 |
| 空置 | 既无人体也无物品 |

**占座**：状态为「有物无人」且持续超过 `occupancy.object_leave_minutes`（默认 30 分钟）。
**豁免时段**：`exempt_time.windows` 中配置的时段（如午休）内不判定占座。

## 四、开发约定

- 所有命令在 **Anaconda Prompt** 的 `(seat)` 环境下执行
- 数据集与模型权重**不进 Git**，通过移动硬盘 / 网盘同步
- 每次提交前先 `git pull`，提交说明写清「做了什么」
- 过程证据统一放 `docs/evidence/`，答辩材料从这里取材

## 五、当前进度

- [x] 环境搭建（Anaconda + YOLOv8 + LabelImg + FastAPI）
- [x] 项目骨架（K01）
- [ ] 数据采集与标注
- [ ] 模型训练与评估
- [ ] 后端服务与判定逻辑
- [ ] 网页看板
- [ ] 测试与结题材料
