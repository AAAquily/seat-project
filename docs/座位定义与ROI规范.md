# 座位定义与 ROI 规范（v2）

> 2026-11-05 与项目负责人确认的口径，替代早期"桌面四等分"的临时模型。

## 一、座位定义

1. **一个座位 = 一把椅子 + 椅子正前方一定范围的桌面**；
2. 一张大桌子旁有 N 把椅子，就有 **N 个座位**（座位数 = 贴桌椅子数）；
3. 判定时在座位区域内统计 person / 物品框，任一物品命中即"有物无人"，person 命中即"有人"。

## 二、数据模型（config/rois.json）

```json
{
  "seat_model": "chair+desk",
  "seats": [
    {
      "id": "B01",                // 座位编号
      "name": "B桌1号",
      "side": "top",              // 椅子所在边：top远边 / bottom近边 / left左端 / right右端
      "chair": [x1, y1, x2, y2],  // 椅子区（画面坐标）
      "desk":  [x1, y1, x2, y2],  // 椅前桌面条带
      "roi":   [x1, y1, x2, y2]   // 判定区域 = chair ∪ desk（外接矩形）
    }
  ]
}
```

`config.yaml` 的 `seats:` 段由 `make_web_rois.py --sync-config` 自动生成，勿手工编辑。

## 三、生成与标定流程

```bash
# 1. 在叠加坐标网格的放大图上人工读出：桌面矩形 + 每把椅子的边与沿边位置
#    （参考 data/video_demo/desk_*_grid.jpg 的做法）
# 2. 把桌子登记进 scripts/make_web_rois.py 的 DESKS 表
python scripts/make_web_rois.py --sync-config   # 生成 rois.json + 预览图 + 覆盖 config.yaml
# 3. 渲染核对
python scripts/judge_seats.py --image <底图>     # 判定结果应与画面事实一致
# 4. 微调仍可手改 rois.json 后用 roi_tool.py 拖框精修
```

几何参数在 `make_web_rois.py` 的 `SIDE_PARAMS`：椅子区与桌面条带的深度按桌子尺寸比例伸缩
（近边椅子在画面里更大；bottom 带深压到椅子实际高度，防止画面更近处的人被算进来）。

## 四、判定规则（scripts/judge_seats.py）

| 规则 | 参数（config.yaml） | 说明 |
|---|---|---|
| 人物归属 | `person_iou_threshold: 0.50` | 人体框主体落在座位 ROI 内；取高值防止**邻桌的人**被算进来 |
| 物品归属 | `object_iou_threshold: 0.10` + `min_object_area_ratio: 0.02` | 物品框落入 ROI 且有最小面积，过滤远端噪点 |
| 独占分配 | — | 每个检测框只归属重叠比例最大的座位，相邻座位不重复计数 |
| 同人去重 | — | 同一座位内两个 person 框交集占较小框 >50% 视为同一人（NMS 漏检兜底） |

## 五、演示素材实测（main_view.jpg，16 座）

| 桌 | 座位数 | 椅子分布 | 判定结果 |
|---|---|---|---|
| A | 6 | 远 2 + 近 3 + 左端 1 | 全空 ✓ |
| B | 4 | 远 2 + 右端 1 + 近 1 | B01/B03 有人 ✓，B02 有物无人（笔记本）✓，B04 空 ✓ |
| C | 6 | 远 3 + 右端 1 + 近 2 | C02/C03 有物无人（白纸）✓，其余未检出物品 |

**已知模型缺口**（ROI 无关，等真实数据重训解决）：
1. A02 俯视白衣人未被 seat_v1 检出（召回不足）；
2. C01/C04/C05/C06 桌面白纸未全部检出（item 召回不足）。

## 六、与真实素材的衔接

真实照片到位后：按第一~三节同模型标定（椅子数即座位数，不设固定 12 座）→
`--sync-config` 更新主配置 → 重新跑 `track_stream.py` 与看板，命令不变。
