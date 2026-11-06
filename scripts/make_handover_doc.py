# -*- coding: utf-8 -*-
"""生成《交接说明报告》Word 文档（python-docx）。

用法：
    python scripts/make_handover_doc.py --out 交接说明报告.docx
"""

import argparse
import os
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BODY_CN = "宋体"
HEAD_CN = "黑体"
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
NOTE_FILL = "DEEBF7"


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def set_run_font(run, name_cn=BODY_CN, size=12, bold=False, color=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name_cn)
    rfonts.set(qn("w:ascii"), "Times New Roman")
    rfonts.set(qn("w:hAnsi"), "Times New Roman")


def para(doc, text="", size=12, bold=False, align=None, indent=True,
         space_before=0, space_after=6, line=1.5, color=None, cn=BODY_CN):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if indent:
        pf.first_line_indent = Pt(size * 2)
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        set_run_font(run, name_cn=cn, size=size, bold=bold, color=color)
    return p


def shade(paragraph, fill):
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def left_border(paragraph, color="2E74B5", size=18):
    ppr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), "6")
    left.set(qn("w:color"), color)
    borders.append(left)
    ppr.append(borders)


def note(doc, text, title="提示"):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.4)
    pf.right_indent = Cm(0.2)
    pf.space_before = Pt(4)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.35
    run = p.add_run(f"{title}：{text}")
    set_run_font(run, size=10.5)
    shade(p, NOTE_FILL)
    left_border(p)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(14 if level == 1 else 10)
    pf.space_after = Pt(6)
    pf.keep_with_next = True
    run = p.add_run(text)
    if level == 1:
        set_run_font(run, name_cn=HEAD_CN, size=16, bold=True, color=ACCENT)
        # 段前加一条浅色下边线
        ppr = p._p.get_or_add_pPr()
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "2")
        bottom.set(qn("w:color"), "9DC3E6")
        borders.append(bottom)
        ppr.append(borders)
    elif level == 2:
        set_run_font(run, name_cn=HEAD_CN, size=13.5, bold=True, color=RGBColor(0x2E, 0x74, 0xB5))
    else:
        set_run_font(run, name_cn=HEAD_CN, size=12, bold=True)
    # 加入导航大纲级别，方便在 Word 里折叠
    ppr = p._p.get_or_add_pPr()
    outline = OxmlElement("w:outlineLvl")
    outline.set(qn("w:val"), str(level - 1))
    ppr.append(outline)
    return p


def bullet(doc, text, size=11.5):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Cm(0.9)
    pf.first_line_indent = Cm(-0.4)
    pf.space_after = Pt(3)
    pf.line_spacing = 1.4
    run = p.add_run("· " + text)
    set_run_font(run, size=size)
    return p


def code_block(doc, lines):
    for line in lines:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.8)
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = 1.15
        run = p.add_run(line)
        set_run_font(run, name_cn="Consolas", size=9.5)
        run.font.name = "Consolas"
        shade(p, "F4F4F4")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def table(doc, headers, rows, widths=None, font_size=9.5, header_fill="DEEBF7"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        run = p.add_run(h)
        set_run_font(run, name_cn=HEAD_CN, size=font_size, bold=True)
        tcpr = hdr[i]._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), header_fill)
        tcpr.append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.2
            bold = str(value).startswith("**")
            text = str(value).replace("**", "")
            run = p.add_run(text)
            set_run_font(run, size=font_size, bold=bold)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    # 表头跨页重复
    trpr = t.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    trpr.append(repeat)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def add_page_number(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("— ")
    set_run_font(run, size=9)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run2 = p.add_run()
    set_run_font(run2, size=9)
    run2._r.append(fld_begin)
    run2._r.append(instr)
    run2._r.append(fld_end)
    run3 = p.add_run(" —")
    set_run_font(run3, size=9)


def header_text(section, text):
    p = section.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(text)
    set_run_font(run, size=9, color=RGBColor(0x80, 0x80, 0x80))


# --------------------------------------------------------------------------
# 文档主体
# --------------------------------------------------------------------------
def build(out_path):
    doc = Document()

    # 页面设置：A4，装订边距
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.4)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CN)

    header_text(sec, "自习室座位占用识别系统 · 交接说明报告 v7b")
    add_page_number(sec)

    # ---------------- 封面 ----------------
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("自习室座位占用识别系统")
    set_run_font(run, name_cn=HEAD_CN, size=30, bold=True, color=ACCENT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run("交 接 说 明 报 告")
    set_run_font(run, name_cn=HEAD_CN, size=22, bold=False, color=RGBColor(0x2E, 0x74, 0xB5))

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    run = p.add_run("安装部署 · 使用说明 · 现状不足 · 改进路线")
    set_run_font(run, size=12, color=RGBColor(0x59, 0x59, 0x59))

    for _ in range(5):
        doc.add_paragraph()

    info = [
        ("交付版本", "v7b（2026-11-06）"),
        ("模型", "YOLOv8 · seat_v7b（主）/ seat_v6（标定场景）"),
        ("回归指标", "28 张自习室图片 89.3%（25/28），人数 MAE 1.68"),
        ("交付对象", "项目协作伙伴"),
        ("文件形态", "seat-project-v7b-20261106.zip（自包含，开箱可跑）"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for key, value in info:
        cells = t.add_row().cells
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r0 = p0.add_run(key + "：")
        set_run_font(r0, size=11, bold=True)
        p1 = cells[1].paragraphs[0]
        r1 = p1.add_run(value)
        set_run_font(r1, size=11)
        cells[0].width = Cm(3.4)
        cells[1].width = Cm(10.5)

    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("高校课程项目 · 内部交接材料")
    set_run_font(run, size=10.5, color=RGBColor(0x80, 0x80, 0x80))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("2026 年 11 月")
    set_run_font(run, size=10.5, color=RGBColor(0x80, 0x80, 0x80))

    doc.add_page_break()

    # ---------------- 目录 ----------------
    heading(doc, "目 录", 1)
    toc = [
        ("一、项目概况", "任务定义、系统组成、技术路线"),
        ("二、安装与部署", "环境要求、安装步骤、自检、启动与常驻"),
        ("三、文件与配置说明", "目录结构、关键配置项、换机位重标、重训流程"),
        ("四、使用说明", "网页看板、命令行、视频与摄像头"),
        ("五、当前能力与验收", "实测指标、验收清单、已知失败样本"),
        ("六、不足与成因分析", "12 项不足清单 + 三条根因"),
        ("七、改进路线", "四周计划、目标指标、不建议的做法"),
        ("八、协作分工与数据需求", "分工建议、数据采集清单、资源需求"),
        ("九、风险与合规", "隐私、误判、机位变动、版本管理"),
        ("十、版本记录", "v6 → v7 → v7b 的演进"),
    ]
    for title, desc in toc:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.6)
        pf.space_after = Pt(8)
        pf.line_spacing = 1.4
        run = p.add_run(title)
        set_run_font(run, name_cn=HEAD_CN, size=12, bold=True)
        run2 = p.add_run("    " + desc)
        set_run_font(run2, size=10.5, color=RGBColor(0x59, 0x59, 0x59))

    note(doc, "阅读顺序建议：先看第二节把环境跑通（约 30 分钟），再用第五节验收，之后按第六、七节挑选要改的点。"
              "所有命令都在项目根目录执行。")

    doc.add_page_break()

    # ---------------- 一、项目概况 ----------------
    heading(doc, "一、项目概况", 1)

    heading(doc, "1.1 任务定义", 2)
    para(doc, "输入一张自习室画面（或一段视频、一路摄像头），输出三件事：")
    bullet(doc, "画面里有几个座位 —— 本项目的第一优先级：宁可多算，不可漏算；")
    bullet(doc, "每个座位当前状态 —— 有人 / 有物无人 / 空置 / 不在画面；")
    bullet(doc, "「有物无人」持续超过 30 分钟即记为占座事件，支持午休等豁免时段。")

    heading(doc, "1.2 座位判定规则（两级）", 2)
    table(doc,
          ["级别", "规则", "实现位置"],
          [["第一级\n座位存在性",
            "① 一把椅子 / 凳 = 一个座位；② 画面受限看不到椅子、但该处检出人 → 也算一个座位（隐式座位）。"
            "两者不重复计：人体框若已压在已知椅子的落座区（覆盖率 ≥ 15%），不再生成隐式座位。",
            "scripts/dynamic_seats.py\ninfer_dynamic_seats"],
           ["第二级\n占座状态",
            "座位范围内检出人 → 有人；无人但有书 / 包 / 杯 / 杂物 → 有物无人；都没有 → 空置；"
            "配置 exists_check: stool 的座位若连凳子都检不到 → 不在画面。",
            "scripts/judge_seats.py\njudge_seats"]],
          widths=[2.6, 9.6, 4.0])

    heading(doc, "1.3 技术路线", 2)
    para(doc, "系统由「检测 → 座位判定 → 状态机 → 看板」四段组成，检测层是两条通道并存：")
    table(doc,
          ["通道", "触发条件", "权重 / 阈值", "特点"],
          [["标定通道", "上传图与参考图 ORB 特征匹配、单应性 RANSAC 内点 ≥ 60",
            "seat_v6 + conf 0.15", "11 座固定表，框位置最准，座位数确定"],
           ["动态通道", "未命中参考图（任意自习室照片）",
            "seat_v7b + COCO yolov8n + conf 0.25", "不依赖 ROI，座位由检测推出；远景密集图自动 1280px 切片（25% 重叠）"]],
          widths=[2.2, 5.4, 4.4, 4.6])
    para(doc, "动态通道另有双模型融合：项目模型与 COCO 模型各跑一遍人体，两框交集 ÷ 较小框面积 > 0.70 视为同一人，"
              "以此兼顾「远处小人体」与「坐姿遮挡人体」。")
    note(doc, "为什么要两个权重？白天外套椅上的帆布包，seat_v6 能给出 0.19 的框，seat_v7b 直接没有；"
              "若标定场景也用 v7b，「有物无人」会被误判成「空置」。这个取舍是实测换来的，但也构成了本版本最大的技术债（见 6.1 第 1 项）。")

    heading(doc, "1.4 占座状态机", 2)
    para(doc, "app/state_machine.py 负责时序判定：连续 N 帧一致才更新状态（防抖）→「有物无人」累计满 30 分钟 → 生成占座事件；"
              "人回来记 RESOLVED，物品被清走记 CLEARED；豁免时段内的时间不计入累计。13 条单元测试覆盖上述规则。")

    # ---------------- 二、安装与部署 ----------------
    heading(doc, "二、安装与部署", 1)

    heading(doc, "2.1 环境要求", 2)
    table(doc,
          ["项目", "要求", "说明"],
          [["操作系统", "Windows 10 / 11", "其他系统可运行，但 .bat 脚本需改写"],
           ["Python", "3.10", "本项目实测环境；3.12 未验证"],
           ["内存", "≥ 8 GB", "1280 输入推理峰值约 2 GB"],
           ["显卡", "可选（NVIDIA ≥ 6 GB 更佳）", "有显卡时把 config 的 model.device 改为 0，可提速 5~10 倍"],
           ["磁盘", "≥ 1 GB", "含数据集与两份权重"],
           ["路径", "纯英文、无空格", "OpenCV 不支持中文路径"]],
          widths=[2.6, 5.4, 8.2])

    heading(doc, "2.2 安装步骤", 2)
    code_block(doc, [
        ":: 1. 解压到纯英文无空格路径，例如 D:\\seat-project",
        "cd /d D:\\seat-project",
        "",
        ":: 2. 建 conda 环境",
        "conda create -n seat python=3.10 -y",
        "conda activate seat",
        "",
        ":: 3. 安装依赖（国内建议走清华源）",
        "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "",
        ":: 4. 自检：依赖 + 权重 + 13 条单测 + real_03 判定",
        "selfcheck.bat",
    ])

    heading(doc, "2.3 自检通过标准", 2)
    para(doc, "selfcheck.bat 正常结束时应出现以下两段输出，数值一致即环境完全正确：")
    code_block(doc, [
        "Ran 13 tests in 0.004s",
        "OK",
        "",
        "[汇总] 在座人数 6｜有人 6 / 有物无人 2 / 空置 3 / 不在画面 0",
    ])

    heading(doc, "2.4 启动、停止与常驻", 2)
    table(doc,
          ["操作", "方式"],
          [["启动", "双击 start_server.bat（自动探测 Python、自动打开浏览器 http://127.0.0.1:8000/）"],
           ["停止", "启动窗口内 Ctrl+C，或双击 stop_server.bat"],
           ["换端口", "命令行执行 set SEAT_PORT=8011 后再双击 start_server.bat"],
           ["指定解释器", "设环境变量 SEAT_PYTHON 指向 python.exe，或直接改 bat 内的 PYTHON_EXE"]],
          widths=[2.6, 13.6])
    para(doc, "若需要「关掉窗口服务仍在跑」，可选两种方式：")
    bullet(doc, "任务计划程序：触发器设为「登录时」，操作指向 run_server_service.bat，勾选「不管用户是否登录都运行」；")
    bullet(doc, "NSSM（推荐）：nssm install SeatServer <python.exe> -m uvicorn app.main:app --host 127.0.0.1 --port 8000，再 nssm start SeatServer。")
    note(doc, "交付机器（原开发机）上创建计划任务与系统服务均被安全策略拦截，因此只能手动双击启动；"
              "这是环境限制，不是程序缺陷。")

    heading(doc, "2.5 常见安装问题", 2)
    table(doc,
          ["现象", "原因", "处理"],
          [["双击 bat 闪退", "路径含中文 / 空格，或未装 Python", "整体移动到 D:\\seat-project 重试"],
           ["提示缺少依赖", "pip 装到了别的环境", "用 python -m pip install 并在提示里确认解释器路径"],
           ["8000 端口打不开", "端口被占用或服务已退出", "stop_server.bat 释放，或 set SEAT_PORT 换端口"],
           ["网页空白 / 502", "服务未启动完成", "看启动窗口有无红色报错，首帧需 5~10 秒加载模型"],
           ["判定框错位、座位数不对", "机位或桌椅位置变化", "用 roi_tool.py 重标座位（见 3.3）"]],
          widths=[3.6, 5.2, 7.4])

    # ---------------- 三、文件与配置 ----------------
    heading(doc, "三、文件与配置说明", 1)

    heading(doc, "3.1 目录结构与用途", 2)
    table(doc,
          ["路径", "内容", "是否必改"],
          [["config/config.yaml", "座位 ROI、判定阈值、双权重、豁免时段、演示底图", "是（唯一常改文件）"],
           ["start_server.bat / stop_server.bat", "一键启停服务（自动探测 Python）", "否"],
           ["selfcheck.bat", "安装自检：依赖 + 权重 + 单测 + 标定图判定", "否"],
           ["app/main.py", "服务接口：图片 / 视频识别、座位、事件、统计", "否"],
           ["app/state_machine.py", "占座状态机（防抖、计时、豁免）", "改规则时才动"],
           ["app/static/index.html", "网页看板（单文件，原生 JS）", "改界面时才动"],
           ["scripts/judge_seats.py", "单图判定（固定座位表）", "否"],
           ["scripts/dynamic_seats.py", "任意场景动态座位识别（两级判定 / 融合 / 切片）", "调算法时动"],
           ["scripts/scene_match.py", "ORB 场景匹配，判断是否标定机位", "换机位时调阈值"],
           ["scripts/batch_test.py", "28 张回归测试，输出准确率与 MAE", "否"],
           ["scripts/track_stream.py", "视频流 / 摄像头时序跟踪", "否"],
           ["scripts/train_yolov8.py", "训练入口（默认 150 轮 / 960 输入）", "重训时用"],
           ["scripts/roi_tool.py", "座位 ROI 标定工具", "换机位必用"],
           ["runs/train/seat_v7b/weights/best.pt", "主权重：动态 / 视频 / 看板", "否"],
           ["runs/train/seat_v6/weights/best.pt", "标定场景权重（低置信物品更敏感）", "否"],
           ["models/yolov8n.pt", "COCO 预训练权重，提供 person / chair 第二路", "否"],
           ["data/raw_real/", "9 张真实自习室照片 + 单座测试视频", "否"],
           ["data/seat_dataset/", "训练 29 / 验证 3 / 测试 5（YOLO 格式 + data.yaml）", "补数据后重划分"],
           ["output/batch_28/", "28 张回归结果（JSON + 报告 + 6 张标注样例）", "否"],
           ["docs/", "过程文档：采集手册、标注规范、训练记录、技术障碍分析", "建议阅读"]],
          widths=[5.0, 9.0, 3.4], font_size=9)

    heading(doc, "3.2 关键配置项", 2)
    table(doc,
          ["配置项", "当前值", "含义与调整建议"],
          [["model.weights", "seat_v7b", "动态通道权重；换模型时改"],
           ["model.fixed_weights", "seat_v6", "标定通道权重；与 weights 配套使用"],
           ["model.conf / scene_conf / video_conf", "0.25 / 0.15 / 0.20", "三通道置信度；漏检多降 0.02~0.05，误检多升"],
           ["model.imgsz", "1280", "推理尺寸；远处小目标漏检可升，速度不够可降"],
           ["model.device", "cpu", "有 NVIDIA 显卡改 0"],
           ["seats[]", "11 座 S01–S11", "固定座位表；换机位 / 挪桌必须重标"],
           ["scene_references", "real_02 / real_03", "标定机位参考图；换机位时替换"],
           ["occupancy.person_iou_threshold", "0.50", "人体框与 ROI 的重叠阈值"],
           ["occupancy.person_spill_own_ratio", "0.75", "跨座溢出主框的剔除上限（调低会重复计数）"],
           ["occupancy.object_leave_minutes", "30", "判定占座的时长"],
           ["exempt_time.windows", "午休 / 闭馆前", "不计占座的时段"]],
          widths=[5.4, 3.4, 8.6], font_size=9)

    heading(doc, "3.3 换机位怎么重标座位", 2)
    para(doc, "ROI 与机位强绑定，这是固定机位方案的固有约束。换机位或桌子被挪动后按以下流程重标：")
    code_block(doc, [
        ":: 1) 新机位空场照片放到 data\\raw_real\\new.jpg",
        "python scripts\\roi_tool.py --image data\\raw_real\\new.jpg",
        "",
        ":: 2) 工具内逐座框选，填 id / name，导出 config\\rois.json",
        ":: 3) 用 rois.json 的 seats 段覆盖 config.yaml 的 seats",
        ":: 4) 把新照片加入 config.yaml 的 scene_references",
        "",
        ":: 5) 核对：逐座输出人 / 物数量",
        "python scripts\\judge_seats.py --image data\\raw_real\\new.jpg ^",
        "       --weights runs\\train\\seat_v6\\weights\\best.pt --conf 0.15",
    ])

    heading(doc, "3.4 重新训练流程", 2)
    code_block(doc, [
        "python scripts\\prelabel.py --images data\\workspace\\images   :: 预标注粗标",
        "python scripts\\verify_labels.py                             :: 标签合法性检查",
        "python scripts\\clean_labels.py --apply                      :: 合并重复框、删噪声框",
        "python scripts\\split_dataset.py                             :: 重划分 + 生成 data.yaml",
        "python scripts\\train_yolov8.py --model models\\yolov8n.pt --epochs 150 --imgsz 960 --name seat_v8",
        "python scripts\\batch_test.py                                :: 回归，必须不退化",
    ])
    para(doc, "标签纪律（都是踩过坑总结的，务必遵守）：")
    bullet(doc, "小物件（书 / 杯）必须在原分辨率局部裁剪并叠加网格读数，整图估坐标会有约 80 px 系统偏移；")
    bullet(doc, "同一物体只标一个框，不同类别不要重复框同一处；")
    bullet(doc, "椅子被完全遮挡时不要凭想象补框，否则会训练出幻影框；person 只标可见人体；")
    bullet(doc, "每次改完标签先 verify_labels + clean_labels，再训练；训练后必须跑 28 图回归对比。")

    # ---------------- 四、使用说明 ----------------
    heading(doc, "四、使用说明", 1)

    heading(doc, "4.1 网页看板", 2)
    table(doc,
          ["入口", "功能"],
          [["实时看板", "座位卡片按状态配色（红=有人、黄=有物无人、灰=空置），定时刷新"],
           ["上传图片", "返回标注图、每座状态、全图人数、耗时"],
           ["上传视频", "逐帧识别并回放；时间倍率可把 40 秒素材压成 30 分钟以演示占座判定"],
           ["立即检测", "用 config.yaml 的 demo.image 跑一次，验证链路是否正常"],
           ["事件 / 统计", "占座事件的开始、结束、处理结果与当日汇总"]],
          widths=[3.0, 13.2])

    heading(doc, "4.2 命令行", 2)
    table(doc,
          ["目的", "命令"],
          [["标定场景单图", "python scripts\\judge_seats.py --image data\\raw_real\\real_03.jpg --weights runs\\train\\seat_v6\\weights\\best.pt --conf 0.15"],
           ["任意场景单图", "python scripts\\judge_seats.py --image data\\raw_real\\real_08.jpg"],
           ["28 图回归", "python scripts\\batch_test.py"],
           ["视频跟踪", "python scripts\\track_stream.py --source video --video data\\raw_real\\video.mp4 --frame-step 15 --seats-key video_seats --db data\\check.db --minutes-per-frame 1.2"],
           ["单元测试", "python -m unittest discover -s tests"]],
          widths=[3.0, 13.2], font_size=9)
    note(doc, "命令行脚本不会自动做双权重路由，所以标定场景要显式带 --weights seat_v6 --conf 0.15，"
              "否则会按 v7b + 0.25 跑，物品可能漏检。这是当前版本的口径不一致点，已列入改进清单 P0。")

    heading(doc, "4.3 视频与摄像头", 2)
    para(doc, "视频通道使用 config.yaml 的 video_seats 座位表；把 --source 改为 0 即读取本机摄像头，"
              "改为 rtsp://… 即读取网络摄像头。时间倍率 time_scale 用于把短视频映射成真实分钟数，方便演示 30 分钟占座规则。")

    # ---------------- 五、能力与验收 ----------------
    heading(doc, "五、当前能力与验收", 1)

    heading(doc, "5.1 实测指标", 2)
    table(doc,
          ["指标", "数值", "说明"],
          [["28 张回归集准确率", "89.3%（25/28）", "容差：基准 <10 人允许 ±2；≥10 人允许 ±20%"],
           ["人数平均绝对误差", "1.68 人", "上一版 v6 为 2.32 人"],
           ["标定机位座位数", "11 座全对", "real_03：6 有人 / 2 有物无人 / 3 空置；全图 6 人"],
           ["模型 mAP50", "0.586（person 0.888）", "37 张图的小数据集，不宜对标公开榜单"],
           ["CPU 单图耗时", "1~2 秒（首帧约 6 秒）", "1280 输入 + 双模型，无多路实时能力"]],
          widths=[4.0, 4.0, 8.2])

    heading(doc, "5.2 验收清单", 2)
    table(doc,
          ["#", "检查项", "期望结果"],
          [["1", "运行 selfcheck.bat", "13 条单测 OK；real_03 输出 6 / 2 / 3，全图 6 人"],
           ["2", "看板上传 real_03", "走标定通道，11 座，6 / 2 / 3，全图 6 人"],
           ["3", "看板上传任意网络自习室图", "走动态通道，给出座位数与人数，标注框不重叠成团"],
           ["4", "运行 batch_test.py", "准确率 ≥ 89.3%，MAE ≤ 1.68"],
           ["5", "上传 video.mp4", "时序出现 空置 → 有人 → 有物无人 → 有人 → 空置"],
           ["6", "看板「立即检测」", "返回标注图，无 500 错误"]],
          widths=[1.2, 6.4, 8.6])

    heading(doc, "5.3 已知失败样本", 2)
    table(doc,
          ["图片", "基准人数", "检出人数", "原因"],
          [["web_f0011", "14", "11", "低分辨率 + 前排遮挡，NMS 把相邻人头合并"],
           ["real_02", "13", "8~10", "边缘 / 画面外人体算不算，基准口径本身不一致；两条通道分别给出 8 与 10"],
           ["real_08", "4", "10", "后墙海报证件照被当成真人，属域外干扰"]],
          widths=[3.0, 2.4, 2.4, 8.4])

    # ---------------- 六、不足 ----------------
    heading(doc, "六、不足与成因分析", 1)
    para(doc, "下面 12 项按「对交付质量的影响」排序，P0 表示不解决就会持续拖累准确率或解释成本。")

    heading(doc, "6.1 不足清单", 2)
    table(doc,
          ["#", "不足", "成因", "影响", "改进方向", "优先级"],
          [["1", "双权重 + 双阈值路由", "为救一个 0.19 置信度的帆布包，标定通道退回 v6",
            "维护两套权重；命令行与服务结果不一致；换场景要重调阈值",
            "单权重 + 分类别阈值 + 类别加权损失重训，补低光与遮挡物品样本", "P0"],
           ["2", "阈值极度敏感", "数据集小、类别不平衡，置信度分布被压平",
            "0.25 漏物品，0.15 进噪声（S03/S04 各多出 3 个物品框）",
            "补 200+ 物品框；物品类过采样；加「物品-桌面」几何约束过滤", "P0"],
           ["3", "密集场景粘连", "输入尺寸下人头像素过少，NMS IoU 0.45 合并相邻人",
            "f0011 类图少算 3 人；大教室座位数偏低", "1280 + SAHI 切片 + yolov8s/m + copy-paste 增广", "P1"],
           ["4", "座位存在性有灰区", "椅子检测不稳（折叠椅、遮挡、凳子形变）；隐式座位可能与椅子重复",
            "座位数忽多忽少，用户最直观的不满点",
            "用单应性把 ROI 投影到地面平面，按人脚点聚类生成座位，改为「座位图」而非逐框判定", "P1"],
           ["5", "海报 / 屏幕 / 照片误检", "训练集缺少此类负样本", "real_08 多算 6 人",
            "补 30+ 张负样本；加人脸尺寸与纹理一致性过滤", "P1"],
           ["6", "ROI 与机位强绑定", "固定机位假设", "挪桌后座位表全错，必须人工重标",
            "新图 ORB 配准后自动变换 ROI，人工只做微调", "P2"],
           ["7", "无时序一致性约束", "每帧独立判定，只看单帧证据", "人走过、弯腰会瞬时误判，座位数帧间跳变",
            "接 ByteTrack，用轨迹 ID 与座位关联，3 帧投票 + 停留时长约束", "P2"],
           ["8", "全图人数口径不统一", "标定通道剔除跨座溢出框后计数，动态通道直接数框",
            "同一张图两条通道 8 与 10，答辩时难以解释",
            "统一为「座位人数之和 + 未入座人数」，前端标注口径", "P0"],
           ["9", "数据集太小且单一", "37 张图，几乎全来自两个机位加网络图", "换场景掉点；mAP 上限低",
            "按 8.2 清单补到 150+ 张，覆盖夜景、拥挤、遮挡、不同桌椅", "P1"],
           ["10", "服务无法开机自启", "原开发机安全策略禁计划任务与服务", "每次演示都要手动双击",
            "伙伴机器用 NSSM；或 PyInstaller 打包单文件 exe", "P2"],
           ["11", "无自动回归门禁", "靠人工跑 batch_test 看结果", "改动可能悄悄退化",
            "batch_test + 单测接入 CI（CPU 小样本），指标下降即失败", "P2"],
           ["12", "CPU 推理偏慢", "1280 输入 + 双模型", "单图 1~2 秒，无法多路", "GPU + 半精度 + 模型蒸馏", "P3"]],
          widths=[0.9, 2.8, 3.6, 3.2, 4.2, 1.4], font_size=8)

    heading(doc, "6.2 三条根因", 2)
    para(doc, "把 12 项收敛成三条，后续所有改进都可以挂在这三条下面：")
    bullet(doc, "根因一｜数据不够：37 张图撑不起 6 类检测，物品类样本尤其稀缺，导致置信度分布平、阈值一动就出事。")
    bullet(doc, "根因二｜单帧语义：只看单帧的框与重叠率，没有时序与空间约束，遮挡、走动、海报都会被当成事实。")
    bullet(doc, "根因三｜口径不统一：座位数、人数在不同通道各有一套统计口径，工程上没有唯一真相，调参与解释都容易跑偏。")

    # ---------------- 七、改进路线 ----------------
    heading(doc, "七、改进路线", 1)

    heading(doc, "7.1 四周计划", 2)
    table(doc,
          ["周次", "目标", "具体动作", "验收"],
          [["第 1 周", "统一口径", "合并双权重（先做分类别阈值试验）；统一人数口径；把 28 图回归 + 11 座标定做成一键脚本并记录基线",
            "命令行与服务结果一致；基线数字进文档"],
           ["第 2 周", "补数据", "按 8.2 清单拍摄、标注；重点补遮挡、夜景、海报负样本、椅子特写",
            "新增 ≥ 50 张标注，交叉复核完成"],
           ["第 3 周", "重训 + 切片", "yolov8s + 1280 + SAHI；分类别阈值；回归对比",
            "28 图 ≥ 92%，MAE ≤ 1.2"],
           ["第 4 周", "时序与工程", "轨迹关联 + 3 帧投票；服务常驻；回归 CI；结题材料",
            "视频通道误判显著下降，演示可自动启动"]],
          widths=[1.8, 2.6, 7.4, 4.4], font_size=9)

    heading(doc, "7.2 目标指标（下一版 v8）", 2)
    table(doc,
          ["指标", "当前 v7b", "v8 目标"],
          [["28 图准确率", "89.3%", "≥ 92%"],
           ["人数 MAE", "1.68 人", "≤ 1.2 人"],
           ["标定机位座位状态", "11 座全对", "保持全对"],
           ["单权重", "双权重（v6 + v7b）", "合并为单权重 + 分类别阈值"],
           ["海报误检", "real_08 多 6 人", "≤ 1 人"]],
          widths=[5.6, 5.4, 5.2], font_size=9.5)

    heading(doc, "7.3 不建议的做法", 2)
    bullet(doc, "不要为提高单张图效果改全局阈值 —— 一定会在别的图上退化（0.55 溢出阈值曾导致跨座重复计数）。")
    bullet(doc, "不要只发 best.pt 不配套 config.yaml —— 座位表、阈值、权重是一个整体版本。")
    bullet(doc, "不要用网络图当验证集反复调参 —— 那是过拟合测试集，答辩一问就露。")

    # ---------------- 八、协作 ----------------
    heading(doc, "八、协作分工与数据需求", 1)

    heading(doc, "8.1 分工建议", 2)
    table(doc,
          ["角色", "职责", "产出"],
          [["算法 / 数据", "补拍与标注、重训、阈值与判定逻辑调整", "新权重 + 标注集 + 回归比测报告"],
           ["工程 / 评测", "服务常驻、回归 CI、看板与接口、盲测集构建", "可自动启动的服务 + CI 结果 + 盲测报告"],
           ["共同", "每周一次回归对比，指标写进版本记录", "版本记录表"]],
          widths=[2.6, 8.6, 5.0])

    heading(doc, "8.2 数据采集清单（按重要性）", 2)
    table(doc,
          ["#", "类型", "数量", "拍摄要求"],
          [["1", "拥挤遮挡场景", "≥ 10 张", "前排挡后排、成排人头密集；白天与夜间各半"],
           ["2", "夜景 / 低照度", "≥ 5 张", "只开部分灯，保留真实噪点，不要开夜景增强"],
           ["3", "椅子特写", "≥ 20 张", "各种椅型（折叠椅、红胶凳、搭外套的椅），含被桌挡半的"],
           ["4", "负样本", "≥ 30 张", "后墙海报、证件照墙、显示器、镜面反射、堆叠空椅"],
           ["5", "入座→离座视频", "2~3 段", "固定机位，每段 ≥ 3 分钟，含放包离开与回来取物"],
           ["6", "布局变动当天照", "1 张 / 次", "变动后立刻拍，用于验证 ROI 失效检测"]],
          widths=[0.9, 3.4, 2.6, 9.3])

    heading(doc, "8.3 需要提供的资源", 2)
    bullet(doc, "硬件：NVIDIA 显卡（≥ 6 GB 显存）可显著缩短训练与推理时间；固定机位支架保证 ROI 不漂。")
    bullet(doc, "环境：一台常开机器（若要长时间演示）；公网部署涉及隐私合规，课程阶段不建议。")
    bullet(doc, "数据：含人物面部的素材仅限课程内部流转，不上公开仓库、不对外演示原图。")

    # ---------------- 九、风险 ----------------
    heading(doc, "九、风险与合规", 1)
    table(doc,
          ["风险", "说明", "应对"],
          [["隐私", "数据集含真实人物面部", "仅课程内部使用；对外演示用网络图或做模糊处理"],
           ["误判", "占座判定用于提示而非处罚", "界面避免「违规」措辞；VIOLATION 仅作内部字段"],
           ["机位变动", "挪桌后不重标会出现「座位数对不上」的假故障", "挪桌当天补拍并重标，写进运维流程"],
           ["版本混乱", "权重、配置、数据三者错配", "任何交付都整套打包，附 batch_28 结果与 SHA256"]],
          widths=[2.6, 7.2, 6.4])

    # ---------------- 十、版本记录 ----------------
    heading(doc, "十、版本记录", 1)
    table(doc,
          ["版本", "时间", "主要变化", "28 图准确率", "人数 MAE"],
          [["v6", "11-06 上午", "标签清理前的基线，6 类检测", "82.1%", "2.32"],
           ["v7", "11-06 晚", "640 输入重训，验证集反而下降，弃用", "—", "—"],
           ["v7b", "11-06 21:00", "960 输入 / 150 轮重训 + 两级座位判定 + 紧致证据框 + 标签清理 19 处", "89.3%", "1.68"]],
          widths=[1.8, 3.2, 7.0, 2.6, 2.2], font_size=9)

    para(doc, "本次交付 = v7b 全量快照（含 v6 权重，供标定通道使用）。", indent=True)
    doc.add_paragraph()
    para(doc, "维护人：（待填）        交付日期：2026 年 11 月 6 日        联系方式：（待填）", indent=False)

    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="交接说明报告.docx")
    args = parser.parse_args()
    path = build(args.out)
    print("saved:", os.path.abspath(path))
