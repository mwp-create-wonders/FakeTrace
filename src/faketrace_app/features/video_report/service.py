from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
import uuid
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ...core.paths import PROJECT_ROOT
from ..localization_report.service import (
    BODY_INDENT_STYLE,
    BODY_STYLE,
    DOUBAO_BASE_URL,
    DOUBAO_MODEL,
    MAX_DOUBAO_WORKERS,
    REQUEST_TIMEOUT_SECONDS,
    SMALL_CENTER_STYLE,
    TITLE_STYLE,
    _append_analysis_section,
    _bullet_line,
    _draw_first_page,
    _draw_later_page,
    _format_percent,
    _keep_table_together,
    _p,
    _parse_upload_time,
    _read_api_key,
    _section_header,
    _style,
    _three_line_table_style,
)


REPORT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf" / "video"

# ==================== 样式 ====================
DETECTOR_HEADER_STYLE = _style("DetectorHeaderCN", 12.5, 18, TA_CENTER)
DETECTOR_VALUE_STYLE = _style("DetectorValueCN", 12.5, 18, TA_CENTER)
SUMMARY_FILENAME_STYLE = _style("SummaryFilenameCN", 9.6, 12, TA_CENTER)
CHART_CAPTION_STYLE = _style("ChartCaptionCN", 10, 14, TA_CENTER)
RESULT_TEXT_STYLE = _style("ResultTextCN", 11.5, 16, TA_LEFT)


@dataclass
class VideoReportItem:
    index: int
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    total_frames: int
    prediction: str
    fake_probability: float
    real_probability: float
    risk: str = "未知"
    analysis: str = ""
    threshold: float = 0.5
    frame_info: list[dict] | None = None
    velocity_l2: list[float] | None = None
    acceleration_l2: list[float] | None = None
    lota_scores: list[float] | None = None
    anomaly_text: str = ""
    suspicious_frame_b64: str | None = None
    suspicious_frame_time: float | None = None
    trend_chart_png: bytes | None = None


@dataclass
class GeneratedReport:
    report_id: str
    path: Path


# ==================== 工具函数 ====================

def _risk_from_fake_probability(value: float) -> str:
    if value >= 0.5:
        return "高"
    if value <= 0.2:
        return "低"
    return "中"


def _prediction_text(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "fake":
        return "伪造（AI生成）"
    if normalized == "real":
        return "真实"
    if "伪造" in str(value) or "fake" in normalized:
        return "伪造（AI生成）"
    return "真实"


def _format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes > 0:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


def _parse_risk(text: str) -> str:
    match = re.search(r"\[?伪造（?AI生成）?风险\]?\s*[：:]\s*([高中低])", text)
    return match.group(1) if match else "未知"


def _compute_stats(arr: list[float]) -> dict:
    if not arr:
        return {}
    a = np.array(arr)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "max": float(np.max(a)),
        "max_idx": int(np.argmax(a)),
        "min": float(np.min(a)),
        "min_idx": int(np.argmin(a)),
    }


def _detect_anomalies(velocity: list[float], acceleration: list[float], frame_info: list[dict]) -> str:
    if not velocity or len(velocity) < 3:
        return ""

    v_stats = _compute_stats(velocity)
    a_stats = _compute_stats(acceleration)

    anomalies = []
    if v_stats and v_stats["max"] > v_stats["mean"] + 1.5 * v_stats["std"]:
        idx = v_stats["max_idx"]
        if idx < len(frame_info):
            info = frame_info[idx]
            time = info.get("actual_time") if info.get("actual_time") is not None else info.get("target_time", idx * 1.0)
            frame_num = info.get("frame_number") or info.get("output_index", idx + 1)
            anomalies.append(f"第{frame_num}帧（{time:.2f}s）速度值{v_stats['max']:.3f}与均值{v_stats['mean']:.3f}存在明显不一致")

    if a_stats and a_stats["max"] > a_stats["mean"] + 1.5 * a_stats["std"]:
        idx = a_stats["max_idx"]
        if idx < len(frame_info):
            info = frame_info[idx]
            time = info.get("actual_time") if info.get("actual_time") is not None else info.get("target_time", idx * 1.0)
            frame_num = info.get("frame_number") or info.get("output_index", idx + 1)
            anomalies.append(f"第{frame_num}帧（{time:.2f}s）加速度值{a_stats['max']:.3f}与均值{a_stats['mean']:.3f}存在明显不一致")

    if anomalies:
        return "；".join(anomalies)
    return "速度与加速度变化整体一致，未检测到明显时序异常"


def _analyze_lota_trend(lota_scores: list[float], is_fake: bool = False) -> tuple[str, str]:
    if not lota_scores:
        return "未知", ""

    mean_score = float(np.mean(lota_scores))
    std_score = float(np.std(lota_scores))

    if mean_score >= 0.7:
        base_desc = f"所有帧的 LOTA 分数均较高（均值 {mean_score:.2f}），表明每一帧的像素级噪声模式都存在异常"
    elif mean_score <= 0.3:
        base_desc = f"所有帧的 LOTA 分数均较低（均值 {mean_score:.2f}），表明每一帧的噪声模式都比较自然"
    else:
        if std_score > 0.2:
            base_desc = f"LOTA 分数在 {min(lota_scores):.2f}~{max(lota_scores):.2f} 之间波动，可能存在帧间不一致"
        else:
            base_desc = f"LOTA 分数处于中等水平（均值 {mean_score:.2f}），噪声模式模棱两可"

    if is_fake and mean_score <= 0.3:
        base_desc += "，主要判据为时序异常"
    elif not is_fake and mean_score >= 0.7:
        base_desc += "，主要判据为时序异常"

    return ("整体偏高" if mean_score >= 0.7 else "整体偏低" if mean_score <= 0.3 else "波动较大" if std_score > 0.2 else "中等等级"), base_desc


# ==================== 折线图绘制 ====================

def _render_trend_chart(
    velocity: list[float],
    acceleration: list[float],
    lota: list[float],
    frame_info: list[dict] | None = None,
    width: float = 6.0,
    height: float = 3.5,
    dpi: int = 150
) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            from matplotlib import font_manager
            installed = {font.name for font in font_manager.fontManager.ttflist}
            for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Songti SC", "PingFang SC"):
                if candidate in installed:
                    plt.rcParams["font.sans-serif"] = [candidate]
                    break
            plt.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        fig, ax1 = plt.subplots(figsize=(width, height), dpi=dpi)

        # X 轴：优先使用 frame_info 中的秒数
        if frame_info and len(frame_info) == len(velocity):
            x = [float(fi.get('target_time') or 0.0) for fi in frame_info]
            x_label = '时间 (秒)'
        else:
            x = list(range(1, len(velocity) + 1))
            x_label = '帧序号'

        color1 = '#E74C3C'
        color2 = '#3498DB'
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel('速度 / 加速度', fontsize=14, color='black')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        line1 = ax1.plot(x, velocity, color=color1, label='速度 L2', linewidth=2.2, marker='o', markersize=7)
        line2 = ax1.plot(x, acceleration, color=color2, label='加速度 L2', linewidth=2.2, marker='s', markersize=7)

        ax1.grid(alpha=0.25, linewidth=0.5)
        if len(x) > 1:
            margin = (max(x) - min(x)) * 0.05
            ax1.set_xlim(min(x) - margin, max(x) + margin)
        else:
            ax1.set_xlim(0.5, 1.5)

        ax2 = ax1.twinx()
        color3 = '#2ECC71'
        ax2.set_ylabel('LOTA 分数', fontsize=14, color=color3)
        ax2.tick_params(axis='y', labelcolor=color3, labelsize=12)
        ax2.set_ylim(0, 1)

        line3 = ax2.plot(x, lota, color=color3, label='LOTA 分数', linewidth=2.2, marker='^', markersize=7)

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=12, framealpha=0.8)

        fig.tight_layout(pad=0.4)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        return buf.getvalue()

    except Exception:
        return b""


# ==================== 图片工具 ====================

def _chart_flowable(png_bytes: bytes, max_width: float, max_height: float) -> RLImage:
    if not png_bytes:
        return RLImage(io.BytesIO(b""), width=10, height=10)

    from PIL import Image
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            w, h = img.size
        scale = min(max_width / w, max_height / h)
        return RLImage(io.BytesIO(png_bytes), width=w * scale, height=h * scale)
    except Exception:
        return RLImage(io.BytesIO(png_bytes), width=max_width, height=max_height)


def _image_flowable(b64_data: str, max_width: float, max_height: float) -> RLImage:
    if not b64_data:
        return RLImage(io.BytesIO(b""), width=10, height=10)

    try:
        if b64_data.startswith("data:image"):
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        from PIL import Image
        with Image.open(io.BytesIO(img_bytes)) as img:
            w, h = img.size
        scale = min(max_width / w, max_height / h)
        return RLImage(io.BytesIO(img_bytes), width=w * scale, height=h * scale)
    except Exception:
        return RLImage(io.BytesIO(b""), width=10, height=10)


def _image_flowable_fixed_width(b64_data: str, fixed_width: float) -> RLImage:
    """固定水平宽度，按原图比例计算高度（不强制正方形）。"""
    if not b64_data:
        return RLImage(io.BytesIO(b""), width=10, height=10)

    try:
        if b64_data.startswith("data:image"):
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        from PIL import Image
        with Image.open(io.BytesIO(img_bytes)) as img:
            w, h = img.size
        if w <= 0:
            return RLImage(io.BytesIO(img_bytes), width=fixed_width, height=fixed_width * 0.75)
        scale = fixed_width / w
        return RLImage(io.BytesIO(img_bytes), width=fixed_width, height=h * scale)
    except Exception:
        return RLImage(io.BytesIO(b""), width=10, height=10)


def _empty_image(width: float, height: float) -> RLImage:
    return RLImage(io.BytesIO(b""), width=width, height=height)


# ==================== 豆包 API ====================

def _doubao_video_prompt(item: VideoReportItem) -> str:
    probability_text = _format_percent(item.fake_probability)
    risk = item.risk

    frame_lines = ""
    if item.frame_info and item.velocity_l2 and item.acceleration_l2 and item.lota_scores:
        for i in range(len(item.frame_info)):
            info = item.frame_info[i]
            frame_num = info.get("frame_number") or info.get("output_index", i + 1)
            time = info.get("actual_time") if info.get("actual_time") is not None else info.get("target_time", i * 1.0)
            frame_lines += (
                f"  帧{frame_num}（{time:.2f}s）: "
                f"速度={item.velocity_l2[i]:.3f}, "
                f"加速度={item.acceleration_l2[i]:.3f}, "
                f"LOTA={item.lota_scores[i]:.3f}\n"
            )

    lota_trend, lota_desc = _analyze_lota_trend(item.lota_scores or [], item.prediction == "fake")
    anomaly_text = item.anomaly_text or "未检测到明显时序异常"

    return (
        "你是一个视频伪造检测的取证分析专家。以下是一段待检测视频的取证数据：\n\n"
        f"【检测结果】\n"
        f"- 伪造概率：{probability_text}\n"
        f"- 风险等级：{risk}\n\n"
        f"【时序一致性证据】\n"
        f"- 速度 L2 范数序列：{item.velocity_l2}\n"
        f"- 加速度 L2 范数序列：{item.acceleration_l2}\n"
        f"- LOTA 分数序列：{item.lota_scores}\n"
        f"- 对应帧信息：\n{frame_lines}\n"
        f"- 自动检测到的时序异常：{anomaly_text}\n\n"
        f"【特征含义】\n"
        "1. 速度 L2 范数：反映帧间运动强度，真实视频变化平滑，AI视频易出现突变或过度平滑。\n"
        "2. 加速度 L2 范数：反映运动变化率，真实视频受物理惯性约束，AI视频缺乏物理规律。\n"
        "3. LOTA 分数：基于低位平面噪声分析，分数高表示该帧像素级噪声模式可疑。\n"
        f"本次 LOTA 呈现【{lota_trend}】趋势。\n\n"
        f"【分析要求】\n"
        "按以下格式输出：\n\n"
        f"[伪造（AI生成）风险]：{risk}\n\n"
        "[取证解析]：\n"
        "1. 整体判断：结合伪造概率和时序特征给出总体判断。\n"
        "2. 时序异常定位：指出速度或加速度异常的帧。\n"
        f"3. LOTA 整体评估：{lota_desc}\n"
        "4. 结论：总结判定依据。"
    )


def _call_doubao(item: VideoReportItem, api_key: str) -> str:
    payload = {
        "model": DOUBAO_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _doubao_video_prompt(item)},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        f"{DOUBAO_BASE_URL}/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Doubao API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Doubao API request failed: {exc}") from exc

    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    parts: list[str] = []
    for output in data.get("output", []) or []:
        for content in output.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts).strip() if parts else json.dumps(data, ensure_ascii=False)[:3000]


def _analyze_with_doubao(items: list[VideoReportItem]) -> None:
    api_key = _read_api_key()
    workers = max(1, min(MAX_DOUBAO_WORKERS, len(items)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(_call_doubao, item, api_key): item for item in items}
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            raw = future.result()
            item.analysis = raw.strip()
            item.risk = _parse_risk(item.analysis)


def _offline_analysis(item: VideoReportItem) -> str:
    risk = item.risk
    probability_text = _format_percent(item.fake_probability)

    anomaly_summary = ""
    if item.anomaly_text and item.anomaly_text != "速度与加速度变化整体一致，未检测到明显时序异常":
        anomaly_summary = f"检测到异常：{item.anomaly_text}。"

    lota_trend, lota_desc = _analyze_lota_trend(item.lota_scores or [], item.prediction == "fake")

    return (
        f"伪造（AI生成）风险：{risk}。"
        f"FakeTrace 给出的伪造概率为 {probability_text}。"
        f"{anomaly_summary}"
        f"LOTA 分数呈现【{lota_trend}】趋势：{lota_desc}。"
        "本报告未启用大模型解析。"
    )


# ==================== PDF 表格 ====================

def _video_info_table(items: list[VideoReportItem]) -> Table:
    data = [[
        _p("序号", SMALL_CENTER_STYLE),
        _p("文件名", SMALL_CENTER_STYLE),
        _p("时长", SMALL_CENTER_STYLE),
        _p("分辨率", SMALL_CENTER_STYLE),
        _p("帧率", SMALL_CENTER_STYLE),
        _p("总帧数", SMALL_CENTER_STYLE),
    ]]
    for item in items:
        data.append([
            _p(str(item.index), SMALL_CENTER_STYLE),
            _p(item.filename, SMALL_CENTER_STYLE),
            _p(_format_duration(item.duration) if item.duration else "未知", SMALL_CENTER_STYLE),
            _p(f"{item.width}×{item.height}" if item.width and item.height else "未知", SMALL_CENTER_STYLE),
            _p(f"{item.fps:.1f}" if item.fps else "未知", SMALL_CENTER_STYLE),
            _p(str(item.total_frames) if item.total_frames else "未知", SMALL_CENTER_STYLE),
        ])
    table = Table(data, colWidths=[13 * mm, 48 * mm, 28 * mm, 32 * mm, 22 * mm, 22 * mm])
    table.setStyle(_three_line_table_style())
    return table


def _summary_table(items: list[VideoReportItem]) -> Table:
    headers = ["序号", "文件名", "预测结果", "伪造概率", "伪造风险"]
    widths = [13 * mm, 68 * mm, 32 * mm, 30 * mm, 21 * mm]
    data = [[_p(header, SMALL_CENTER_STYLE) for header in headers]]
    for item in items:
        data.append([
            _p(str(item.index), SMALL_CENTER_STYLE),
            _p(item.filename, SUMMARY_FILENAME_STYLE),
            _p(_prediction_text(item.prediction), SMALL_CENTER_STYLE),
            _p(_format_percent(item.fake_probability), SMALL_CENTER_STYLE),
            _p(item.risk, SMALL_CENTER_STYLE),
        ])
    table = Table(data, colWidths=widths)
    table.setStyle(_three_line_table_style())
    return table


def _build_chart_and_suspicious(item: VideoReportItem) -> Table:
    chart_img = _chart_flowable(item.trend_chart_png, 78 * mm, 50 * mm) if item.trend_chart_png else _empty_image(78 * mm, 50 * mm)

    if item.suspicious_frame_b64:
        suspicious_img = _image_flowable(item.suspicious_frame_b64, 65 * mm, 50 * mm)
    else:
        suspicious_img = _empty_image(65 * mm, 50 * mm)

    if item.suspicious_frame_time is not None:
        suspicious_caption = f"图2：可疑帧（{item.suspicious_frame_time:.2f}s）"
    else:
        suspicious_caption = "图2：可疑帧"

    table = Table(
        [
            [chart_img, suspicious_img],
            [
                _p("图1：速度、加速度、LOTA 变化趋势", SMALL_CENTER_STYLE),
                _p(suspicious_caption, SMALL_CENTER_STYLE),
            ],
        ],
        colWidths=[78 * mm, 65 * mm],
        rowHeights=[50 * mm, 10 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _build_analysis_text(item: VideoReportItem) -> list:
    story = []
    if item.anomaly_text and item.anomaly_text != "速度与加速度变化整体一致，未检测到明显时序异常":
        anomaly_style = _style("AnomalyTextCN", 10, 14, 0)
        story.append(_p(f"📊 时序异常定位：{item.anomaly_text}", anomaly_style))
        story.append(Spacer(1, 2 * mm))

    if item.lota_scores:
        lota_trend, lota_desc = _analyze_lota_trend(item.lota_scores, item.prediction == "fake")
        lota_style = _style("LotaTrendCN", 10, 14, 0)
        story.append(_p(f"📊 LOTA 整体评估：{lota_desc}", lota_style))
        story.append(Spacer(1, 4 * mm))

    return story


# ==================== 主报告 ====================

def _build_story(
    items: list[VideoReportItem],
    model_name: str,
    test_id: str,
    upload_time: datetime,
    report_time: datetime,
    include_ai_analysis: bool,
) -> list[Any]:
    story: list[Any] = []
    upload_time_text = upload_time.strftime("%Y-%m-%d %H:%M:%S")
    report_time_text = report_time.strftime("%Y-%m-%d %H:%M:%S")

    story.append(Spacer(1, 12 * mm))
    story.append(_p(f"测试编号 {test_id} 视频伪造检测取证报告", TITLE_STYLE))
    story.append(Spacer(1, 14 * mm))

    # ===== 第1章：基本信息 =====
    story.append(_section_header(f"1. 测试编号 {test_id} 任务基本信息"))
    story.append(Spacer(1, 7 * mm))
    story.extend([
        _bullet_line("取证功能归类：视频伪造检测"),
        _bullet_line(f"待取证视频数量：{len(items)}"),
        _bullet_line(f"视频上传时间：{upload_time_text}"),
        _bullet_line(f"报告生成时间：{report_time_text}"),
        _bullet_line(f"模型选择：{model_name}"),
        # ===== 阈值在这里 =====
        _bullet_line(f"判定阈值：{items[0].threshold:.4f}"),
        _bullet_line(f"用以分析的大模型API版本：{DOUBAO_MODEL if include_ai_analysis else '未启用'}"),
        _bullet_line("视频信息："),
        Spacer(1, 3 * mm),
        _keep_table_together(_video_info_table(items)),
        Spacer(1, 12 * mm),
    ])

    # ===== 第2章：取证结果 =====
    story.append(_section_header("2. 取证结果"))
    story.append(Spacer(1, 6 * mm))

    for item in items:
        # ===== 简化：只显示得分和阈值 =====
        result_text = (
            f"预测结果：{_prediction_text(item.prediction)}　｜　"
            f"得分：{item.fake_probability:.4f}（阈值：{item.threshold:.4f}）"
        )
        story.append(
            KeepTogether([
                _p(f"（{item.index}）序号{item.index}: 视频 {item.filename} 取证结果", BODY_STYLE),
                Spacer(1, 4 * mm),
                _p(result_text, RESULT_TEXT_STYLE),
                Spacer(1, 5 * mm),

                _build_chart_and_suspicious(item),
                Spacer(1, 4 * mm),

                *_build_analysis_text(item),
                Spacer(1, 7 * mm),
            ])
        )

    # ===== 第3章 =====
    _append_analysis_section(
        story,
        "3. 取证结果解析",
        [
            (
                f"（{item.index}）序号{item.index}: 视频 {item.filename} 取证结果解析",
                [item.analysis],
            )
            for item in items
        ],
    )

    # ===== 第4章 =====
    story.append(_section_header("4. 总结"))
    story.append(Spacer(1, 10 * mm))
    story.append(
        _p(
            f"经 FakeTrace 判定，用户上传的 {len(items)} 个待取证视频的检测结果如下。"
            "请注意，被判定为高风险伪造的视频，请您谨慎使用和传播，以免造成不良影响。",
            BODY_STYLE,
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(_keep_table_together(_summary_table(items)))
    return story


# ==================== 主入口 ====================

def build_video_report(payload: dict[str, Any]) -> GeneratedReport:
    model_key = str(payload.get("model") or "").lower()
    model_name = payload.get("model_name") or "TRI"
    test_id = str(payload.get("test_id") or "V-000001").strip() or "V-000001"
    include_ai_analysis = bool(payload.get("include_ai_analysis"))
    upload_time = _parse_upload_time(payload.get("upload_time"))

    raw_items = payload.get("items") or []
    if not raw_items:
        raise ValueError("No video report items provided.")

    items: list[VideoReportItem] = []
    for index, raw in enumerate(raw_items, start=1):
        fake_probability = float(raw.get("fake_probability") or 0.0)
        real_probability = float(raw.get("real_probability") or (1.0 - fake_probability))

        item = VideoReportItem(
            index=index,
            filename=str(raw.get("filename") or f"video_{index}.mp4"),
            duration=float(raw.get("duration") or 0.0),
            width=int(raw.get("width") or 0),
            height=int(raw.get("height") or 0),
            fps=float(raw.get("fps") or 0.0),
            total_frames=int(raw.get("total_frames") or 0),
            prediction=str(raw.get("prediction") or ("fake" if fake_probability >= 0.5 else "real")),
            fake_probability=max(0.0, min(1.0, fake_probability)),
            real_probability=max(0.0, min(1.0, real_probability)),
            threshold=float(raw.get("threshold") or 0.5),
            frame_info=raw.get("frame_info"),
            velocity_l2=raw.get("velocity_l2"),
            acceleration_l2=raw.get("acceleration_l2"),
            lota_scores=raw.get("lota_scores"),
            suspicious_frame_b64=raw.get("suspicious_frame_b64"),
            suspicious_frame_time=raw.get("suspicious_frame_time"),
        )
        item.risk = _risk_from_fake_probability(item.fake_probability)

        if item.velocity_l2 and item.acceleration_l2 and item.frame_info:
            item.anomaly_text = _detect_anomalies(item.velocity_l2, item.acceleration_l2, item.frame_info)

        if item.velocity_l2 and item.acceleration_l2 and item.lota_scores:
            item.trend_chart_png = _render_trend_chart(
                item.velocity_l2,
                item.acceleration_l2,
                item.lota_scores,
                item.frame_info,
            )

        items.append(item)

    if include_ai_analysis:
        _analyze_with_doubao(items)
        for item in items:
            if item.risk == "未知":
                item.risk = _risk_from_fake_probability(item.fake_probability)
    else:
        for item in items:
            item.analysis = _offline_analysis(item)

    report_time = datetime.now().astimezone()
    report_id = uuid.uuid4().hex
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_DIR / f"video_report_{report_id}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(
        _build_story(items, model_name, test_id, upload_time, report_time, include_ai_analysis),
        onFirstPage=_draw_first_page,
        onLaterPages=_draw_later_page,
    )
    return GeneratedReport(report_id=report_id, path=output_path)


def resolve_video_report_path(report_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", report_id):
        raise FileNotFoundError("Invalid report id.")
    path = REPORT_OUTPUT_DIR / f"video_report_{report_id}.pdf"
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {report_id}")
    return path
