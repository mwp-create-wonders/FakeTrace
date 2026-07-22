from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
    _data_url_size_kb,
    _decode_data_url,
    _draw_first_page,
    _draw_later_page,
    _extension_from_data_url,
    _format_percent,
    _image_flowable,
    _image_size,
    _keep_table_together,
    _p,
    _parse_upload_time,
    _read_api_key,
    _section_header,
    _split_analysis,
    _style,
    _three_line_table_style,
)


REPORT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"

MODEL_REPORT_NAMES = {
    "mf2da": "检测方法1（MF2DA）",
    "marc": "检测方法2（MARC）",
    "forensic_moe": "检测方法3（Forensic-MoE）",
    "forgelens": "检测方法4（ForgeLens）",
    "lota": "检测方法5（LOTA）",
    "univfd": "检测方法6（UnivFD）",
}

DETECTOR_HEADER_STYLE = _style("DetectorHeaderCN", 12.5, 18, TA_CENTER)
DETECTOR_VALUE_STYLE = _style("DetectorValueCN", 12.5, 18, TA_CENTER)
SUMMARY_FILENAME_STYLE = _style("SummaryFilenameCN", 9.6, 12, TA_CENTER)


@dataclass
class DetectorReportImage:
    index: int
    filename: str
    original_url: str
    prediction: str
    fake_probability: float
    real_probability: float
    risk: str = "未知"
    analysis: str = ""


@dataclass
class GeneratedReport:
    report_id: str
    path: Path


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


def _parse_risk(text: str) -> str:
    match = re.search(r"\[?伪造（?AI生成）?风险\]?\s*[：:]\s*([高中低])", text)
    return match.group(1) if match else "未知"


def _doubao_prompt(item: DetectorReportImage) -> str:
    probability_text = _format_percent(item.fake_probability)
    risk = _risk_from_fake_probability(item.fake_probability)
    return (
        "这张图是待取证的原始图片。FakeTrace 图像伪造检测系统已经给出了该图片的"
        f"伪造概率 fake_probability={probability_text}。请基于原图内容和该检测概率进行取证解析。"
        "风险等级规则必须遵循：伪造概率大于等于 50% 为高风险，小于等于 20% 为低风险，"
        "其余为中风险。"
        f"因此本次风险等级应为：{risk}。请严格按照以下格式输出：\n"
        f"[伪造（AI生成）风险]：{risk}\n"
        "[取证解析]：结合原图可见内容、AI生成图像常见痕迹以及 FakeTrace 给出的伪造概率，"
        "说明该图片为什么属于上述风险等级。"
    )


def _call_doubao(item: DetectorReportImage, api_key: str) -> str:
    payload = {
        "model": DOUBAO_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": item.original_url},
                    {"type": "input_text", "text": _doubao_prompt(item)},
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


def _analyze_with_doubao(items: list[DetectorReportImage]) -> None:
    api_key = _read_api_key()
    workers = max(1, min(MAX_DOUBAO_WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(_call_doubao, item, api_key): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            item.analysis = re.sub(r"\n{3,}", "\n\n", future.result().strip())
            item.risk = _parse_risk(item.analysis)


def _image_info_table(items: list[DetectorReportImage]) -> Table:
    data = [[
        _p("序号", SMALL_CENTER_STYLE),
        _p("文件名", SMALL_CENTER_STYLE),
        _p("文件格式", SMALL_CENTER_STYLE),
        _p("分辨率", SMALL_CENTER_STYLE),
        _p("文件大小", SMALL_CENTER_STYLE),
    ]]
    for item in items:
        width, height = _image_size(item.original_url)
        data.append([
            _p(str(item.index), SMALL_CENTER_STYLE),
            _p(item.filename, SMALL_CENTER_STYLE),
            _p(_extension_from_data_url(item.original_url, item.filename), SMALL_CENTER_STYLE),
            _p(f"{width}*{height}", SMALL_CENTER_STYLE),
            _p(_data_url_size_kb(item.original_url), SMALL_CENTER_STYLE),
        ])
    table = Table(data, colWidths=[22 * mm, 58 * mm, 32 * mm, 32 * mm, 26 * mm])
    table.setStyle(_three_line_table_style())
    return table


def _detector_result_table(item: DetectorReportImage) -> Table:
    image_width = 38 * mm
    image_height = 52 * mm
    col_width = 42.5 * mm
    table = Table(
        [
            [
                _p("原图", DETECTOR_HEADER_STYLE),
                _p("预测结果", DETECTOR_HEADER_STYLE),
                _p("伪造概率", DETECTOR_HEADER_STYLE),
                _p("真实概率", DETECTOR_HEADER_STYLE),
            ],
            [
                _image_flowable(item.original_url, image_width, image_height),
                _p(_prediction_text(item.prediction), DETECTOR_VALUE_STYLE),
                _p(_format_percent(item.fake_probability), DETECTOR_VALUE_STYLE),
                _p(_format_percent(item.real_probability), DETECTOR_VALUE_STYLE),
            ],
        ],
        colWidths=[col_width] * 4,
        rowHeights=[13 * mm, 58 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _result_text(item: DetectorReportImage) -> str:
    return (
        f"经 FakeTrace 取证系统检测，该图片预测结果为{_prediction_text(item.prediction)}，"
        f"伪造概率为 {_format_percent(item.fake_probability)}，"
        f"真实概率为 {_format_percent(item.real_probability)}，"
        f"对应伪造风险为{item.risk}。"
    )


def _summary_table(items: list[DetectorReportImage]) -> Table:
    headers = ["序号", "文件名", "预测结果", "伪造概率", "真实概率", "伪造风险"]
    widths = [13 * mm, 58 * mm, 32 * mm, 23 * mm, 23 * mm, 21 * mm]
    data: list[list[Paragraph]] = [[_p(header, SMALL_CENTER_STYLE) for header in headers]]
    for item in items:
        data.append([
            _p(str(item.index), SMALL_CENTER_STYLE),
            _p(item.filename, SUMMARY_FILENAME_STYLE),
            _p(_prediction_text(item.prediction), SMALL_CENTER_STYLE),
            _p(_format_percent(item.fake_probability), SMALL_CENTER_STYLE),
            _p(_format_percent(item.real_probability), SMALL_CENTER_STYLE),
            _p(item.risk, SMALL_CENTER_STYLE),
        ])
    table = Table(data, colWidths=widths)
    table.setStyle(_three_line_table_style())
    return table


def _build_story(
    items: list[DetectorReportImage],
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
    story.append(_p(f"测试编号 {test_id} 伪造检测取证报告", TITLE_STYLE))
    story.append(Spacer(1, 14 * mm))
    story.append(_section_header(f"1.  测试编号 {test_id} 任务基本信息"))
    story.append(Spacer(1, 7 * mm))
    story.extend(
        [
            _bullet_line("取证功能归类：  伪造检测"),
            _bullet_line(f"待取证图片数量：  {len(items)}"),
            _bullet_line(f"图片上传时间：  {upload_time_text}"),
            _bullet_line(f"报告生成时间：  {report_time_text}"),
            _bullet_line(f"模型选择：  {model_name}"),
            _bullet_line(f"用以分析的大模型API版本：  {DOUBAO_MODEL if include_ai_analysis else '未启用'}"),
            _bullet_line("图片信息："),
            Spacer(1, 3 * mm),
            _keep_table_together(_image_info_table(items)),
            Spacer(1, 12 * mm),
            _section_header("2.  取证结果"),
            Spacer(1, 6 * mm),
        ]
    )

    for item in items:
        story.append(
            KeepTogether(
                [
                    _p(f"（{item.index}） 序号{item.index}: 图片 {item.filename} 取证结果", BODY_STYLE),
                    Spacer(1, 4 * mm),
                    _detector_result_table(item),
                    Spacer(1, 5 * mm),
                    _p(_result_text(item), BODY_STYLE),
                    Spacer(1, 7 * mm),
                ]
            )
        )

    if include_ai_analysis:
        _append_analysis_section(
            story,
            "3.  取证结果解析",
            [
                (
                    f"（{item.index}） 序号{item.index}: 图片 {item.filename} 取证结果解析",
                    _split_analysis(item.analysis),
                )
                for item in items
            ],
        )

    story.append(_section_header("4.  总结"))
    story.append(Spacer(1, 10 * mm))
    story.append(
        _p(
            f"经 FakeTrace 判定，用户上传的 {len(items)} 张待取证图片的检测结果如下。"
            "请注意，被判定为高风险伪造的图片，请您谨慎使用和传播，以免造成不良影响。",
            BODY_STYLE,
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(_keep_table_together(_summary_table(items)))
    return story


def build_detector_report(payload: dict[str, Any]) -> GeneratedReport:
    model_key = str(payload.get("model") or "").lower()
    model_name = MODEL_REPORT_NAMES.get(model_key, str(payload.get("model_name") or "检测模型"))
    test_id = str(payload.get("test_id") or "D-000001").strip() or "D-000001"
    include_ai_analysis = bool(payload.get("include_ai_analysis"))
    upload_time = _parse_upload_time(payload.get("upload_time"))

    raw_items = payload.get("items") or []
    if not raw_items:
        raise ValueError("No detector report items provided.")

    items: list[DetectorReportImage] = []
    for index, raw in enumerate(raw_items, start=1):
        fake_probability = float(raw.get("fake_probability") or 0.0)
        real_probability = float(raw.get("real_probability") or (1.0 - fake_probability))
        item = DetectorReportImage(
            index=index,
            filename=str(raw.get("filename") or f"image_{index}.png"),
            original_url=str(raw.get("original_image_url") or ""),
            prediction=str(raw.get("prediction") or ("fake" if fake_probability >= 0.5 else "real")),
            fake_probability=max(0.0, min(1.0, fake_probability)),
            real_probability=max(0.0, min(1.0, real_probability)),
        )
        _decode_data_url(item.original_url)
        item.risk = _risk_from_fake_probability(item.fake_probability)
        items.append(item)

    if include_ai_analysis:
        _analyze_with_doubao(items)
        for item in items:
            if item.risk == "未知":
                item.risk = _risk_from_fake_probability(item.fake_probability)
    else:
        for item in items:
            item.analysis = (
                f"[伪造（AI生成）风险]：{item.risk}\n"
                f"[取证解析]：FakeTrace 给出的伪造概率为 {_format_percent(item.fake_probability)}，"
                "本报告未启用大模型解析。"
            )

    report_time = datetime.now().astimezone()
    report_id = uuid.uuid4().hex
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_DIR / f"detector_report_{report_id}.pdf"
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


def resolve_detector_report_path(report_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", report_id):
        raise FileNotFoundError("Invalid report id.")
    path = REPORT_OUTPUT_DIR / f"detector_report_{report_id}.pdf"
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {report_id}")
    return path
