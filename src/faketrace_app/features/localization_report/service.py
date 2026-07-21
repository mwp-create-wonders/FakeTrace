from __future__ import annotations

import ast
import base64
import io
import json
import mimetypes
import os
import re
import textwrap
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ...core.paths import PROJECT_ROOT


REPORT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = "doubao-seed-2-0-pro-260215"
REQUEST_TIMEOUT_SECONDS = 120
MAX_DOUBAO_WORKERS = 4

MODEL_REPORT_NAMES = {
    "effunetpp": "定位方法1（EffUNet）",
    "fassa": "定位方法2（Fassa）",
    "trufor": "定位方法3（TruFor）",
    "catnet": "定位方法4（CAT-Net）",
}

DOUBAO_PROMPT = (
    "第一张图是待取证的原始图片，第二张图是我们设计的取证系统定位出的疑似篡改区域。"
    "偏红的代表疑似被 AI 局部伪造过的，请重点关注偏红色的区域。"
    "一旦出现大面积偏红的区域，立即认为属于高风险伪造。请你按照以下格式进行输出：\n"
    "[伪造风险]：高 / 中 / 低\n"
    "[伪造区域分析]：根据我们的系统输出的热力图，分析原图，指出为什么偏红的区域可能是被伪造过的。"
)


@dataclass
class ReportImage:
    index: int
    filename: str
    original_url: str
    localization_map_url: str
    overlay_url: str
    suspicious_ratio: float
    score: float | None = None
    confidence_map_url: str | None = None
    extra_fields: dict[str, Any] | None = None
    risk: str = "未知"
    analysis: str = ""


@dataclass
class GeneratedReport:
    report_id: str
    path: Path


def _register_fonts() -> tuple[str, str, str, str]:
    songti_path = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
    times_path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    if songti_path.exists() and times_path.exists():
        pdfmetrics.registerFont(TTFont("FakeTraceSongtiBold", str(songti_path), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("FakeTraceSongti", str(songti_path), subfontIndex=4))
        pdfmetrics.registerFont(TTFont("FakeTraceTimes", str(times_path)))
        header_cn_font = "FakeTraceSongti"
        for fang_song_path in (
            Path("/System/Library/Fonts/Supplemental/STFangsong.ttf"),
            Path("/System/Library/Fonts/Supplemental/FangSong.ttf"),
            Path("/Library/Fonts/STFangsong.ttf"),
            Path("/Library/Fonts/FangSong.ttf"),
            Path("C:/Windows/Fonts/simfang.ttf"),
        ):
            if fang_song_path.exists():
                pdfmetrics.registerFont(TTFont("FakeTraceFangSong", str(fang_song_path)))
                header_cn_font = "FakeTraceFangSong"
                break
        return "FakeTraceSongti", "FakeTraceSongtiBold", "FakeTraceTimes", header_cn_font

    # Windows keeps the same Songti/Hei/Times pairing using the system CJK fonts.
    windows_songti_path = Path("C:/Windows/Fonts/simsun.ttc")
    if windows_songti_path.exists():
        pdfmetrics.registerFont(TTFont("FakeTraceSongti", str(windows_songti_path), subfontIndex=0))

        bold_path = Path("C:/Windows/Fonts/simhei.ttf")
        if bold_path.exists():
            pdfmetrics.registerFont(TTFont("FakeTraceSongtiBold", str(bold_path)))
            bold_font = "FakeTraceSongtiBold"
        else:
            bold_font = "FakeTraceSongti"

        windows_times_path = Path("C:/Windows/Fonts/times.ttf")
        if windows_times_path.exists():
            pdfmetrics.registerFont(TTFont("FakeTraceTimes", str(windows_times_path)))
            times_font = "FakeTraceTimes"
        else:
            times_font = "Times-Roman"

        header_cn_font = "FakeTraceSongti"
        fang_song_path = Path("C:/Windows/Fonts/simfang.ttf")
        if fang_song_path.exists():
            pdfmetrics.registerFont(TTFont("FakeTraceFangSong", str(fang_song_path)))
            header_cn_font = "FakeTraceFangSong"
        return "FakeTraceSongti", bold_font, times_font, header_cn_font

    # Fallback keeps PDF generation available on Linux servers. It will not be
    # as close to the sample macOS/WPS typography as the Songti/Times setup.
    try:
        pdfmetrics.registerFont(TTFont("FakeTraceDejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
        return "FakeTraceDejaVu", "FakeTraceDejaVu", "FakeTraceDejaVu", "FakeTraceDejaVu"
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("No usable PDF font found. Install Songti/Times or DejaVu fonts.") from exc


FONT_CN, FONT_CN_BOLD, FONT_EN, FONT_HEADER_CN = _register_fonts()
ASCII_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/\\\-+%]*")


def _style(name: str, size: float, leading: float, alignment: int = TA_LEFT, **kwargs: Any) -> ParagraphStyle:
    return ParagraphStyle(
        name=name,
        fontName=kwargs.pop("fontName", FONT_CN),
        fontSize=size,
        leading=leading,
        alignment=alignment,
        wordWrap="CJK",
        splitLongWords=True,
        **kwargs,
    )


TITLE_STYLE = _style("TitleCN", 22, 30, TA_CENTER, fontName=FONT_CN_BOLD, textColor=colors.black)
SECTION_STYLE = _style("SectionCN", 15, 18, fontName=FONT_CN_BOLD, textColor=colors.HexColor("#243f63"))
BODY_STYLE = _style("BodyCN", 12.5, 24)
BODY_INDENT_STYLE = _style("BodyIndentCN", 12.5, 24, leftIndent=8 * mm)
SMALL_CENTER_STYLE = _style("SmallCenterCN", 11.5, 15, TA_CENTER)
PAGE_HEADER_TEXT = "FakeTrace 多模态数字内容取证系统"
SECTION_HEADER_FILL = colors.HexColor("#c5d9f1")
SECTION_HEADER_LINE = colors.HexColor("#315f9a")


def _rich_text(text: Any) -> str:
    raw = str(text)
    pieces: list[str] = []
    cursor = 0
    for match in ASCII_RUN_RE.finditer(raw):
        if match.start() > cursor:
            pieces.append(escape(raw[cursor:match.start()]))
        pieces.append(f'<font name="{FONT_EN}">{escape(match.group(0))}</font>')
        cursor = match.end()
    if cursor < len(raw):
        pieces.append(escape(raw[cursor:]))
    return "".join(pieces)


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_rich_text(text), style)


class SectionHeader(Flowable):
    def __init__(self, title: str, width: float = 170 * mm, height: float = 14.5 * mm):
        super().__init__()
        self.title = title
        self.width = width
        self.height = height

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        canvas.saveState()
        canvas.setFillColor(SECTION_HEADER_FILL)
        canvas.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        canvas.setStrokeColor(SECTION_HEADER_LINE)
        canvas.setLineWidth(2)
        canvas.line(0, 0, self.width, 0)
        canvas.restoreState()

        paragraph = _p(self.title, SECTION_STYLE)
        _, paragraph_height = paragraph.wrap(self.width - 20, self.height)
        paragraph.drawOn(canvas, 10, (self.height - paragraph_height) / 2 + 0.5)


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("Expected image data URL.")
    header, encoded = data_url.split(",", 1)
    mime = header[5:].split(";", 1)[0] or "image/png"
    return base64.b64decode(encoded), mime


def _image_size(data_url: str) -> tuple[int, int]:
    image_bytes, _ = _decode_data_url(data_url)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.width, image.height


def _data_url_size_kb(data_url: str) -> str:
    image_bytes, _ = _decode_data_url(data_url)
    return f"{max(1, round(len(image_bytes) / 1024))}KB"


def _extension_from_data_url(data_url: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix:
        return "jpg" if suffix == "jpeg" else suffix
    _, mime = _decode_data_url(data_url)
    guessed = mimetypes.guess_extension(mime) or ".png"
    return guessed.lstrip(".")


def _image_flowable(data_url: str, max_width: float, max_height: float) -> RLImage:
    image_bytes, _ = _decode_data_url(data_url)
    with Image.open(io.BytesIO(image_bytes)) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return RLImage(io.BytesIO(image_bytes), width=width * scale, height=height * scale)


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "未提供"


def _risk_from_ratio(value: float) -> str:
    if value >= 0.10:
        return "高"
    if value <= 0.05:
        return "低"
    return "中"


def _parse_risk(text: str) -> str:
    match = re.search(r"\[?伪造风险\]?\s*[：:]\s*([高中低])", text)
    return match.group(1) if match else "未知"


def _split_analysis(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ["[伪造风险]：未知", "[伪造区域分析]：大模型未返回有效解析。"]
    paragraphs: list[str] = []
    for line in lines:
        if len(line) <= 64:
            paragraphs.append(line)
        else:
            paragraphs.extend(textwrap.wrap(line, width=64, break_long_words=False, replace_whitespace=False))
    return paragraphs


def _read_api_key() -> str:
    env_key = os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY")
    if env_key:
        return env_key.strip()

    # Development fallback used by the existing try_report experiment.
    sample_path = PROJECT_ROOT / "try_report" / "API.py"
    if sample_path.exists():
        tree = ast.parse(sample_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "api_key":
                        value = ast.literal_eval(node.value)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
    raise RuntimeError("Doubao API key is not configured. Set DOUBAO_API_KEY or ARK_API_KEY.")


def _call_doubao(item: ReportImage, api_key: str) -> str:
    payload = {
        "model": DOUBAO_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": item.original_url},
                    {"type": "input_image", "image_url": item.localization_map_url},
                    {"type": "input_text", "text": DOUBAO_PROMPT},
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


def _analyze_with_doubao(items: list[ReportImage]) -> None:
    api_key = _read_api_key()
    workers = max(1, min(MAX_DOUBAO_WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(_call_doubao, item, api_key): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            item.analysis = re.sub(r"\n{3,}", "\n\n", future.result().strip())
            item.risk = _parse_risk(item.analysis)


def _section_header(title: str) -> SectionHeader:
    return SectionHeader(title)


def _draw_page_number(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFont(FONT_CN, 10)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 9 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def _draw_mixed_font_text(canvas: Canvas, text: str, x: float, y: float, size: float) -> None:
    cursor_x = x
    cursor = 0
    for match in ASCII_RUN_RE.finditer(text):
        if match.start() > cursor:
            segment = text[cursor:match.start()]
            canvas.setFont(FONT_HEADER_CN, size)
            canvas.drawString(cursor_x, y, segment)
            cursor_x += canvas.stringWidth(segment, FONT_HEADER_CN, size)
        segment = match.group(0)
        canvas.setFont(FONT_EN, size)
        canvas.drawString(cursor_x, y, segment)
        cursor_x += canvas.stringWidth(segment, FONT_EN, size)
        cursor = match.end()
    if cursor < len(text):
        segment = text[cursor:]
        canvas.setFont(FONT_HEADER_CN, size)
        canvas.drawString(cursor_x, y, segment)


def _draw_page_header(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#333333"))
    _draw_mixed_font_text(canvas, PAGE_HEADER_TEXT, doc.leftMargin, A4[1] - 11 * mm, 12)
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(0.45)
    canvas.line(doc.leftMargin, A4[1] - 13.2 * mm, A4[0] - doc.rightMargin, A4[1] - 13.2 * mm)
    canvas.restoreState()


def _draw_first_page(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    _draw_page_header(canvas, doc)
    _draw_page_number(canvas, doc)


def _draw_later_page(canvas: Canvas, doc: SimpleDocTemplate) -> None:
    _draw_page_header(canvas, doc)
    _draw_page_number(canvas, doc)


def _bullet_line(text: str) -> Table:
    table = Table([[_p("●", BODY_STYLE), _p(text, BODY_STYLE)]], colWidths=[10 * mm, 150 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    return table


def _three_line_table_style() -> TableStyle:
    return TableStyle(
        [
            ("LINEABOVE", (0, 0), (-1, 0), 1.4, colors.black),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 1.4, colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def _image_info_table(items: list[ReportImage]) -> Table:
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


def _image_table(item: ReportImage) -> Table:
    image_width = 54.5 * mm
    image_height = 54.5 * mm
    table = Table(
        [
            [
                _image_flowable(item.original_url, image_width, image_height),
                _image_flowable(item.localization_map_url, image_width, image_height),
                _image_flowable(item.overlay_url, image_width, image_height),
            ],
            [_p("原图", SMALL_CENTER_STYLE), _p("热力图", SMALL_CENTER_STYLE), _p("叠加图", SMALL_CENTER_STYLE)],
        ],
        colWidths=[image_width] * 3,
        rowHeights=[image_height, 11 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
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


def _has_score(items: list[ReportImage]) -> bool:
    return any(item.score is not None for item in items)


def _has_confidence(items: list[ReportImage]) -> bool:
    return any(item.confidence_map_url for item in items)


def _summary_table(items: list[ReportImage]) -> Table:
    include_score = _has_score(items)
    include_confidence = _has_confidence(items)
    headers = ["序号", "文件名", "伪造区域占比"]
    widths = [20 * mm, 58 * mm, 34 * mm]
    if include_score:
        headers.append("整体得分")
        widths.append(28 * mm)
    if include_confidence:
        headers.append("置信度")
        widths.append(24 * mm)
    headers.append("伪造风险")
    widths.append(26 * mm)

    data = [[_p(header, SMALL_CENTER_STYLE) for header in headers]]
    for item in items:
        row = [_p(str(item.index), SMALL_CENTER_STYLE), _p(item.filename, SMALL_CENTER_STYLE), _p(_format_percent(item.suspicious_ratio), SMALL_CENTER_STYLE)]
        if include_score:
            row.append(_p(_format_percent(item.score) if item.score is not None else "未提供", SMALL_CENTER_STYLE))
        if include_confidence:
            row.append(_p("已提供" if item.confidence_map_url else "未提供", SMALL_CENTER_STYLE))
        row.append(_p(item.risk, SMALL_CENTER_STYLE))
        data.append(row)

    table = Table(data, colWidths=widths)
    table.setStyle(_three_line_table_style())
    return table


def _result_text(item: ReportImage) -> str:
    parts = [
        f"经 FakeTrace 取证系统的判断，该图片伪造区域占比为 {_format_percent(item.suspicious_ratio)}。"
        "具体的伪造区域定位您可以查看上方的热力图，热力图中颜色偏红的区域为疑似伪造区域，颜色偏蓝的区域为真实区域。"
    ]
    if item.score is not None:
        parts.append(f"模型同时返回整体篡改得分 {_format_percent(item.score)}。")
    if item.confidence_map_url:
        parts.append("模型同时返回置信图，可用于辅助观察定位结果的可信程度。")
    return "".join(parts)


def _parse_upload_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.astimezone()
        return parsed.astimezone()
    except ValueError:
        return datetime.now().astimezone()


def _build_story(
    items: list[ReportImage],
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
    story.append(_p(f"测试编号 {test_id} 伪造定位取证报告", TITLE_STYLE))
    story.append(Spacer(1, 14 * mm))
    story.append(_section_header(f"1.  测试编号 {test_id} 任务基本信息"))
    story.append(Spacer(1, 7 * mm))
    story.extend(
        [
            _bullet_line("取证功能归类：  伪造定位"),
            _bullet_line(f"待取证图片数量：  {len(items)}"),
            _bullet_line(f"图片上传时间：  {upload_time_text}"),
            _bullet_line(f"报告生成时间：  {report_time_text}"),
            _bullet_line(f"模型选择：  {model_name}"),
            _bullet_line(f"用以分析的大模型API版本：  {DOUBAO_MODEL if include_ai_analysis else '未启用'}"),
            _bullet_line("图片信息："),
            Spacer(1, 3 * mm),
            _image_info_table(items),
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
                    _image_table(item),
                    Spacer(1, 5 * mm),
                    _p(_result_text(item), BODY_STYLE),
                    Spacer(1, 7 * mm),
                ]
            )
        )

    if include_ai_analysis:
        story.append(PageBreak())
        story.append(_section_header("3.  取证结果分析"))
        story.append(Spacer(1, 7 * mm))
        for item in items:
            story.append(_p(f"（{item.index}） 序号{item.index}: 图片 {item.filename} 取证结果分析", BODY_STYLE))
            story.append(Spacer(1, 3 * mm))
            for paragraph in _split_analysis(item.analysis):
                story.append(_p(paragraph, BODY_INDENT_STYLE))
                story.append(Spacer(1, 1.5 * mm))
            story.append(Spacer(1, 6 * mm))

    story.append(_section_header("4.  总结"))
    story.append(Spacer(1, 10 * mm))
    story.append(
        _p(
            f"经 FakeTrace 判定，用户上传的 {len(items)} 张待取证图片的取证结果如下。"
            "请注意，被判定为高风险伪造的图片，请您谨慎使用和传播，以免造成不良影响。",
            BODY_STYLE,
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(_summary_table(items))
    return story


def build_localization_report(payload: dict[str, Any]) -> GeneratedReport:
    model_key = str(payload.get("model") or "").lower()
    model_name = MODEL_REPORT_NAMES.get(model_key, str(payload.get("model_name") or "定位模型"))
    test_id = str(payload.get("test_id") or "L-000001").strip() or "L-000001"
    include_ai_analysis = bool(payload.get("include_ai_analysis"))
    upload_time = _parse_upload_time(payload.get("upload_time"))

    raw_items = payload.get("items") or []
    if not raw_items:
        raise ValueError("No localization report items provided.")

    items: list[ReportImage] = []
    for index, raw in enumerate(raw_items, start=1):
        item = ReportImage(
            index=index,
            filename=str(raw.get("filename") or f"image_{index}.png"),
            original_url=str(raw.get("original_image_url") or ""),
            localization_map_url=str(raw.get("localization_map_url") or ""),
            overlay_url=str(raw.get("overlay_url") or ""),
            suspicious_ratio=float(raw.get("suspicious_ratio") or 0.0),
            score=float(raw["score"]) if raw.get("score") is not None else None,
            confidence_map_url=raw.get("confidence_map_url"),
            extra_fields=raw.get("extra_fields") if isinstance(raw.get("extra_fields"), dict) else None,
        )
        _decode_data_url(item.original_url)
        _decode_data_url(item.localization_map_url)
        _decode_data_url(item.overlay_url)
        if item.confidence_map_url:
            _decode_data_url(item.confidence_map_url)
        items.append(item)

    if include_ai_analysis:
        _analyze_with_doubao(items)
        for item in items:
            if item.risk == "未知":
                item.risk = _risk_from_ratio(item.suspicious_ratio)
    else:
        for item in items:
            item.risk = _risk_from_ratio(item.suspicious_ratio)

    report_time = datetime.now().astimezone()
    report_id = uuid.uuid4().hex
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_OUTPUT_DIR / f"localization_report_{report_id}.pdf"
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


def resolve_report_path(report_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", report_id):
        raise FileNotFoundError("Invalid report id.")
    path = REPORT_OUTPUT_DIR / f"localization_report_{report_id}.pdf"
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {report_id}")
    return path
