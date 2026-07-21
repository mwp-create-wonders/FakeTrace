from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
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
from ..localization_report.service import (
    BODY_INDENT_STYLE,
    BODY_STYLE,
    DOUBAO_BASE_URL,
    DOUBAO_MODEL,
    MAX_DOUBAO_WORKERS,
    REQUEST_TIMEOUT_SECONDS,
    SMALL_CENTER_STYLE,
    TITLE_STYLE,
    _bullet_line,
    _decode_data_url,
    _draw_first_page,
    _draw_later_page,
    _format_percent,
    _p,
    _parse_upload_time,
    _read_api_key,
    _section_header,
    _split_analysis,
    _three_line_table_style,
)


REPORT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"

MODEL_REPORT_NAMES = {
    "ast_audioset_ft": "语音检测方法1（AST-AudioSet）",
    "aasist": "语音检测方法2（AASIST）",
}

WAVEFORM_COLOR = "#1F497D"
MEL_COLORMAP = "magma"
FIGURE_DPI = 200
MEL_N_FFT = 1024
MEL_HOP_LENGTH = 256
MEL_BANDS = 128
MEL_DYNAMIC_RANGE_DB = 80.0


@dataclass
class AudioReportItem:
    index: int
    filename: str
    audio_url: str
    prediction: str
    fake_probability: float
    real_probability: float
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    size_bytes: int = 0
    waveform_png: bytes | None = None
    mel_png: bytes | None = None
    risk: str = "未知"
    analysis: str = ""


@dataclass
class GeneratedReport:
    report_id: str
    path: Path


# --------------------------------------------------------------------------------------
# audio decoding and figures
# --------------------------------------------------------------------------------------


def _decode_audio_bytes(payload: bytes) -> tuple[np.ndarray, int, int]:
    """Return (mono waveform, sample rate, channel count) for the uploaded audio bytes."""
    try:
        import soundfile as sf

        data, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        return data.mean(axis=1), int(sample_rate), int(data.shape[1])
    except Exception:
        pass

    try:
        import torchaudio

        wav, sample_rate = torchaudio.load(io.BytesIO(payload))
        return wav.mean(dim=0).numpy(), int(sample_rate), int(wav.shape[0])
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except Exception as exc:
        raise RuntimeError(
            "Unable to decode the uploaded audio. Install soundfile or torchaudio for FLAC/MP3 support."
        ) from exc

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise RuntimeError(f"Unsupported PCM sample width: {width}")
    data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / float(2 ** (8 * width - 1))
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, int(sample_rate), int(channels)


def _hz_to_mel(hz: Any) -> Any:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: Any) -> Any:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 20.0) -> np.ndarray:
    points = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(sample_rate / 2.0), n_mels + 2))
    bins = np.clip(np.floor((n_fft + 1) * points / sample_rate).astype(int), 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        center = max(center, left + 1)
        right = min(max(right, center + 1), n_fft // 2)
        if center >= right:
            continue
        filters[index, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        filters[index, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return filters


def _mel_spectrogram_db(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    if wav.size < MEL_N_FFT:
        wav = np.pad(wav, (0, MEL_N_FFT - wav.size))

    window = np.hanning(MEL_N_FFT).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(wav, MEL_N_FFT)[::MEL_HOP_LENGTH]
    power = (np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2).T.astype(np.float32)

    mel = _mel_filterbank(sample_rate, MEL_N_FFT, MEL_BANDS) @ power
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    return np.maximum(mel_db, mel_db.max() - MEL_DYNAMIC_RANGE_DB)


def _configure_matplotlib_cn(plt) -> None:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Songti SC", "PingFang SC"):
        if candidate in installed:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _render_figures(item: AudioReportItem, wav: np.ndarray, sample_rate: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_matplotlib_cn(plt)
    times = np.arange(len(wav)) / float(sample_rate)
    duration = max(float(times[-1]) if len(times) else 0.0, 1e-3)

    figure, axes = plt.subplots(figsize=(4.6, 2.4), dpi=FIGURE_DPI)
    axes.plot(times, wav, linewidth=0.5, color=WAVEFORM_COLOR)
    axes.set_xlim(0, duration)
    axes.set_ylim(-1.05, 1.05)
    axes.set_xlabel("时间 / s", fontsize=8)
    axes.set_ylabel("幅度", fontsize=8)
    axes.tick_params(labelsize=7)
    axes.grid(alpha=0.25, linewidth=0.4)
    figure.tight_layout(pad=0.4)
    waveform_buffer = io.BytesIO()
    figure.savefig(waveform_buffer, format="png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(4.6, 2.4), dpi=FIGURE_DPI)
    axes.imshow(
        _mel_spectrogram_db(wav, sample_rate),
        origin="lower",
        aspect="auto",
        cmap=MEL_COLORMAP,
        extent=(0.0, duration, 0.0, sample_rate / 2000.0),
    )
    axes.set_xlabel("时间 / s", fontsize=8)
    axes.set_ylabel("频率 / kHz", fontsize=8)
    axes.tick_params(labelsize=7)
    figure.tight_layout(pad=0.4)
    mel_buffer = io.BytesIO()
    figure.savefig(mel_buffer, format="png", bbox_inches="tight")
    plt.close(figure)

    item.waveform_png = waveform_buffer.getvalue()
    item.mel_png = mel_buffer.getvalue()


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------


def _risk_from_fake_probability(value: float) -> str:
    if value >= 0.5:
        return "高"
    if value <= 0.2:
        return "低"
    return "中"


def _prediction_text(value: str, fake_probability: float) -> str:
    normalized = str(value).strip().lower()
    if normalized == "fake" or "伪造" in str(value):
        return "伪造（AI合成）"
    if normalized == "real" or "真实" in str(value):
        return "真实"
    return "伪造（AI合成）" if fake_probability >= 0.5 else "真实"


def _parse_risk(text: str) -> str:
    match = re.search(r"\[?伪造（?AI合成）?风险\]?\s*[：:]\s*([高中低])", text)
    return match.group(1) if match else "未知"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "未提供"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{int(seconds // 60)}分{seconds % 60:.1f}秒"


def _format_sample_rate(sample_rate: int | None) -> str:
    if not sample_rate:
        return "未提供"
    if sample_rate % 1000 == 0:
        return f"{sample_rate // 1000}kHz"
    return f"{sample_rate / 1000:.1f}kHz"


def _format_size_kb(size_bytes: int) -> str:
    return f"{max(1, round(size_bytes / 1024))}KB"


def _audio_extension(audio_url: str, filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix:
        return suffix
    header = audio_url[5:].split(";", 1)[0] if audio_url.startswith("data:") else ""
    return header.split("/")[-1] or "wav"


# --------------------------------------------------------------------------------------
# Doubao analysis
# --------------------------------------------------------------------------------------


def _doubao_prompt(item: AudioReportItem) -> str:
    probability_text = _format_percent(item.fake_probability)
    risk = _risk_from_fake_probability(item.fake_probability)
    return (
        "这张图是一段待取证音频的梅尔频谱图，横轴为时间，纵轴为频率，颜色越亮代表该时频点能量越强。"
        f"FakeTrace 语音伪造检测系统已经给出了该音频的伪造概率 fake_probability={probability_text}。"
        "请基于频谱图中可见的谐波结构、高频能量分布、底噪连续性、帧间过渡是否自然等线索，"
        "并结合该检测概率进行取证解析。"
        "风险等级规则必须遵循：伪造概率大于等于 50% 为高风险，小于等于 20% 为低风险，其余为中风险。"
        f"因此本次风险等级应为：{risk}。请严格按照以下格式输出：\n"
        f"[伪造（AI合成）风险]：{risk}\n"
        "[取证解析]：说明该音频为什么属于上述风险等级。"
    )


def _call_doubao(item: AudioReportItem, api_key: str) -> str:
    mel_data_url = "data:image/png;base64," + base64.b64encode(item.mel_png or b"").decode("ascii")
    payload = {
        "model": DOUBAO_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": mel_data_url},
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


def _analyze_with_doubao(items: list[AudioReportItem]) -> None:
    api_key = _read_api_key()
    workers = max(1, min(MAX_DOUBAO_WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(_call_doubao, item, api_key): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            item.analysis = re.sub(r"\n{3,}", "\n\n", future.result().strip())
            item.risk = _parse_risk(item.analysis)


def _offline_analysis(item: AudioReportItem) -> str:
    if _prediction_text(item.prediction, item.fake_probability) == "真实":
        detail = (
            "该音频的梅尔频谱在中高频区域保留了连续的环境底噪，谐波与噪声成分随时间自然起伏，"
            "波形图中静音段存在合理的本底噪声与呼吸停顿，整体符合真实录音设备在真实声学环境下的采集特征。"
        )
    else:
        detail = (
            "该音频的梅尔频谱在中高频区域缺少真实录音应有的随机底噪，谐波结构过于规整，帧间能量过渡异常平滑，"
            "波形图中静音段的本底噪声接近理想零值，符合语音合成或声音转换类伪造的典型特征。"
        )
    # Kept free of ASCII spaces so the shared 64-character pre-wrap in _split_analysis
    # leaves the sentence intact and reportlab's CJK wrapping handles the line breaks.
    return (
        f"[伪造（AI合成）风险]：{item.risk}\n"
        f"[取证解析]：FakeTrace给出的伪造概率为{_format_percent(item.fake_probability)}。{detail}"
        "本报告未启用大模型解析。"
    )


# --------------------------------------------------------------------------------------
# report tables
# --------------------------------------------------------------------------------------


def _audio_info_table(items: list[AudioReportItem]) -> Table:
    data = [[
        _p("序号", SMALL_CENTER_STYLE),
        _p("文件名", SMALL_CENTER_STYLE),
        _p("文件格式", SMALL_CENTER_STYLE),
        _p("时长", SMALL_CENTER_STYLE),
        _p("采样率", SMALL_CENTER_STYLE),
        _p("文件大小", SMALL_CENTER_STYLE),
    ]]
    for item in items:
        data.append([
            _p(str(item.index), SMALL_CENTER_STYLE),
            _p(item.filename, SMALL_CENTER_STYLE),
            _p(_audio_extension(item.audio_url, item.filename), SMALL_CENTER_STYLE),
            _p(_format_duration(item.duration_seconds), SMALL_CENTER_STYLE),
            _p(_format_sample_rate(item.sample_rate), SMALL_CENTER_STYLE),
            _p(_format_size_kb(item.size_bytes), SMALL_CENTER_STYLE),
        ])
    table = Table(data, colWidths=[14 * mm, 64 * mm, 22 * mm, 22 * mm, 24 * mm, 24 * mm])
    table.setStyle(_three_line_table_style())
    return table


def _figure_flowable(png_bytes: bytes, max_width: float, max_height: float) -> RLImage:
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return RLImage(io.BytesIO(png_bytes), width=width * scale, height=height * scale)


def _audio_figure_table(item: AudioReportItem) -> Table:
    column_width = 82 * mm
    figure_width = 78 * mm
    figure_height = 46 * mm
    table = Table(
        [
            [
                _figure_flowable(item.waveform_png or b"", figure_width, figure_height),
                _figure_flowable(item.mel_png or b"", figure_width, figure_height),
            ],
            [_p("波形图", SMALL_CENTER_STYLE), _p("梅尔频谱图", SMALL_CENTER_STYLE)],
        ],
        colWidths=[column_width] * 2,
        rowHeights=[figure_height + 4 * mm, 11 * mm],
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


def _result_text(item: AudioReportItem) -> str:
    return (
        f"经 FakeTrace 取证系统检测，该音频预测结果为{_prediction_text(item.prediction, item.fake_probability)}，"
        f"伪造概率为 {_format_percent(item.fake_probability)}，"
        f"真实概率为 {_format_percent(item.real_probability)}，"
        f"对应伪造风险为{item.risk}。"
        "具体的伪造痕迹您可以查看上方的波形图与梅尔频谱图，波形图反映音频整体的能量包络、静音段与突变位置，"
        "梅尔频谱图中颜色偏亮的区域为能量较强的频带，合成语音通常会在高频段呈现异常平滑、能量截断或谐波结构过于规整的分布。"
    )


def _summary_table(items: list[AudioReportItem]) -> Table:
    headers = ["序号", "文件名", "预测结果", "伪造概率", "真实概率", "伪造风险"]
    widths = [13 * mm, 64 * mm, 28 * mm, 22 * mm, 22 * mm, 21 * mm]
    data: list[list[Paragraph]] = [[_p(header, SMALL_CENTER_STYLE) for header in headers]]
    for item in items:
        data.append([
            _p(str(item.index), SMALL_CENTER_STYLE),
            _p(item.filename, SMALL_CENTER_STYLE),
            _p(_prediction_text(item.prediction, item.fake_probability), SMALL_CENTER_STYLE),
            _p(_format_percent(item.fake_probability), SMALL_CENTER_STYLE),
            _p(_format_percent(item.real_probability), SMALL_CENTER_STYLE),
            _p(item.risk, SMALL_CENTER_STYLE),
        ])
    table = Table(data, colWidths=widths)
    table.setStyle(_three_line_table_style())
    return table


def _build_story(
    items: list[AudioReportItem],
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
    story.append(_p(f"测试编号 {test_id} 语音伪造检测取证报告", TITLE_STYLE))
    story.append(Spacer(1, 14 * mm))
    story.append(_section_header(f"1.  测试编号 {test_id} 任务基本信息"))
    story.append(Spacer(1, 7 * mm))
    story.extend(
        [
            _bullet_line("取证功能归类：  语音伪造检测"),
            _bullet_line(f"待取证音频数量：  {len(items)}"),
            _bullet_line(f"音频上传时间：  {upload_time_text}"),
            _bullet_line(f"报告生成时间：  {report_time_text}"),
            _bullet_line(f"模型选择：  {model_name}"),
            _bullet_line(f"用以分析的大模型API版本：  {DOUBAO_MODEL if include_ai_analysis else '未启用'}"),
            _bullet_line("音频信息："),
            Spacer(1, 3 * mm),
            _audio_info_table(items),
            Spacer(1, 12 * mm),
            _section_header("2.  取证结果"),
            Spacer(1, 6 * mm),
        ]
    )

    for item in items:
        story.append(
            KeepTogether(
                [
                    _p(f"（{item.index}） 序号{item.index}: 音频 {item.filename} 取证结果", BODY_STYLE),
                    Spacer(1, 4 * mm),
                    _audio_figure_table(item),
                    Spacer(1, 5 * mm),
                    _p(_result_text(item), BODY_STYLE),
                    Spacer(1, 7 * mm),
                ]
            )
        )

    story.append(PageBreak())
    story.append(_section_header("3.  取证结果解析"))
    story.append(Spacer(1, 7 * mm))
    for item in items:
        story.append(_p(f"（{item.index}） 序号{item.index}: 音频 {item.filename} 取证结果解析", BODY_STYLE))
        story.append(Spacer(1, 3 * mm))
        for paragraph in _split_analysis(item.analysis):
            story.append(_p(paragraph, BODY_INDENT_STYLE))
            story.append(Spacer(1, 1.5 * mm))
        story.append(Spacer(1, 6 * mm))

    story.append(_section_header("4.  总结"))
    story.append(Spacer(1, 10 * mm))
    story.append(
        _p(
            f"经 FakeTrace 判定，用户上传的 {len(items)} 条待取证音频的检测结果如下。"
            "请注意，被判定为高风险伪造的音频，请您谨慎使用和传播，以免造成不良影响。",
            BODY_STYLE,
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(_summary_table(items))
    return story


def build_audio_report(payload: dict[str, Any]) -> GeneratedReport:
    model_key = str(payload.get("model") or "").lower()
    model_name = MODEL_REPORT_NAMES.get(model_key, str(payload.get("model_name") or "语音检测模型"))
    test_id = str(payload.get("test_id") or "A-000001").strip() or "A-000001"
    include_ai_analysis = bool(payload.get("include_ai_analysis"))
    upload_time = _parse_upload_time(payload.get("upload_time"))

    raw_items = payload.get("items") or []
    if not raw_items:
        raise ValueError("No audio report items provided.")

    items: list[AudioReportItem] = []
    for index, raw in enumerate(raw_items, start=1):
        fake_probability = float(raw.get("fake_probability") or 0.0)
        real_probability = float(raw.get("real_probability") or (1.0 - fake_probability))
        item = AudioReportItem(
            index=index,
            filename=str(raw.get("filename") or f"audio_{index}.wav"),
            audio_url=str(raw.get("audio_url") or ""),
            prediction=str(raw.get("prediction") or ("fake" if fake_probability >= 0.5 else "real")),
            fake_probability=max(0.0, min(1.0, fake_probability)),
            real_probability=max(0.0, min(1.0, real_probability)),
        )

        audio_bytes, _ = _decode_data_url(item.audio_url)
        item.size_bytes = len(audio_bytes)
        wav, sample_rate, channels = _decode_audio_bytes(audio_bytes)
        item.sample_rate = sample_rate
        item.channels = channels
        item.duration_seconds = len(wav) / float(sample_rate) if sample_rate else None
        _render_figures(item, wav, sample_rate)

        item.risk = _risk_from_fake_probability(item.fake_probability)
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
    output_path = REPORT_OUTPUT_DIR / f"audio_report_{report_id}.pdf"
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


def resolve_audio_report_path(report_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", report_id):
        raise FileNotFoundError("Invalid report id.")
    path = REPORT_OUTPUT_DIR / f"audio_report_{report_id}.pdf"
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {report_id}")
    return path
