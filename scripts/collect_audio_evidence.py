from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_predict_csv(predict_summary: dict, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    raw_path = predict_summary.get("predict_csv")
    if raw_path:
        return Path(raw_path)
    return None


def load_prediction_counts(path: Path | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if path is None or not path.is_file():
        return counts
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "predict" not in (reader.fieldnames or []):
            return counts
        for row in reader:
            label = str(row.get("predict", "")).strip()
            if label:
                counts[label] += 1
    return counts


def format_prediction_counts(counts: Counter[str]) -> str:
    if not counts:
        return ""
    return "predictions " + ", ".join(f"{label}={counts[label]}" for label in sorted(counts))


def format_predict_summary(predict: dict, counts: Counter[str]) -> str:
    if not predict and not counts:
        return ""
    parts = []
    if predict:
        parts.append(f"{predict.get('num_files')} files")
        parts.append(f"threshold {predict.get('fake_threshold')}")
    count_summary = format_prediction_counts(counts)
    if count_summary:
        parts.append(count_summary)
    if predict and predict.get("submission_zip"):
        parts.append(f"submission {predict.get('submission_zip')}")
    return ", ".join(parts)


def format_threshold_summary(threshold: dict) -> str:
    if not threshold:
        return ""
    best = threshold.get("best") if isinstance(threshold.get("best"), dict) else {}
    metric = threshold.get("metric") or best.get("metric")
    best_threshold = best.get("threshold")
    parts = []
    if metric:
        parts.append(f"metric {metric}")
    if best_threshold is not None:
        parts.append(f"best threshold {best_threshold}")
    if metric and best.get(metric) is not None:
        parts.append(f"best {metric} {best.get(metric)}")
    return ", ".join(parts)


def escape_markdown_cell(value: str) -> str:
    return str(value).replace("|", "\\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FakeTrace audio evidence into report-ready CSV and Markdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--healthcheck", type=Path, default=Path("output/audio_healthcheck/audio_healthcheck.json"))
    parser.add_argument("--predict-summary", type=Path, default=Path("output/audio_predict_smoke/predict_summary.json"))
    parser.add_argument("--predict-csv", type=Path, default=None, help="Optional predict.csv path; defaults to predict_summary.predict_csv.")
    parser.add_argument("--threshold-summary", type=Path, default=Path("output/audio_thresholds/threshold_scan_summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/audio_evidence"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    health = load_json(args.healthcheck)
    predict = load_json(args.predict_summary)
    threshold = load_json(args.threshold_summary)
    predict_csv = resolve_predict_csv(predict, args.predict_csv)
    prediction_counts = load_prediction_counts(predict_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "evidence": "preprocessing_healthcheck",
            "status": "available" if health else "missing",
            "key_result": (
                f"{health.get('num_rows')} rows, tensor length {health.get('expected_num_samples')}, "
                f"missing {health.get('missing_count')}"
                if health
                else ""
            ),
            "artifact": str(args.healthcheck),
        },
        {
            "evidence": "model_prediction_smoke",
            "status": "available" if predict or prediction_counts else "missing",
            "key_result": format_predict_summary(predict, prediction_counts),
            "artifact": str(args.predict_summary),
        },
        {
            "evidence": "threshold_calibration",
            "status": "available" if threshold else "missing",
            "key_result": format_threshold_summary(threshold),
            "artifact": str(args.threshold_summary),
        },
    ]

    csv_path = args.output_dir / "audio_evidence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["evidence", "status", "key_result", "artifact"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_dir / "audio_evidence.md"
    lines = [
        "# Audio Evidence Summary",
        "",
        "| Evidence | Status | Key result | Artifact |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {escape_markdown_cell(row['evidence'])} | {escape_markdown_cell(row['status'])} | "
            f"{escape_markdown_cell(row['key_result'])} | `{escape_markdown_cell(row['artifact'])}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print({"csv": str(csv_path.resolve()), "markdown": str(md_path.resolve())})


if __name__ == "__main__":
    main()
