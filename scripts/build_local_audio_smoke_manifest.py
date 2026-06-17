from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small local audio manifest from an ATADD manifest by matching source_name to local audio files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=32, type=int)
    parser.add_argument("--type", dest="audio_type", default="", help="Optional type filter, e.g. speech/music.")
    parser.add_argument("--label", default="", help="Optional label filter, e.g. 0/1.")
    parser.add_argument(
        "--per-label",
        default=0,
        type=int,
        help="Build a balanced manifest with up to N local files for each label.",
    )
    parser.add_argument(
        "--per-type",
        default=0,
        type=int,
        help=(
            "Build a balanced manifest with up to N local files for each type. "
            "When combined with --per-label, values must match and selection is per label/type bucket."
        ),
    )
    return parser.parse_args()


def resolve_local_row(item: dict[str, str], audio_dir: Path) -> dict[str, str] | None:
    local_path = audio_dir / Path(item["source_name"]).name
    if not local_path.is_file():
        return None
    return {
        "audio_path": str(local_path.resolve()),
        "label": item["label"],
        "type": item.get("type", ""),
        "source_name": item["source_name"],
    }


def balance_spec(args: argparse.Namespace) -> tuple[tuple[str, ...], int] | None:
    if args.per_label < 0 or args.per_type < 0:
        raise ValueError("--per-label and --per-type must be non-negative.")
    if args.per_label and args.per_type:
        if args.per_label != args.per_type:
            raise ValueError("--per-label and --per-type must match when used together.")
        return ("label", "type"), args.per_label
    if args.per_label:
        return ("label",), args.per_label
    if args.per_type:
        return ("type",), args.per_type
    return None


def select_balanced_rows(
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
    per_bucket: int,
) -> list[dict[str, str]]:
    counts: Counter[tuple[str, ...]] = Counter()
    selected = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        if counts[key] >= per_bucket:
            continue
        selected.append(row)
        counts[key] += 1
    return selected


def main() -> None:
    args = parse_args()
    spec = balance_spec(args)
    rows = []
    with args.source_manifest.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "source_name" not in (reader.fieldnames or []):
            raise ValueError("Source manifest must contain source_name.")
        if "label" not in (reader.fieldnames or []):
            raise ValueError("Source manifest must contain label.")

        for item in reader:
            if args.audio_type and item.get("type") != args.audio_type:
                continue
            if args.label and str(item.get("label")) != args.label:
                continue
            row = resolve_local_row(item, args.audio_dir)
            if row is None:
                continue
            rows.append(row)
            if spec is None and len(rows) >= args.limit:
                break

    if spec is not None:
        rows = select_balanced_rows(rows, fields=spec[0], per_bucket=spec[1])

    if not rows:
        raise RuntimeError("No matching local audio rows were found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_path", "label", "type", "source_name"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output": str(args.output.resolve()),
        "rows": len(rows),
    }
    if spec is not None:
        summary["balanced_by"] = "+".join(spec[0])
        summary["per_bucket"] = spec[1]
    print(summary)


if __name__ == "__main__":
    main()
