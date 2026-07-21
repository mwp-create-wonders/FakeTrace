from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .audio_cli import AUDIO_COMMANDS, main as audio_main
from .config import load_config
from .inference_engine import MARCInferenceEngine, collect_image_paths
from .paths import DEFAULT_CONFIG_PATH


def _build_image_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FakeTrace image detector CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path. Can be passed multiple times.",
    )
    input_group.add_argument("--image-dir", help="Directory containing images.")

    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config JSON path.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--recursive", action="store_true", help="Scan image-dir recursively.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--image-size", type=int, default=None, help="Override input image size.")
    parser.add_argument("--threshold", type=float, default=None, help="Override real/fake threshold.")
    parser.add_argument("--save-json", default="", help="Optional JSON output path.")
    parser.add_argument("--save-csv", default="", help="Optional CSV output path.")
    return parser


def _print_root_help() -> None:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="FakeTrace unified CLI for image and audio workflows",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.epilog = (
        "Examples:\n"
        "  python3 app.py --image sample.jpg\n"
        "  python3 app.py detect --image-dir ./images --recursive\n"
        "  python3 app.py audio audio-predict --config configs/audio/ast_audioset_ft.yaml \\\n"
        "    --checkpoint models/audio/best.pt --audio-dir ./audio --output-dir ./output/audio\n"
        "  python3 audio_app.py audio-predict ...  # legacy wrapper"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        help="Use 'detect' for image CLI or 'audio' for audio experiment commands.",
    )
    parser.print_help()
    print("\nAudio commands:", ", ".join(sorted(AUDIO_COMMANDS)))


def _build_overrides(args: argparse.Namespace) -> dict:
    overrides = {}
    for key in ("checkpoint", "device", "batch_size", "image_size", "threshold"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    return overrides


def _print_results(results, threshold: float) -> None:
    print("\nFakeTrace CLI inference")
    print(f"Threshold: real_probability >= {threshold:.3f} => real")
    print("-" * 92)
    print(f"{'prediction':10s} {'real_prob':>10s} {'fake_prob':>10s}  file")
    print("-" * 92)
    for item in results:
        print(
            f"{item.prediction:10s} "
            f"{item.real_probability:10.4f} "
            f"{item.fake_probability:10.4f}  "
            f"{item.filename}"
        )
    print("-" * 92)
    print(f"Total: {len(results)} image(s)\n")


def _save_outputs(results, save_json: str, save_csv: str) -> None:
    rows = [item.to_dict() for item in results]
    if save_json:
        json_path = Path(save_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
        print(f"Saved JSON: {json_path}")

    if save_csv:
        csv_path = Path(save_csv).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "path",
                    "filename",
                    "real_probability",
                    "fake_probability",
                    "prediction",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {csv_path}")


def _run_image_cli(argv: list[str]) -> None:
    args = _build_image_parser().parse_args(argv)
    source_paths = args.image if args.image else [args.image_dir]
    paths = collect_image_paths(source_paths, recursive=args.recursive)
    config = load_config(args.config, _build_overrides(args))
    engine = MARCInferenceEngine(config)
    results = engine.predict_paths(paths)
    print(f"Device: {engine.device}")
    print(f"Checkpoint: {config.checkpoint}")
    _print_results(results, config.threshold)
    _save_outputs(results, args.save_json, args.save_csv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_root_help()
        return

    head = argv[0]
    if head in {"audio", "aud"}:
        audio_main(argv[1:])
        return
    if head in AUDIO_COMMANDS:
        audio_main(argv)
        return
    if head in {"detect", "image"}:
        _run_image_cli(argv[1:])
        return
    _run_image_cli(argv)


if __name__ == "__main__":
    main()
