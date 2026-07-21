from __future__ import annotations

import argparse


AUDIO_COMMANDS = {
    "audio-train",
    "audio-eval",
    "audio-predict",
    "audio-threshold-scan",
    "audio-healthcheck",
}

def _run_train(args: argparse.Namespace):
    from .features.audio.experiment import run_train

    return run_train(args)


def _run_eval(args: argparse.Namespace):
    from .features.audio.experiment import run_eval

    return run_eval(args)


def _run_predict(args: argparse.Namespace):
    from .features.audio.predict_export import run_predict

    return run_predict(args)


def _run_threshold_scan(args: argparse.Namespace):
    from .features.audio.threshold_scan import run_threshold_scan

    return run_threshold_scan(args)


def _run_healthcheck(args: argparse.Namespace):
    from .features.audio.healthcheck import run_healthcheck

    return run_healthcheck(args)


def add_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("audio-train", help="Train an audio real/fake model.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--train-manifest", required=True, type=str)
    parser.add_argument("--val-manifest", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--enable-augment", action="store_true")
    parser.add_argument("--init-checkpoint", default=None, type=str)
    parser.add_argument("--init-nonstrict", action="store_true")
    parser.set_defaults(func=_run_train)


def add_eval_parser(subparsers) -> None:
    parser = subparsers.add_parser("audio-eval", help="Evaluate an audio model checkpoint.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.set_defaults(func=_run_eval)


def add_predict_parser(subparsers) -> None:
    parser = subparsers.add_parser("audio-predict", help="Batch audio prediction and export predict.csv.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", default=None, type=str)
    input_group.add_argument("--audio-dir", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--pattern", default="*.flac", type=str)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fake-threshold", default=0.5, type=float)
    parser.add_argument("--threshold-summary", default=None, type=str, help="Use best.threshold from threshold_scan_summary.json.")
    parser.add_argument("--save-probs", action="store_true")
    parser.set_defaults(func=_run_predict)


def add_threshold_scan_parser(subparsers) -> None:
    parser = subparsers.add_parser("audio-threshold-scan", help="Scan fake-probability thresholds on a labeled manifest.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--metric", default="macro_f1", type=str)
    parser.add_argument("--thresholds", default=None, type=str, help="Comma-separated thresholds, e.g. 0.5,0.1,0.01.")
    parser.add_argument("--min-threshold", default=1e-7, type=float)
    parser.add_argument("--max-threshold", default=1e-2, type=float)
    parser.add_argument("--steps", default=101, type=int)
    parser.add_argument("--logspace", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(func=_run_threshold_scan)


def add_healthcheck_parser(subparsers) -> None:
    parser = subparsers.add_parser("audio-healthcheck", help="Check audio config, manifest, and preprocessing without loading a model.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--output-dir", default="", type=str)
    parser.add_argument("--max-samples", default=3, type=int)
    parser.add_argument("--max-missing", default=10, type=int)
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.set_defaults(func=_run_healthcheck)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FakeTrace audio experiment CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_train_parser(subparsers)
    add_eval_parser(subparsers)
    add_predict_parser(subparsers)
    add_threshold_scan_parser(subparsers)
    add_healthcheck_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
