# Audio Remote Workflow

This workflow keeps FakeTrace as the local integration and reporting repo while
running real ATADD training, evaluation, and submission work on the cloud GPU
machine.

## Environment

Current ATADD memory points to the remote working directory:

```bash
/root/autodl-tmp/ATADD
```

Use the active cloud endpoint recorded in the private project memory or local
secret notes. Do not commit credentials, tokens, or SSH details.

For Hugging Face downloads on the cloud machine:

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token <your-token>
```

## Standard Run Pattern

Long jobs should run inside `tmux`, `screen`, or `nohup` with explicit log files.
Do not rely on an interactive SSH session staying alive.

Recommended tmux setup:

```bash
cd /root/autodl-tmp/ATADD
tmux new -s atadd-audio
```

Run commands from the project root and keep outputs under `outputs/` or
`reports/` so they can be copied back to FakeTrace/reporting later.

## Train

Example AST AudioSet run:

```bash
cd /root/autodl-tmp/ATADD
export PYTHONPATH=src
python -m atadd.train \
  --config configs/experiments/ast_audioset_ft.yaml \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --output-dir outputs/ast_audioset_ft \
  --device cuda \
  --seed 42 \
  2>&1 | tee outputs/ast_audioset_ft/train.log
```

For background execution:

```bash
mkdir -p outputs/ast_audioset_ft
nohup bash -lc 'export PYTHONPATH=src; python -m atadd.train --config configs/experiments/ast_audioset_ft.yaml --train-manifest data/manifests/train.csv --val-manifest data/manifests/val.csv --output-dir outputs/ast_audioset_ft --device cuda --seed 42' \
  > outputs/ast_audioset_ft/train.log 2>&1 &
tail -f outputs/ast_audioset_ft/train.log
```

Expected artifacts:

- `outputs/ast_audioset_ft/best.pt`
- `outputs/ast_audioset_ft/training_log.csv`
- `outputs/ast_audioset_ft/run_summary.json`

## Evaluate

```bash
export PYTHONPATH=src
python -m atadd.eval \
  --config configs/experiments/ast_audioset_ft.yaml \
  --checkpoint outputs/ast_audioset_ft/best.pt \
  --manifest data/manifests/val.csv \
  --output-dir outputs/ast_audioset_ft_eval \
  --device cuda
```

Expected artifact:

- `outputs/ast_audioset_ft_eval/eval_summary.json`

## Threshold Calibration

The current ATADD memory says the strongest progress results came from
fake-sensitive threshold calibration, not from another full retrain. The model
appears to underestimate `fake`, and the useful threshold range is much lower
than the default `0.5`.

In FakeTrace, use the migrated scanner on a labeled validation/progress-style
manifest:

```bash
python audio_app.py audio-threshold-scan \
  --config configs/audio/ast_audioset_ft.yaml \
  --checkpoint output/audio_ast/best.pt \
  --manifest data/manifests/val.csv \
  --output-dir output/audio_thresholds \
  --metric track2_macro_f1
```

Default scan strategy:

- logspace scan
- `1e-7` to `1e-2`
- `101` thresholds

The scanner writes:

- `threshold_scan.csv`
- `threshold_scan_summary.json`

For a focused ATADD progress rerun, start near the known useful region:

```bash
python audio_app.py audio-threshold-scan \
  --config configs/audio/ast_audioset_ft.yaml \
  --checkpoint output/audio_ast/best.pt \
  --manifest data/manifests/val.csv \
  --output-dir output/audio_thresholds_refine \
  --metric track2_macro_f1 \
  --thresholds 0.000001,0.000003,0.000005,0.00001
```

## Predict And Package

Use the best threshold from a scan summary:

```bash
python audio_app.py audio-predict \
  --config configs/audio/ast_audioset_ft.yaml \
  --checkpoint output/audio_ast/best.pt \
  --audio-dir data/eval_audio \
  --output-dir output/audio_submission \
  --threshold-summary output/audio_thresholds/threshold_scan_summary.json \
  --save-probs
```

Expected artifacts:

- `predict.csv`
- `submission.zip`
- `predict_summary.json`

For ATADD/Codabench-style submission, check that `submission.zip` contains
exactly one file named `predict.csv`.

```bash
python - <<'PY'
import zipfile
with zipfile.ZipFile('output/audio_submission/submission.zip') as zf:
    print(zf.namelist())
PY
```

## Result Backfill

After every meaningful run, copy or summarize these into the reporting branch:

- config path
- checkpoint path
- threshold
- `macro_f1`
- `track2_macro_f1`
- `speech_macro_f1`
- `sound_macro_f1`
- `singing_macro_f1`
- `music_macro_f1`
- submission zip name

Keep team boundaries clear:

- This track is for feature-extractor and training-enhancement routes such as
  AST, XLSR, WavLM, MERT, CLAP, ASP, TF-ASP, and waveform augmentation.
- Avoid overwriting or duplicating teammates' AASIST and ALLM baseline results.

## Paper Hooks

Use the generated artifacts directly in the report:

- `training_log.csv`: training curve and convergence description.
- `eval_summary.json`: evaluation setup and headline metrics.
- `threshold_scan.csv`: threshold calibration table/plot.
- `threshold_scan_summary.json`: best threshold and final selected result.
- `predict_summary.json`: final submission metadata.
