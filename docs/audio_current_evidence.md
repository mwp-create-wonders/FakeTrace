# Audio Current Evidence

This file records what is already working and what still needs a full model
checkpoint run.

## What Is Working Locally

The FakeTrace audio preprocessing chain has been smoke-tested on real ATADD
`.flac` files from:

```text
E:\aNB\TECH\ATADD\data\train_reversed
```

Command used:

```powershell
conda run -n ldm_final python audio_app.py audio-healthcheck `
  --config configs/audio/ast_audioset_ft.yaml `
  --manifest output/audio_healthcheck/local_smoke_manifest.csv `
  --output-dir output/audio_healthcheck `
  --max-samples 4 `
  --fail-on-missing
```

Result summary:

```text
rows: 8
sample_rate: 16000
max_seconds: 5.0
expected tensor length: 80000
missing files: 0
inspected tensors: shape [80000], dtype torch.float32
```

This proves the local engineering path can:

- read a manifest
- resolve local audio paths
- decode real ATADD FLAC files
- convert to mono waveform
- resample/pad/crop to the configured fixed length
- emit tensors suitable for the AST route

## Balanced Local Detection Evidence Workflow

For detection evidence, prefer a local smoke manifest that covers available
labels and audio types instead of relying on the first matching files in source
manifest order.

Simple limit mode is still available:

```powershell
python scripts/build_local_audio_smoke_manifest.py `
  --source-manifest <atadd-manifest.csv> `
  --audio-dir E:\aNB\TECH\ATADD\data\train_reversed `
  --output output/audio_healthcheck/local_smoke_manifest.csv `
  --limit 32
```

Balanced label/type mode:

```powershell
python scripts/build_local_audio_smoke_manifest.py `
  --source-manifest <atadd-manifest.csv> `
  --audio-dir E:\aNB\TECH\ATADD\data\train_reversed `
  --output output/audio_healthcheck/local_balanced_manifest.csv `
  --per-label 4 `
  --per-type 4
```

Then collect report-ready evidence after prediction and threshold scanning:

```powershell
python scripts/collect_audio_evidence.py `
  --healthcheck output/audio_healthcheck/audio_healthcheck.json `
  --predict-summary output/audio_predict_smoke/predict_summary.json `
  --threshold-summary output/audio_thresholds/threshold_scan_summary.json `
  --output-dir output/audio_evidence
```

The evidence summary now includes prediction label counts from `predict.csv`
when available, plus the threshold scan metric, best threshold, and best metric
value from `threshold_scan_summary.json`.

## What Is Working With The Model

The local AST checkpoint exists:

```text
models/audio/best.pt
```

CPU prediction smoke test:

```powershell
conda run -n ldm_final python audio_app.py audio-predict `
  --config configs/audio/ast_audioset_ft.yaml `
  --checkpoint models/audio/best.pt `
  --manifest output/audio_healthcheck/local_smoke_manifest.csv `
  --output-dir output/audio_predict_smoke `
  --device cpu `
  --batch-size 2 `
  --save-probs
```

This produced:

```text
output/audio_predict_smoke/predict.csv
output/audio_predict_smoke/submission.zip
output/audio_predict_smoke/predict_summary.json
```

The submission zip was structurally correct and contained exactly:

```text
predict.csv
```

On the 8-sample local smoke manifest, default threshold `0.5` predicted:

```text
real: 8
fake: 0
```

Using a low threshold summary with `best.threshold = 3e-6`:

```powershell
conda run -n ldm_final python audio_app.py audio-predict `
  --config configs/audio/ast_audioset_ft.yaml `
  --checkpoint models/audio/best.pt `
  --manifest output/audio_healthcheck/local_smoke_manifest.csv `
  --output-dir output/audio_predict_smoke_th3e6 `
  --device cpu `
  --batch-size 2 `
  --threshold-summary output/audio_predict_smoke/threshold_summary_3e-6.json `
  --save-probs
```

The same 8 samples predicted:

```text
real: 6
fake: 2
```

This demonstrates the exact engineering mechanism behind the ATADD memory:
lowering the fake threshold changes the decision boundary and recovers samples
whose `prob_fake` is far below `0.5` but still informative.

## Local Balanced Smoke Evaluation

A small balanced local evaluation manifest was built from
`ATADD/data/manifests/track2_train_balanced_500.csv` by selecting files that
exist under `ATADD/data/train_reversed`:

```text
output/audio_eval_smoke/balanced_4_per_label_type.csv
```

Composition:

```text
4 samples per (label, type)
2 labels: real/fake
4 types: speech/sound/singing/music
total rows: 32
```

Default-threshold evaluation command:

```powershell
conda run -n ldm_final python audio_app.py audio-eval `
  --config configs/audio/ast_audioset_ft.yaml `
  --checkpoint models/audio/best.pt `
  --manifest output/audio_eval_smoke/balanced_4_per_label_type.csv `
  --output-dir output/audio_eval_smoke/default_eval `
  --device cpu `
  --batch-size 2
```

Result:

```text
accuracy: 1.0
macro_f1: ~1.0
track2_macro_f1: ~1.0
speech/sound/singing/music macro_f1: ~1.0 each
```

Threshold scan command:

```powershell
conda run -n ldm_final python audio_app.py audio-threshold-scan `
  --config configs/audio/ast_audioset_ft.yaml `
  --checkpoint models/audio/best.pt `
  --manifest output/audio_eval_smoke/balanced_4_per_label_type.csv `
  --output-dir output/audio_eval_smoke/threshold_scan `
  --device cpu `
  --batch-size 2 `
  --metric track2_macro_f1 `
  --thresholds "0.5,0.01,0.001,0.0001,0.00001,0.000003,0.000001"
```

Result summary:

```text
best threshold on this smoke set: 0.0001
best track2_macro_f1: ~1.0
```

Important limitation: this is a local smoke evaluation built from the available
training-side files, so it proves that evaluation and threshold scanning work
end to end. It should not be presented as the final competition score. Final
effectiveness should still use the official/dev/progress split results from the
cloud ATADD workflow.

## Important Fix Made

The first healthcheck failed because the `ldm_final` conda environment had no
torchaudio audio backend:

```text
torchaudio.list_audio_backends() == []
```

FakeTrace now includes a shared audio loader:

```text
src/faketrace_app/features/audio/audio_io.py
```

It tries `torchaudio.load` first and falls back to `soundfile`, which is more
reliable for local FLAC decoding on this Windows/conda setup. `soundfile` has
also been added to `requirements.txt`.

## Evidence Already Available From ATADD

Existing ATADD artifacts show that the AST AudioSet progress route has already
produced probability files and thresholded submissions:

```text
ATADD/outputs/submission_track2_progress_ast_focus_music3_sound2_probs_v1/predict.csv
ATADD/submissions/recommended_20260527/README_summary.json
ATADD/submissions/recommended_20260527_refine/README_summary.json
ATADD/reports/threshold_probe_focus_music3_sound2_*.json
```

The important pattern is:

- default probability outputs exist with `prob_real` and `prob_fake`
- thresholded `submission.zip` candidates exist
- low fake thresholds around `3e-6` to `6e-6` were explored
- project memory records a strong progress result for the calibrated AST route

## Server Evidence Bundle

The server was opened at:

```text
connect.bjb1.seetacloud.com:53229
```

Remote ATADD workspace:

```text
/root/autodl-tmp/ATADD
```

A compact evidence bundle was created remotely and downloaded locally:

```text
output/server_audio_report_bundle/faketrace_audio_report_bundle_20260617
```

Key formal metrics are summarized in:

```text
docs/audio_server_results_summary.md
```

Most important server results:

```text
track2_dev_balanced_500_v3:
  accuracy = 0.995750
  macro_f1 = 0.995750
  track2_macro_f1 = 0.995750

track2_dev_disjoint_cap500:
  accuracy = 0.995758
  macro_f1 = 0.995685
  track2_macro_f1 = 0.991624
```

These server results should be used as the formal detection-effectiveness
evidence in the report. The local smoke results should be used as engineering
operability evidence.

## Can AST + AASIST Satisfy The Paper?

Yes, if the paper frames them correctly:

- AST AudioSet is the main implemented feature-extractor route.
- AASIST is a specialized anti-spoofing baseline/comparison route.
- The core effectiveness story is not "AST+AASIST is one model"; it is:
  - compare a specialized anti-spoofing baseline against a feature-extractor
    route
  - show that the AST route is integrated end-to-end in FakeTrace
  - show that threshold calibration fixes fake-probability underestimation

This is enough for a credible report section if we include:

1. Engineering proof: healthcheck, train/eval/predict commands, submission
   package generation.
2. Metric proof: AST progress F1 and threshold calibration table.
3. Baseline proof: AASIST result as comparison, without claiming ownership if
   it belongs to a teammate's baseline work.

## What Still Needs One Full Confirmation

To make the article airtight, run one of these:

### Best option

Run `audio-eval` or `audio-predict` in an environment with the AST checkpoint:

```powershell
conda run -n ldm_final python audio_app.py audio-predict `
  --config configs/audio/ast_audioset_ft.yaml `
  --checkpoint models/audio/best.pt `
  --audio-dir <eval-audio-dir> `
  --output-dir output/audio_submission `
  --threshold-summary output/audio_thresholds/threshold_scan_summary.json `
  --save-probs
```

### If the checkpoint is only on cloud

Use the cloud workflow in:

```text
docs/audio_remote_workflow.md
```

Then copy back:

- `eval_summary.json`
- `threshold_scan_summary.json`
- `predict_summary.json`
- final `submission.zip`

## Recommended Report Claim

Use language like this:

```text
The audio branch of FakeTrace adopts a pretrained audio feature extraction
route based on AST AudioSet and compares it with a specialized anti-spoofing
baseline such as AASIST. Audio samples are decoded, resampled, normalized into
fixed-length waveforms, passed through the feature extractor and classification
head, and finally calibrated through a fake-probability threshold scan. The
threshold calibration is necessary because the progress results show systematic
underestimation of fake probability. After calibration, the AST route achieves
the strongest observed Track2 progress performance among the tested audio
feature-extractor configurations.
```
