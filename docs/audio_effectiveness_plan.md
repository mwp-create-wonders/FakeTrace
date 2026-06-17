# Audio Effectiveness Plan

This note defines how to prove that the FakeTrace audio module is effective in
engineering and in the competition report.

## Main Claim

The audio branch should claim a complete and reproducible audio deepfake
detection pipeline:

```text
audio file -> decoding -> resampling -> fixed-length waveform -> feature extractor
-> pooling/classifier -> probability -> threshold calibration -> predict.csv/submission.zip
```

Do not frame the contribution as only "we added a model." The stronger claim is
that the system supports training, evaluation, threshold calibration, batch
prediction, and reportable evidence.

## Recommended Technical Story

Use AST AudioSet as the main implemented route and AASIST as the baseline /
reference comparison.

### AST AudioSet Route

Role in the paper:

- Primary route for your implementation narrative.
- Represents the feature-extractor direction from your team assignment.
- Fits the current ATADD memory: best progress result came from the
  `focus_music3_sound2` AST route and fake-sensitive threshold calibration.

What to emphasize:

- Spectrogram-style pretrained audio representation.
- Transfer learning from large-scale audio pretraining.
- Flexible pooling/classifier head, including mean pooling and TF-ASP.
- Better engineering compatibility with the FakeTrace unified pipeline.
- Threshold calibration substantially improves progress results when the model
  underestimates fake probability.

### AASIST Route

Role in the paper:

- Strong audio anti-spoofing baseline or teammate comparison route.
- Use it to show that FakeTrace is not evaluated only against a weak baseline.
- Keep its ownership separate if it belongs to another teammate's workstream.

What to emphasize:

- AASIST is a specialized anti-spoofing architecture.
- It is valuable as a baseline, but it is not the main "feature extractor +
  training enhancement" contribution for this branch.

### Combined Framing

Use this framing:

```text
We evaluated a specialized anti-spoofing baseline represented by AASIST, and
implemented a feature-extractor-based route represented by AST AudioSet. The AST
route was integrated into FakeTrace with a full training/evaluation/prediction
toolchain. Further threshold calibration corrected the model's systematic
underestimation of fake probability and improved Track2 progress performance.
```

Avoid saying "AST+AASIST" as if it is one hybrid model unless a real fusion is
implemented. In the current codebase, they should be treated as two routes:

- AASIST: baseline/comparison.
- AST: implemented mainline.

## Evidence Needed For The Report

### Engineering Evidence

Minimum evidence that the audio branch "runs":

1. `audio-healthcheck` output:
   - manifest rows loaded
   - label/type distribution
   - audio decode success
   - waveform tensor shape after preprocessing
2. `audio-train` output:
   - `training_log.csv`
   - `run_summary.json`
   - `best.pt`
3. `audio-eval` output:
   - `eval_summary.json`
4. `audio-threshold-scan` output:
   - `threshold_scan.csv`
   - `threshold_scan_summary.json`
5. `audio-predict` output:
   - `predict.csv`
   - `submission.zip`
   - `predict_summary.json`

### Metric Evidence

For the report table, collect:

- model route
- config
- threshold
- accuracy
- macro F1
- Track2 macro F1
- speech F1
- sound F1
- singing F1
- music F1

The most important row is the calibrated AST route. The ATADD memory currently
records the best progress candidate as a low-threshold AST route around
`3e-6`, with `track2_macro_f1` around `82.85` on the progress feedback.

### Ablation Evidence

Recommended compact ablation table:

| Route | Threshold | Main finding |
| --- | --- | --- |
| AST AudioSet default | 0.5 | Reasonable baseline, but fake probability is underestimated |
| AST AudioSet calibrated | best from scan | Strongest progress result |
| AASIST baseline | default/best available | Specialized anti-spoofing comparison |
| AST + TF-ASP | best from scan | Optional pooling-head ablation |
| AST + augmentation | best from scan | Optional training-enhancement ablation |

Only include rows that were actually run or are clearly sourced from teammate
records.

## Immediate Development Tasks

1. Run `audio-healthcheck` on a small labeled manifest.
2. Run or copy the cloud AST checkpoint and validate `audio-eval`.
3. Run `audio-threshold-scan` with the default `1e-7..1e-2` logspace range.
4. Generate a submission using `--threshold-summary`.
5. Backfill results into a table for the report.

## Report Placement

### First Chapter

Use AST and AASIST to motivate the two families of audio detection:

- specialized anti-spoofing networks
- pretrained feature extractor transfer learning

### Second Chapter

Describe the AST route implementation:

- preprocessing
- feature extractor
- classifier/pooling head
- training loop
- threshold calibration
- FakeTrace integration

### Third Chapter

Show:

- data/task setup
- baseline comparison
- threshold calibration result
- final selected result and submission artifacts

## Risk Notes

- If only local healthcheck runs but no real checkpoint is available, the report
  can claim engineering readiness, not final detection effectiveness.
- If AASIST numbers are from a teammate or official baseline, cite them as
  baseline/comparison results and do not claim them as this branch's
  implementation unless the code is integrated.
- If progress-set threshold is used, describe it as calibration on feedback data
  and avoid implying it is a universally optimal threshold.
