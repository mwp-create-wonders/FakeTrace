# Audio Server Results Summary

Server evidence bundle:

```text
output/server_audio_report_bundle/faketrace_audio_report_bundle_20260617
```

Remote ATADD workspace:

```text
/root/autodl-tmp/ATADD
```

## Main AST Route

```text
config: configs/experiments/ast_audioset_ft_music_sound_focus.yaml
checkpoint: outputs/track2_subset_ast_focus_music3_sound2_v1/best.pt
```

## Dev Evaluation

| Evaluation split | Accuracy | Macro F1 | Track2 Macro F1 | Speech F1 | Sound F1 | Singing F1 | Music F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `track2_dev_balanced_500_v3` | 0.995750 | 0.995750 | 0.995750 | 0.992000 | 0.998000 | 0.999000 | 0.994000 |
| `track2_dev_disjoint_cap500` | 0.995758 | 0.995685 | 0.991624 | 0.992000 | 0.996000 | 1.000000 | 0.978495 |

These are the formal detection-effectiveness results to use in the report.

## Progress Prediction

```text
num_files: 45875
predict_csv: outputs/submission_track2_progress_ast_focus_music3_sound2_probs_v1/predict.csv
checkpoint: outputs/track2_subset_ast_focus_music3_sound2_v1/best.pt
```

## Threshold Probe

| Threshold | Fake count | Real count | Changed vs 5e-6 |
| ---: | ---: | ---: | ---: |
| `8e-6` | 34321 | 11554 | 644 |
| `7e-6` | 34500 | 11375 | 465 |
| `6e-6` | 34709 | 11166 | 256 |
| `4e-6` | 35295 | 10580 | 330 |
| `3e-6` | 35743 | 10132 | 778 |

This supports the fake-sensitive calibration narrative: the default `0.5`
threshold is too conservative for progress-style submission decisions, while
low-threshold probing materially changes the real/fake distribution.

## Paper Wording

```text
On the Track2 balanced dev subset, the AST AudioSet route achieved 99.575%
accuracy and 99.575% Track2 macro-F1. On a disjoint capped dev subset, it
maintained 99.576% accuracy and 99.162% Track2 macro-F1, showing strong
cross-subset robustness. For progress-set submission, we further performed
fake-probability threshold probing and found that low thresholds in the 3e-6 to
8e-6 range substantially changed the real/fake decision distribution, which
motivated the final fake-sensitive calibration strategy.
```
