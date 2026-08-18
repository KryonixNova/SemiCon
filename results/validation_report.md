# Drift-Sense validation report

- checkpoint: `puthere/checkpoints/production_v3/best.pt`
- device: NVIDIA GeForce RTX 5060 Ti
- python: 3.12.13
- timing method: wall-clock via time.perf_counter() around the single call to model.predict(reference, search); excludes checkpoint/model loading (a one-time cost, reported separately as model_load_time_s) and PNG decode (not part of the localization algorithm itself)

| condition | n | mean err (px) | median err (px) | worst err (px) | pass@5px | pass@4px | pass@2px | pass@1px | median runtime (ms) |
|---|---|---|---|---|---|---|---|---|---|
| noise=normal_geom=normal | 50 | 0.64 | 0.57 | 1.56 | 1.000 | 1.000 | 1.000 | 0.860 | 21.9 |
| noise=harsh_geom=normal | 50 | 4.32 | 3.59 | 16.52 | 0.640 | 0.540 | 0.340 | 0.220 | 21.9 |
| noise=normal_geom=drift | 50 | 0.67 | 0.59 | 1.92 | 1.000 | 1.000 | 1.000 | 0.860 | 22.0 |
| noise=harsh_geom=drift | 50 | 4.28 | 3.44 | 16.54 | 0.640 | 0.540 | 0.340 | 0.220 | 22.0 |

**Pooled:** n=200, mean error=2.47px

## Failure case

![failure case](failure_case.png)

Worst case (16.5px error) occurred under imaging_noise_profile=harsh, geometric_profile=drift, with model confidence=0.481. The model was still relatively confident despite the large error, suggesting a genuine structural look-alike (repeated-pattern ambiguity) rather than an easily-flagged low-confidence guess.
