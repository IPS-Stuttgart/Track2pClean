# Track2p Benchmark CI

The `Track2p benchmark` workflow runs the guarded Track2p ablation suite and stores the generated benchmark evidence in a durable `track2p-benchmark-results` artifact.

The workflow expects repository variables that point to data available on the runner:

| variable | meaning |
| --- | --- |
| `TRACK2P_DATA_PATH` | Track2p dataset root or a single subject directory. |
| `TRACK2P_REFERENCE_PATH` | Manual-ground-truth root or reference file. |
| `TRACK2P_TRANSFORM_TYPE` | Registration transform; defaults to `affine` in the workflow. |
| `TRACK2P_REFERENCE_KIND` | Reference type; defaults to `manual-gt` in the benchmark script. |
| `TRACK2P_PLANE` | Plane name; defaults to `plane0`. |
| `TRACK2P_INPUT_FORMAT` | Input format; defaults to `auto`. |
| `TRACK2P_MAX_GAP` | Maximum forward session gap; defaults to `2`. |
| `TRACK2P_RUN_CALIBRATED_LOSO` | `auto`, `true`, or `false`; `auto` runs LOSO only when at least two subjects are found. |
| `TRACK2P_PAIRWISE_COST_KWARGS_JSON` | Optional JSON object forwarded to pairwise-cost construction. |
| `TRACK2P_INCLUDE_POLICY_PRUNED_EXPERIMENT` | Optional `true`/`false` flag to add the conservative prune-only Track2p-policy row to the comparison. |

The workflow writes these files under `benchmark-results/`:

| file | purpose |
| --- | --- |
| `track2p_benchmark_manifest.json` | Exact manifest generated for the run. |
| `run_metadata.json` | Subjects, calibrated-LOSO policy, and the pinned PyRecEst revision. |
| `track2p_baseline.csv` | Track2p baseline subject-level rows. |
| `track2p_policy_pruned.csv` | Optional conservative prune-only Track2p-policy rows when `TRACK2P_INCLUDE_POLICY_PRUNED_EXPERIMENT=true`. |
| `global_registered_iou.csv` | Global-assignment registered-IoU subject-level rows. |
| `global_roi_aware.csv` | Global-assignment ROI-aware subject-level rows. |
| `global_calibrated_loso.csv` | Optional calibrated LOSO subject-level rows. |
| `comparison.md` | Human-readable aggregate comparison table. |
| `comparison.csv` | Machine-readable aggregate comparison table. |
| `workflow-summary.md` | Concise GitHub Actions summary assembled from the artifacts and regression gates. |

The prune-only policy row is intentionally opt-in. It never adds rescue edges; it only removes threshold-accepted policy edges with weak threshold margin, weak row/column competition margin, and weak area or centroid evidence.

## Optional regression gates

Set repository variables or workflow-dispatch inputs to make the job fail when the best Track2pClean ablation drops below a configured threshold. Inputs override repository variables for one manual run.

| gate | condition |
| --- | --- |
| `TRACK2P_MIN_BEST_PAIRWISE_F1_MACRO` | Best non-baseline `pairwise_f1_macro` must be at least this value. |
| `TRACK2P_MIN_BEST_COMPLETE_TRACK_F1_MACRO` | Best non-baseline `complete_track_f1_macro` must be at least this value. |
| `TRACK2P_MIN_PAIRWISE_F1_MACRO_DELTA_OVER_BASELINE` | Best non-baseline `pairwise_f1_macro` minus the baseline value must be at least this value. |
| `TRACK2P_MIN_COMPLETE_TRACK_F1_MACRO_DELTA_OVER_BASELINE` | Best non-baseline `complete_track_f1_macro` minus the baseline value must be at least this value. |

`TRACK2P_BASELINE_APPROACH` can be used to change the baseline label used for delta gates. Its default is `Track2p baseline`, matching the generated comparison manifest.

The summary uploaded to the GitHub Actions UI includes a pass/fail table for all configured gates and the rendered aggregate comparison. This makes benchmark regressions visible even before downloading artifacts.
