# FullMHT Method Invariant Checklist, 2026-06-26

This checklist records the non-benchmark evidence required before the FullMHT
identity-history row can be discussed as an original method rather than a tuned
cleanup row.  These invariants are intentionally label-free: they prove where the
method can make different choices before any manual-GT metric is read.

| method claim | required regression | evidence required |
| --- | --- | --- |
| Calibrated association likelihood is scan-assignment active, not just an exposed score | `test_calibrated_likelihood_flips_scan_assignment_from_local_overlap` | calibrated likelihood selects the globally coherent edge over a higher registered-IoU local edge |
| No-prior continuation is a birth/death dynamics term | `test_no_prior_continuation_likelihood_opens_scan_assignment_over_death` | the likelihood opens a no-prior continuation where local scoring leaves the row missed/dead |
| Growth-history prediction changes scan-time identity selection | `test_growth_history_prediction_flips_scan_assignment_to_coherent_history` | the history-conditioned growth penalty flips assignment toward the coherent prior trajectory |
| Scan-history pruning is a true history term, not a terminal metric patch | `test_scan_history_conflict_demo_rejects_local_motion_break` and `test_scan_history_conflict_demo_zero_weight_matches_local_score` | local-score pruning keeps the higher raw-score motion break; positive history weight flips to the coherent path, while zero weight exactly matches local scoring |
| Full MHT can beat greedy local history search | `test_full_mht_conflict_demo_mht_history_beats_greedy` | beam search keeps an alternate history until later evidence makes it best |
| Pairwise-good tracking can still be complete-track-bad | `test_full_mht_conflict_demo_pairwise_good_can_be_complete_bad` | greedy remains high pairwise F1 while losing complete identity; full MHT recovers it |
| The constructed witness is label-free in selection | `test_full_mht_conflict_demo_selection_is_reference_independent` | selected paths do not change when only the evaluation reference is changed |
| Method layers do not read GT/audit columns | `test_full_mht_method_layers_do_not_read_gt_or_audit_columns` | FullMHT method-layer modules contain no forbidden GT/audit tokens |
| The leakage scan keeps covering future FullMHT method files | `test_full_mht_no_gt_leakage_scan_covers_all_method_layers` | new FullMHT decision, integration, model, promotion, exposure, and witness files must be explicitly scanned |

## How To Use This Checklist

Run these regressions together with the identity-history manifest, sensitivity,
and exposure bundle.  A good benchmark result without this checklist is not
enough for the paper claim, because it would not prove that the result comes from
label-free identity-history reasoning.

If any invariant fails, keep FullMHT exploratory even if one metric table looks
attractive.  The method story requires all three pieces at once:

```text
scan-assignment likelihoods and dynamics are active
full-history beam search beats greedy in a controlled conflict
scan-history pruning rejects locally attractive but history-incoherent continuations
real-data frozen manifests pass metric, sensitivity, exposure, and no-GT gates
```
