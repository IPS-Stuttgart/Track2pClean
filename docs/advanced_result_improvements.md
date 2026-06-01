# Advanced result-improvement workstreams

This document accompanies the third incremental improvement patch.  The patch is
not meant to replace the existing benchmark pipeline; it adds modular utilities
that can be wired into LOSO experiments one at a time.

## Workstreams covered

1. **Bayesian uncertainty-aware assignment**
   - `bayescatrack.association.advanced_uncertainty`
   - Converts registration diagnostics, gated/empty masks, covariance mismatch,
     local rank, and activity availability into an explicit edge reliability
     matrix.

2. **Track-level smoothing and repair suggestions**
   - `bayescatrack.association.track_refinement`
   - Fits simple smooth track trajectories and flags high-residual detections or
     edges that should be split or relinked.

3. **Split/merge segmentation diagnostics**
   - `bayescatrack.association.segmentation_events`
   - Detects one-to-many and many-to-one ROI events from overlap, containment,
     weighted Dice, and area-ratio components.

4. **Subject/session-specific calibration intercepts**
   - `bayescatrack.association.session_adaptive_calibration`
   - Adds observable, non-GT context shifts for session gap, ROI density,
     registration quality, cell probability, trace availability, and backend.

5. **Density-aware local competition and FOV context**
   - `bayescatrack.association.context_descriptors`
   - Builds lightweight local density, graph-neighborhood, and FOV patch-moment
     descriptors that can be used as additional calibrated features.

6. **Multi-hypothesis and ensemble/consensus tracking**
   - `bayescatrack.association.multi_hypothesis`
   - Keeps top-k candidate edges, enumerates beam-search track hypotheses, and
     extracts consensus edges across model variants.

7. **Joint registration-assignment scaffolding**
   - `bayescatrack.association.joint_registration_assignment`
   - Provides a callback-based loop where high-confidence assignment anchors can
     be fed back into registration.

8. **Biological growth/deformation priors**
   - `bayescatrack.association.growth_priors`
   - Adds affine and radial growth penalties and estimates an affine growth field
     from complete tracks.

9. **Negative evidence from absence**
   - `bayescatrack.association.absence_model`
   - Converts observability cues into per-ROI absence/gap penalties.

10. **Cross-plane consistency**
    - `bayescatrack.association.multiplane_consistency`
    - Aggregates shared registration quality across planes and can add a shared
      quality penalty.

11. **Active learning, stratified benchmarking, synthetic stress manifests, and
    probability rejection curves**
    - `bayescatrack.experiments.advanced_improvement_workbench`
    - Standalone CLI workbench for label-candidate ranking, metric
      stratification, stress-manifest generation, and precision/recall threshold
      tables.

## Example commands

Rank edge-ranking or teacher-audit rows for additional manual annotation:

```bash
python -m bayescatrack.experiments.advanced_improvement_workbench active-labels \
  --input results/edge_ranking.csv \
  --output results/active_label_candidates.csv \
  --max-rows 250
```

Stratify benchmark outputs by subject and registration backend:

```bash
python -m bayescatrack.experiments.advanced_improvement_workbench stratify \
  --input results/benchmark.csv \
  --output results/benchmark_stratified.csv \
  --group-field subject \
  --group-field registration_backend \
  --metric pairwise_f1 \
  --metric complete_track_f1
```

Create a stress-test benchmark manifest:

```bash
python -m bayescatrack.experiments.advanced_improvement_workbench stress-manifest \
  --data-root data/synthetic-stress \
  --reference-root data/synthetic-stress \
  --output-root results/stress \
  --output benchmarks/stress_suite.json
```

Create the paper-facing Track2p result-improvement manifest recommended for the
next round of benchmark runs:

```bash
bayescatrack advanced track2p-improvement-manifest \
  --data-root ../benchmark-raw-suite2p-subjects \
  --reference-root ../benchmark-raw-suite2p-subjects \
  --output-root results/improvements \
  --max-gap 2 \
  --transform-type fov-affine \
  --output benchmarks/track2p_result_improvements.json
```

The generated manifest now also enables all-seed benchmark scoring, exposes the
auto-registration candidate list used when `--transform-type auto` is selected,
uses bilinear FOV-affine mask warping for weighted-mask experiments, and includes
reliability-aware/pruned ROI-aware shifted-cost rows. These rows wire
`bayescatrack.association.advanced_uncertainty` into the normal global assignment
path, so questionable edges from weak registration diagnostics, gated components,
empty warped ROIs, or poor local margins can be penalized before candidate
pruning and path-cover assignment. The suite also includes a LOSO solver-prior
run whose search objective is `complete_track_f1`, keeping the main optimization
target aligned with paper-facing complete-track quality.

Run the manifest after reviewing or editing the generated JSON:

```bash
bayescatrack benchmark suite benchmarks/track2p_result_improvements.json \
  --output-dir . \
  --progress
```

The generated suite compares Track2p, registered-IoU solver-prior sweeps,
Track2p-policy reproduction, component cleanup, coherence-gated suffix stitching,
DP-rescued and prune-only Track2p-policy variants, shifted/ROI-aware costs,
higher-order consistency, activity tie-breaking, oracle GT-link diagnostics,
auto-registration selection, uncertainty-aware pruning, dynamic edge priors,
fold-clean complete-track-F1 solver-prior tuning, local-evidence calibrated LOSO,
configurable hard negatives, histogram-gradient calibration, monotone LOSO
ranking, and registration QA.

Pass `--include-experimental-policy-dp` when generating the manifest to add the
wider, more aggressive DP-rescue row in addition to the default conservative DP
row.

The benchmark CLI also exposes one-command hooks for the next result-quality
round that were not part of the initial improvement patches:

```bash
bayescatrack benchmark track2p \
  --data ../benchmark-raw-suite2p-subjects \
  --reference ../benchmark-raw-suite2p-subjects \
  --reference-kind manual-gt \
  --method global-assignment \
  --transform-type auto \
  --auto-registration-candidates none,fov-translation,fov-affine,affine,rigid,local-affine-grid,tps,bspline,optical-flow \
  --registration-options-json '{"min_nonrigid_inverse_warp_valid_fraction":0.92}' \
  --segmentation-event-json '{"min_overlap_fraction":0.2,"min_weighted_dice":0.15}' \
  --joint-refinement-json '{"high_confidence_quantile":0.05,"min_anchor_edges":12,"cost_relief":0.2}' \
  --consensus-prior-json '{"variant_costs":["registered-iou","registered-shifted-iou","roi-aware-shifted"],"min_votes":2,"relief":0.2}' \
  --track-refinement-json '{"residual_z_threshold":3.5,"split_bad_edges":true}' \
  --postsolve-relink-json '{"max_edge_cost":6.0,"min_cost_improvement":0.25}'
```

Use `bayescatrack benchmark select-structured-objective --nested-held-out-field subject`
on the resulting CSV files to select pipeline configurations by complete-track
F1 without choosing hyperparameters on the held-out subject.

For targeted command-line experiments, the Track2p benchmark now accepts
`--seed-sessions all`, `--auto-registration-candidates`,
`--fov-affine-mask-warp-mode`, and `--edge-uncertainty-json` alongside the
existing candidate-pruning, dynamic-prior, solver-prior, and
higher-order-consistency options.

Build a rejection-threshold precision/recall table:

```bash
python -m bayescatrack.experiments.advanced_improvement_workbench pr-table \
  --input results/calibrated_edge_probabilities.csv \
  --probability-column probability \
  --label-column label \
  --output results/probability_thresholds.csv
```

## Recommended review order

1. Use the standalone workbench commands first because they do not affect normal
   tracking output.
2. Add `advanced_uncertainty` and `context_descriptors` as optional calibrated
   feature sources.
3. Evaluate `track_refinement` and `segmentation_events` on held-out error
   ledgers before applying automated repairs.
4. Wire `growth_priors`, `absence_model`, and `multiplane_consistency` into
   fold-internal benchmark variants only after their diagnostics improve
   edge-ranking and complete-track metrics.
