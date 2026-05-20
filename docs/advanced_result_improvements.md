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
