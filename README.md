# Track2pClean

Track2pClean is a post-hoc cleanup and benchmark toolkit for Track2p-style and related calcium-imaging datasets. It focuses on Track2p / Suite2p data ingestion, ROI representation, registration-aware association costs, complete-track-aware residual cleanup, PyRecEst-ready residual-MHT experiments, and reproducible Track2p benchmark ablations.

## Package layout

The repository and user-facing command name are now `Track2pClean`. The historical Python source namespace remains `bayescatrack` for compatibility with existing manifests, scripts, and result bundles:

```text
Track2pClean/
  src/
    bayescatrack/
      core/
      association/
      datasets/
        track2p/
      evaluation/
      experiments/
      io/
        suite2p.py
        track2p.py
    track2pclean/   # thin compatibility namespace / CLI module
  tests/
```

## What it gives you

- Loads Track2p-style subject/session directories.
- Reconstructs Suite2p ROI masks from `stat.npy`.
- Computes ROI centroids and spatial covariance matrices.
- Builds constant-velocity state moments that can initialize PyRecEst filters.
- Builds ROI-aware pairwise association costs and standard `SessionAssociationBundle` objects.
- Registers later-session ROIs into an earlier session's coordinate frame before association.
- Stitches registration-aware pairwise matches into full predicted track rows.
- Cleans suspicious Track2p-style false continuations with residual edit candidates.
- Runs PyRecEst-backed residual-MHT cleanup experiments over bounded candidate sets.
- Loads Track2p reference identities and scores pairwise and complete-track predictions.
- Runs Track2p baseline, PyRecEst global-assignment, and LOSO calibrated-cost benchmark ablations.
- Exports per-session measurements and state moments to a single `.npz` archive.
- Includes a CLI for quick inspection.

## Architecture boundary

Track2pClean owns calcium-imaging semantics such as Track2p/Suite2p ingestion, ROI representation, ROI-derived association costs, registration-aware ROI warping, residual cleanup, and Track2p benchmark scoring. PyRecEst is the backend for generic estimation and hypothesis-selection primitives such as distributions, filters, point-set registration, multi-session assignment, and residual-MHT selection.

See [docs/architecture.md](docs/architecture.md) for the detailed boundary and the Track2p improvement workstream. See [docs/track2p-benchmark-ci.md](docs/track2p-benchmark-ci.md) for the guarded benchmark workflow, published artifacts, and optional regression gates.

## CLI examples

Inspect one subject:

```bash
track2pclean summary /path/to/jm039 --plane plane0
```

The historical command also remains available for compatibility:

```bash
bayescatrack summary /path/to/jm039 --plane plane0
```

Export PyRecEst-ready states:

```bash
track2pclean export /path/to/jm039 /tmp/jm039_plane0.npz \
  --plane plane0 \
  --input-format auto \
  --weighted-masks \
  --weighted \
  --velocity-variance 25.0
```

Validate that PyRecEst objects can be instantiated during export:

```bash
track2pclean export /path/to/jm039 /tmp/jm039_plane0.npz \
  --validate-pyrecest
```

Run the Track2p default benchmark row against independent manual ground truth:

```bash
track2pclean benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method track2p-baseline \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt
```

If a subject directory contains `ground_truth.csv`, the benchmark can use it as the reference automatically. You can also point `--reference` at a `ground_truth.csv` file or at a separate ground-truth root and declare `--reference-kind manual-gt`.

Ground-truth ROI indices are validated against the loaded Suite2p ROI indices. The benchmark keeps all Suite2p `stat.npy` rows by default and lets calibrated costs use Suite2p `iscell` probability as a soft feature rather than discarding low-confidence ROIs before association.

Pass `--no-include-non-cells` only for a legacy hard-filtered ablation; validation will fail clearly if such filtering removes any referenced ROI.

The benchmark refuses Track2p outputs and already row-aligned Suite2p rows as references by default because those are not independent evidence for a paper-facing comparison. For plumbing checks only, pass `--allow-track2p-as-reference-for-smoke-test`.

Sparse manual ground truth is handled by scoring only predicted tracks whose seed-session ROI appears in the reference seed set; this avoids counting unlabelled cells as false positives.

Install the optional Track2p/elastix registration backend when running affine or rigid registration:

```bash
python -m pip install ".[track2p]"
```

Use `--transform-type affine` or `--transform-type rigid` to request Track2p's registration stack. For hosted benchmark runs without this optional backend, use `--transform-type fov-translation`; that selects Track2pClean's integer FOV phase-correlation fallback explicitly.

Create a small synthetic Suite2p-style subject for benchmark development:

```python
from bayescatrack.datasets.track2p import (
    SyntheticFalsePositiveRoi,
    SyntheticTrack2pSubjectConfig,
    write_synthetic_track2p_subject,
)

generated = write_synthetic_track2p_subject(
    "/tmp/track2pclean-synthetic",
    SyntheticTrack2pSubjectConfig(
        subject_name="jm_synthetic",
        missing_detections=((1, 2),),
        non_cell_tracks=(0,),
        false_positive_rois=(
            SyntheticFalsePositiveRoi(session_index=1, center_yx=(15.0, 15.0)),
        ),
    ),
)
```

Run the clean global-assignment ablation with registered IoU costs and skip edges:

```bash
track2pclean benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost registered-iou \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --transform-type fov-translation \
  --max-gap 2
```

Add triplet-projected higher-order consistency to penalize pairwise links that cannot be embedded into coherent three-session paths:

```bash
track2pclean benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost roi-aware \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --transform-type fov-translation \
  --max-gap 2 \
  --higher-order-triplet-weight 0.25 \
  --higher-order-support-top-k 8 \
  --higher-order-support-cost-cap 4.0 \
  --higher-order-max-penalty 2.0
```

Run the Track2pClean ROI-aware cost ablation:

```bash
track2pclean benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost roi-aware \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --transform-type fov-translation \
  --max-gap 2
```

Run the LOSO calibrated-cost ablation:

```bash
track2pclean benchmark track2p \
  --data /path/to/track2p_zenodo \
  --method global-assignment \
  --cost calibrated \
  --split leave-one-subject-out \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --transform-type fov-translation \
  --max-gap 2
```

Run the richer LOSO calibrated-cost path with split Suite2p ROI-stat features, local evidence components, automatic registration selection, and the configurable hard-negative calibration harness:

```bash
track2pclean benchmark track2p-loso-calibration \
  --data /path/to/track2p_zenodo \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --transform-type auto \
  --weighted-masks \
  --pairwise-cost-kwargs-json '{"local_evidence_components": true}' \
  --calibration-feature-set rich \
  --calibration-model hist-gradient-boosting
```

Use calibrated candidate pruning and dynamic edge priors to reject ambiguous links before global assignment:

```bash
track2pclean benchmark track2p-loso-calibration \
  --data /path/to/track2p_zenodo \
  --reference /path/to/manual_ground_truth_root \
  --reference-kind manual-gt \
  --transform-type auto \
  --include-non-cells \
  --cell-probability-threshold 0.0 \
  --pairwise-cost-kwargs-json '{"local_evidence_components": true}' \
  --calibration-feature-set rich \
  --candidate-pruning-json '{"row_top_k": 20, "column_top_k": 20, "probability_threshold": 0.1}' \
  --dynamic-edge-prior-json '{"session_gap_weight": 0.25, "cell_probability_weight": 0.5, "registration_empty_roi_weight": 8.0}' \
  --calibration-model hist-gradient-boosting
```

For diagnosis-first tuning, run `edge-ranking`, then select feature names with `track2pclean benchmark select-edge-ranking-features`, and finally select benchmark variants by complete-track F1 with `track2pclean benchmark select-structured-objective`.

The benchmark prints a compact table by default and can also write JSON or CSV via `--format json --output results.json` or `--format csv --output results.csv`.

Analyze radial growth from an existing track table:

```bash
track2pclean growth radial \
  --subject /path/to/jm039 \
  --tracks results/predicted_tracks.csv \
  --center tracked-centroid \
  --rows-output results/radial_displacements.csv \
  --output results/radial_growth_summary.md
```
