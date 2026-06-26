# Architecture Boundary

Track2pClean is the calcium-imaging tracking cleanup and benchmark layer. PyRecEst is the generic estimation and hypothesis-selection backend. The boundary matters because Track2pClean needs to improve Track2p-style outputs with domain-specific ROI handling and residual cleanup, while PyRecEst should remain reusable for tracking and estimation problems outside calcium imaging.

## Track2pClean owns calcium-imaging semantics

Keep ROI- and dataset-specific behavior in Track2pClean:

- Track2p and Suite2p loading, directory discovery, and reference parsing.
- Suite2p ROI fields such as `ypix`, `xpix`, `lam`, `overlap`, `med`, `iscell`, and ROI feature vectors.
- Dense ROI masks, weighted masks, overlap exclusion, centroids, spatial covariances, and ROI feature extraction.
- Registration-aware ROI warping into a common imaging frame.
- Pairwise association cost construction from ROI overlap, mask cosine similarity, area ratios, cell probabilities, centroid distances, and dataset-calibrated gates.
- Complete-track-aware residual cleanup and terminal false-continuation diagnostics.
- Track2p benchmark evaluation and score reporting.

These choices are specific to longitudinal calcium-imaging data and should be easy to tune against Track2p baselines without expanding PyRecEst's public surface with neuroscience-specific APIs.

## PyRecEst owns generic backend primitives

Use PyRecEst for reusable estimation and optimization components:

- Gaussian distributions, filters, and tracker state representations.
- Point-set registration and transform estimation over generic landmarks.
- Global multi-session assignment from pairwise costs or scores.
- Observation-specific birth/death costs, gap penalties, and assignment result utilities.
- Residual hypothesis selection for bounded edit sets.

Track2pClean should convert calcium-imaging observations into PyRecEst-ready arrays, cost matrices, state moments, and residual edit candidates, then call PyRecEst for the generic backend step.

## Integration Pattern

The intended dependency direction is one-way:

```text
Track2p/Suite2p data
    -> Track2pClean ROI representation, registration, costs, cleanup, and benchmarks
    -> PyRecEst registration, assignment, filtering, and state estimation
    -> Track2pClean labels, exports, diagnostics, and Track2p comparison
```

That means Track2pClean can expose convenience methods such as `build_session_pair_association_bundle()` and `build_registered_session_pair_association_bundle()`, but their ROI cost and cleanup policies should remain in Track2pClean. PyRecEst should receive generic matrices, point arrays, or residual edit candidates, not Suite2p dictionaries or Track2p folder structures.

## Track2p Improvement Workstream

To improve Track2p-style outputs, Track2pClean should keep benchmarkable decisions close to the ROI pipeline:

- Preserve weighted ROI masks where available instead of reducing every ROI to a binary support.
- Tune association costs on registered ROIs rather than raw session coordinates.
- Record cost components for ablation and error analysis.
- Score predictions against Track2p references with precision, recall, and F1.
- Treat PyRecEst backend swaps as controlled experiments while keeping the same Track2pClean ROI inputs and evaluation metrics.

This keeps the scientific comparison in Track2pClean and the generic estimation algorithms in PyRecEst.
