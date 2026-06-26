# FullMHT Complete-History Method Protocol, 2026-06-26

This document defines the paper-facing bar for promoting FullMHT from an
exploratory Track2p cleanup experiment to an original method row.

The core claim is not that MHT is a prettier way to veto one bad edge. The claim
is that longitudinal calcium-imaging tracking can be formulated as Bayesian
identity-history selection: pairwise-good local links can still produce
complete-track failures, and a bounded MHT beam can compare whole identity
histories rather than isolated links.

## Method Hypothesis

A defensible FullMHT method should combine four label-free terms:

1. **Proposal prior**: Track2p edges are treated as strong survival proposals,
   not as immutable ground truth.
2. **Association likelihood**: registered IoU, shifted IoU, area/shape, cell
   probability, centroid distance, growth residual, growth Mahalanobis, and local
   deformation enter as likelihood-ratio evidence rather than hand-picked GT
   ledgers.
3. **Identity dynamics**: missed detections, no-prior continuations, prior
   switches, and gap reactivations are explicit history events with costs or
   likelihoods.
4. **Complete-history objective**: terminal selection may prefer a lower local
   scan score when the complete identity history is more plausible.

The current branch implements all four hooks. The first positive benchmark row is
still the fixed prior-veto hazard, but the branch now also has a calibrated
prior-edge survival likelihood row, an opt-in terminal completion objective, and
an opt-in scan-history pruning objective ready for manifest-level evaluation.

## Current Evidence Map

| layer | current status | evidence | decision |
| --- | --- | --- | --- |
| Full scan-assignment beam | implemented | `track2p-policy-full-mht` | keep |
| Greedy-vs-MHT conflict | constructed positive | `track2p-policy-full-mht-conflict-demo` | use as method intuition |
| Scan-history conflict | constructed positive | `full_mht_scan_history_conflict_demo` | use as method invariant |
| Real-data greedy beam ablation | frozen, not yet run | `FullMHTGreedyPrior2` in `benchmarks/full_mht_prior_veto_manifest.json` | run on server |
| Calibrated association likelihood | implemented, benchmark-negative | `docs/full_mht_calibrated_likelihood_notes.md` | keep as architecture, not row |
| Identity dynamics penalties | implemented, mostly collapse to proposal solution | `track2p_prior_*` diagnostics | keep |
| Identity-diverse beam | implemented, exposes cleaner alternatives | calibrated-likelihood notes | keep |
| Terminal completion objective | implemented, not yet benchmarked | `benchmarks/full_mht_terminal_completion_probe_manifest.json`, `docs/full_mht_terminal_completion_objective.md` | run as method probe |
| Scan-time history pruning | implemented, not yet benchmarked | `benchmarks/full_mht_scan_history_dynamics_probe_manifest.json`, `docs/full_mht_history_dynamics_objective.md` | run as method probe |
| Fixed prior-veto hazard | first positive FullMHT-owned result | `docs/full_mht_prior_risk_notes.md` | freeze and validate |
| Calibrated prior-edge survival | integrated, not yet benchmarked | `full_mht_prior_survival_model.py`, `FullMHTPriorSurvival` manifest row | run on server |
| Manifest-level reproduction | manifest + adapter committed | `benchmarks/full_mht_prior_veto_manifest.json` | run on server |
| Sensitivity/exposure bundle | committed, not yet run | `benchmarks/full_mht_prior_survival_sensitivity_manifest.json`, `docs/full_mht_prior_survival_validation.md` | run on server |

## Conflict Demonstrations

`track2p-policy-full-mht-conflict-demo` now provides two controlled ablations:

- `local-edge-dead-end`: a locally stronger first edge leads to a missed final
  observation, while the MHT beam preserves the weaker first edge until the later
  continuation makes the complete history win.
- `pairwise-good-complete-bad`: the same principle is embedded among many stable
  tracks, so the greedy result remains pairwise-good but creates a wrong complete
  identity. FullMHT keeps the alternative middle edge alive and recovers the
  complete path.

`full_mht_scan_history_conflict_demo` isolates the scan-history pruning layer:
local-score pruning selects a higher-score second edge that causes a severe
within-history motion break, while scan-history-aware pruning selects the lower
raw-score but internally coherent identity history. This demo uses no references,
benchmark scores, or audit labels; it is only an executable invariant for the
label-free history objective.

The conflict scenarios are the method-story invariants: pairwise-good or
locally-good assignment can still be complete-track-bad, and a history beam can
fix the identity history without reading GT labels. They are not benchmark
evidence; they show what the full MHT architecture can do that deterministic
local selection cannot.

Run them with:

```bash
python -m bayescatrack.experiments.track2p_policy_full_mht_conflict_demo \
  --scenario pairwise-good-complete-bad

python -m bayescatrack.experiments.full_mht_scan_history_conflict_demo \
  --output results/full_mht_scan_history_conflict_selected.csv \
  --candidate-output results/full_mht_scan_history_conflict_candidates.csv
```

## Real-Data Greedy Ablation

The canonical manifest now includes:

```text
FullMHTGreedyPrior2
```

This row uses the same proposal-prior settings and scan candidate generator as
`FullMHTPrior2`, but fixes `beam_width = 1`. It is the real-data counterpart of
the conflict demo. If it ties the beam row, the benchmark still does not prove a
history-search advantage. If it loses to the beam row, the method has direct
real-data evidence that preserving alternate identity histories matters.

## Terminal Completion Probe

The branch now has an opt-in complete-history terminal objective:

```text
terminal_incomplete_history_weight
```

The objective penalizes missing observations in non-empty terminal histories. It
is label-free and acts only during beam pruning / final hypothesis selection after
candidate histories exist. The frozen probe manifest is:

```text
benchmarks/full_mht_terminal_completion_probe_manifest.json
```

It tests weights `0.25`, `0.50`, and `1.00` against `Track2p` and
`FullMHTPrior2`. This should be treated as a method probe, not a promoted row,
until it improves complete-track F1 without damaging pairwise F1 and behaves
stably across at least two nearby weights.

## Current Positive Row

The current positive non-teacher FullMHT row is:

```text
FullMHTPriorVetoScaled
```

It keeps the Track2p proposal prior strong, but gives low survival likelihood to
a narrowly defined suspicious Track2p prior edge. In the canonical lightweight
manual-GT benchmark it improved the FullMHT/Track2p proposal control:

| row | pairwise F1 micro | complete-track F1 micro |
| --- | ---: | ---: |
| Track2p | 0.965116 | 0.924370 |
| FullMHTPrior2 | 0.965116 | 0.924370 |
| FullMHTPriorVetoScaled | 0.965919 | 0.932203 |

The important methodological distinction from residual cleanup is that the bad
prior edge is penalized during scan-assignment history selection. It is not
removed after a completed Track2p solution is scored.

## Candidate Method Row

The next paper-facing candidate is:

```text
FullMHTPriorSurvival
```

It uses the same Track2p proposal prior and scan-assignment beam, but replaces
the hand-gated prior-veto pocket with a calibrated, label-free prior-edge survival
log-likelihood ratio. The manifest row is now frozen enough to run, but no
benchmark result has been recorded yet.

## Non-Promotion Conditions

Do not present FullMHT as a final method if any of the following remain true:

- The frozen manifest cannot reproduce the positive row.
- The candidate survival row does not match or improve the fixed prior-veto row.
- The positive row depends on inspecting manual-GT audit columns.
- Exposure audit shows the prior-veto or survival hazard fires broadly across
  non-GT Track2p-style subjects.
- A nearby threshold perturbation selects true-positive removals or causes
  complete-track loss.
- `FullMHTGreedyPrior2` ties the beam row and no constructed or real-data case
  demonstrates a history-level advantage.
- The terminal completion objective improves only at a single fragile weight or
  damages pairwise F1 by rewarding over-linking.
- The scan-history pruning objective improves only at a single fragile weight or
  simply reproduces local-score pruning.
- Deterministic edge gating over the same candidates produces exactly the same
  behavior without any history-level conflict or history-level benefit.
- The paper text cannot distinguish the benchmark row from post-hoc growth-veto
  cleanup.

## Promotion Gates

FullMHT can be promoted as a paper method only after these gates pass:

| gate | required evidence |
| --- | --- |
| Manifest reproduction | `bayescatrack benchmark suite benchmarks/full_mht_prior_veto_manifest.json` reproduces Track2p, FullMHTPrior2, FullMHTGreedyPrior2, FullMHTPriorVetoScaled, and FullMHTPriorSurvival rows |
| No-GT leakage | tests confirm scoring functions do not read `edge_status_against_gt`, `pairwise_delta_if_removed`, `complete_delta_if_removed`, reference identity, or manual-GT status |
| Exposure audit | all Track2p-style subjects report rare prior-veto/survival hazards and no subject receives a broad set of missed prior successors |
| Sensitivity | `benchmarks/full_mht_prior_survival_sensitivity_manifest.json`, `benchmarks/full_mht_terminal_completion_probe_manifest.json`, and `benchmarks/full_mht_scan_history_dynamics_probe_manifest.json` show nearby settings do not collapse pairwise or complete-track metrics |
| Greedy ablation | `FullMHTGreedyPrior2` is compared against the beam rows; a tie is reported honestly as no real-data history-search advantage yet |
| Conflict demonstration | constructed demos, and ideally one real benchmark subject, show a locally better edge loses to a better complete history |
| Reporting | complete-track and pairwise metrics are reported together, with micro/macro variants where relevant |

## Implemented Method Jump

The fixed prior-veto row is promising but still too close to a gated hazard. The
branch now implements three method jumps: a calibrated survival probability for
Track2p prior edges, a terminal complete-history objective, and scan-time
motion-history pruning:

```text
log p(edge survives | label-free diagnostics)
terminal penalty for incomplete seed-anchored histories
scan-time penalty for internally incoherent partial identity histories
```

Candidate features include:

- registered IoU and shifted IoU
- growth residual and growth Mahalanobis
- endpoint cell probabilities and minimum endpoint confidence
- area/shape ratio
- row/column assignment ranks
- terminal-edge and complete-component indicators
- local-neighbor deformation consistency
- terminal missing-observation counts in non-empty identity histories
- immediate within-history motion deterioration between successive selected edges

The MHT score can now combine:

```text
proposal prior + association likelihood + prior-edge survival likelihood
+ missed-detection / death likelihood + scan-time history objective
+ terminal identity-history objective
```

This reduces the current hand-gated prior-veto pocket to a calibrated model layer
when `FullMHTPriorSurvival` is enabled, and it gives both the scan-time beam and
the terminal beam an explicit complete-history objective.

## Server Commands To Run Next

Use the Python 3.12 environment on the benchmark server. The complete validation
recipe lives in:

```text
docs/full_mht_prior_survival_validation.md
docs/full_mht_terminal_completion_objective.md
docs/full_mht_history_dynamics_objective.md
```

Minimum command bundle:

```bash
REPO=/home/florianpfaff/codex-runs/BayesCaTrack
PY="$REPO/.venv312/bin/python"
cd "$REPO"
git fetch origin
git checkout codex/full-mht-prototype
git reset --hard origin/codex/full-mht-prototype
export PYTHONPATH="$REPO/src"

"$PY" -m pytest -q \
  tests/test_benchmark_manifest_full_mht_integration.py \
  tests/test_full_mht_manifest_decision.py \
  tests/test_full_mht_prior_survival_model.py \
  tests/test_full_mht_prior_survival_integration.py \
  tests/test_full_mht_terminal_completion_integration.py \
  tests/test_full_mht_scan_history_dynamics_integration.py \
  tests/test_full_mht_scan_history_conflict_demo.py \
  tests/test_full_mht_no_gt_leakage.py \
  tests/test_track2p_policy_full_mht_conflict_demo.py \
  tests/test_track2p_policy_full_mht_growth_prior.py::test_full_mht_prior_veto_scoring_does_not_read_gt_audit_columns

"$PY" -m bayescatrack.experiments.full_mht_scan_history_conflict_demo \
  --output "$REPO/results/full_mht_scan_history_conflict_selected.csv" \
  --candidate-output "$REPO/results/full_mht_scan_history_conflict_candidates.csv"

OUT="$REPO/results/full_mht_prior_survival_manifest_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_veto_manifest.json \
  --output-dir "$OUT" \
  --summary-format table

SENS="$REPO/results/full_mht_prior_survival_sensitivity_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SENS"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_prior_survival_sensitivity_manifest.json \
  --output-dir "$SENS" \
  --summary-format table

COMP="$REPO/results/full_mht_terminal_completion_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMP"
"$PY" -m bayescatrack benchmark suite \
  benchmarks/full_mht_terminal_completion_probe_manifest.json \
  --output-dir "$COMP" \
  --summary-format table
```

Record the output directories, comparison tables, and promote/keep-exploratory
judgment in `docs/full_mht_prior_survival_validation.md`,
`docs/full_mht_terminal_completion_objective.md`, and
`docs/full_mht_manifest_integration_notes.md` after the run.
