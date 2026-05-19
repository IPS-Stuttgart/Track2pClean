#!/usr/bin/env bash
set -euo pipefail

# Copy BayesCaTrack benchmark workflow/local-run artifacts into this paper repository.
# Usage:
#   scripts/copy_bayescatrack_artifacts.sh /path/to/unpacked/artifact-or-results-dir
#
# The source may be either a GitHub Actions artifact root that contains flat files,
# an artifact root that contains a results/ directory, or the local RESULTS_DIR from
# scripts/full_track2p_benchmark_commands.md.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/unpacked/bayescatrack-artifact-or-results-dir" >&2
  exit 2
fi

  exit 2
fi

copy_result() {
  local rel="$1"
  local destination="results/$rel"
  local source=""

  for candidate in "$artifact_dir/$rel" "$artifact_dir/results/$rel"; do
    if [[ -f "$candidate" ]]; then
      source="$candidate"
      break
    fi
  done

  if [[ -n "$source" ]]; then
    mkdir -p "$(dirname "$destination")"
    cp "$source" "$destination"
    echo "copied $rel"
  fi
}

copy_figure() {
  local rel="$1"
  local destination="$rel"
  local source=""

  for candidate in "$artifact_dir/$rel" "$artifact_dir/results/$rel"; do
    if [[ -f "$candidate" ]]; then
      source="$candidate"
      break
    fi
  done

  if [[ -n "$source" ]]; then
    mkdir -p "$(dirname "$destination")"
    cp "$source" "$destination"
    echo "copied $rel"
  fi
}

# Legacy flat workflow artifacts.
for name in \
  comparison.csv \
  comparison.md \
  track2p_baseline.csv \
  global_iou_gap1.csv \
  global_iou_gap2.csv \
  global_registered_iou_gap1.csv \
  global_registered_iou_gap2.csv \
  global_roi_aware_gap1.csv \
  global_roi_aware_gap2.csv \
  global_registered_iou_tuned_priors_gap2.csv \
  global_roi_aware_tuned_priors_gap2.csv \
  global_calibrated_loso_gap2.csv \
  global_monotone_loso_gap2.csv \
  fov_affine_soft_iou_comparison.csv \
  fov_affine_soft_iou_comparison.md \
  fov_affine_shifted_iou_comparison.csv \
  fov_affine_shifted_iou_comparison.md \
  raw_suite2p_roi_diagnostics.csv \
  raw_suite2p_roi_diagnostics.md; do
  copy_result "$name"
done

# Structured local-run artifacts from scripts/full_track2p_benchmark_commands.md.
for name in \
  validation/full_suite2p_roi_validation.csv \
  validation/full_suite2p_roi_validation.md \
  validation/manual_gt_roi_index_audit.csv \
  validation/manual_gt_roi_index_audit.md \
  oracles/oracle_gt_links.csv \
  registration/registration_affine_summary.csv \
  registration/registration_affine_links.csv \
  registration/oracle_affine_summary.csv \
  registration/oracle_affine_links.csv \
  registration/growth_spatial_summary.csv \
  edge_ranking/registered_iou_edge_ranking.csv \
  edge_ranking/registered_iou_edge_ranking_summary.csv \
  edge_ranking/roi_aware_edge_ranking.csv \
  edge_ranking/roi_aware_edge_ranking_summary.csv \
  teacher/teacher_audit_summary.csv \
  teacher/teacher_edges.csv \
  teacher/teacher_focus_bayes_missed_track2p.csv \
  teacher/teacher_training_rows.csv \
  teacher/teacher_debug_details.csv \
  teacher/teacher_debug_summary.csv \
  teacher/teacher_debug_metrics.csv \
  benchmarks/track2p_baseline.csv \
  benchmarks/global_registered_iou_gap1.csv \
  benchmarks/global_registered_iou_gap2.csv \
  benchmarks/global_roi_aware_gap2.csv \
  benchmarks/global_registered_iou_fov_affine_tuned_gap2.csv \
  benchmarks/global_registered_iou_tuned_priors_gap2.csv \
  benchmarks/global_roi_aware_tuned_priors_gap2.csv \
  benchmarks/global_calibrated_loso_gap2.csv \
  benchmarks/global_monotone_loso_gap2.csv \
  benchmarks/comparison.csv \
  benchmarks/comparison.md \
  benchmarks/comparison_with_oracles.csv \
  benchmarks/comparison_with_oracles.md; do
  copy_result "$name"
done

copy_figure figures/benchmark_comparison.svg
