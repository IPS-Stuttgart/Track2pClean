"""Shared parser choices for BayesCaTrack experiment CLIs."""

from __future__ import annotations

REGISTRATION_TRANSFORM_CHOICES: tuple[str, ...] = (
    "auto",
    "affine",
    "rigid",
    "fov-translation",
    "fov-affine",
    "bspline",
    "b-spline",
    "thin-plate-spline",
    "tps",
    "landmark-tps",
    "local-affine-grid",
    "optical-flow",
    "none",
)

REGISTRATION_QA_TRANSFORM_CHOICES: tuple[str, ...] = (
    *REGISTRATION_TRANSFORM_CHOICES[:-1],
    "gt-affine-oracle",
    "none",
)

ASSOCIATION_COST_CHOICES: tuple[str, ...] = (
    "registered-iou",
    "registered-soft-iou",
    "registered-shifted-iou",
    "roi-aware",
    "roi-aware-local",
    "roi-aware-shifted",
    "calibrated",
)

ASSOCIATION_COST_CHOICES_WITHOUT_CALIBRATED: tuple[str, ...] = tuple(
    cost for cost in ASSOCIATION_COST_CHOICES if cost != "calibrated"
)

REGISTRATION_QA_COST_CHOICES: tuple[str, ...] = (
    "registered-iou",
    "registered-soft-iou",
    "registered-shifted-iou",
    "roi-aware",
    "roi-aware-local",
    "roi-aware-shifted",
    "calibrated",
)

REGISTRATION_TRANSFORM_HELP = (
    "Registration transform. Supports Track2p affine/rigid, NumPy FOV affine/translation, "
    "and growth-aware transforms bspline, tps, local-affine-grid, and optical-flow."
)
