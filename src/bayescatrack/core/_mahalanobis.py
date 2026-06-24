"""Mahalanobis centroid-distance extension for Track2p ROI association."""

from __future__ import annotations

from typing import Any

import numpy as np
from bayescatrack._pyrecest_pairwise_features import pairwise_mahalanobis_distances

_MAHALANOBIS_INSTALLED_ATTR = "_bayescatrack_mahalanobis_installed"


def install_mahalanobis_pairwise_features(calcium_plane_cls: type[Any]) -> None:
    """Install Mahalanobis centroid-distance helpers on ``CalciumPlaneData``.

    The core bridge implementation is intentionally kept compact. This small
    extension adds covariance-normalized centroid distances while preserving the
    existing public class and method names.
    """

    if getattr(calcium_plane_cls, _MAHALANOBIS_INSTALLED_ATTR, False):
        return

    original_build_pairwise_cost_matrix = calcium_plane_cls.build_pairwise_cost_matrix

    def pairwise_mahalanobis_centroid_distances(
        self: Any,
        other: Any,
        *,
        order: str = "xy",
        weighted: bool = False,
        regularization: float = 1.0e-6,
    ) -> np.ndarray:
        """Return covariance-normalized centroid distances for all ROI pairs.

        The distance for pair ``(i, j)`` is

        ``sqrt((mu_i - mu_j)^T (Sigma_i + Sigma_j)^-1 (mu_i - mu_j))``.
        """

        regularization = _validate_nonnegative_finite_scalar(
            "regularization",
            regularization,
        )
        if self.n_rois == 0 or other.n_rois == 0:
            return np.zeros((self.n_rois, other.n_rois), dtype=float)

        centroids_self = self.centroids(order=order, weighted=weighted)
        centroids_other = other.centroids(order=order, weighted=weighted)
        covariances_self = self.position_covariances(
            order=order,
            weighted=weighted,
            regularization=regularization,
        )
        covariances_other = other.position_covariances(
            order=order,
            weighted=weighted,
            regularization=regularization,
        )

        return np.asarray(
            pairwise_mahalanobis_distances(
                centroids_self,
                covariances_self,
                centroids_other,
                covariances_other,
                regularization=0.0,
            ),
            dtype=float,
        )

    # pylint: disable=too-many-arguments,too-many-locals
    def build_pairwise_cost_matrix(
        self: Any,
        other: Any,
        *,
        order: str = "xy",
        weighted_centroids: bool = False,
        centroid_weight: float = 1.0,
        mahalanobis_weight: float = 0.0,
        mahalanobis_regularization: float = 1.0e-6,
        centroid_scale: float | None = None,
        max_centroid_distance: float | None = None,
        iou_weight: float = 6.0,
        mask_cosine_weight: float = 2.0,
        area_weight: float = 0.5,
        roi_feature_weight: float = 0.25,
        feature_names: Any = None,
        cell_probability_weight: float = 0.0,
        soft_iou: bool = False,
        large_cost: float = 1.0e6,
        similarity_epsilon: float = 1.0e-6,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build a ROI-aware association cost matrix with Mahalanobis features."""

        mahalanobis_regularization = _validate_nonnegative_finite_scalar(
            "mahalanobis_regularization",
            mahalanobis_regularization,
        )
        mahalanobis_weight = _validate_nonnegative_finite_scalar(
            "mahalanobis_weight",
            mahalanobis_weight,
        )

        base_cost, components = original_build_pairwise_cost_matrix(
            self,
            other,
            order=order,
            weighted_centroids=weighted_centroids,
            centroid_weight=centroid_weight,
            centroid_scale=centroid_scale,
            max_centroid_distance=max_centroid_distance,
            iou_weight=iou_weight,
            mask_cosine_weight=mask_cosine_weight,
            area_weight=area_weight,
            roi_feature_weight=roi_feature_weight,
            feature_names=feature_names,
            cell_probability_weight=cell_probability_weight,
            soft_iou=soft_iou,
            large_cost=large_cost,
            similarity_epsilon=similarity_epsilon,
            return_components=True,
        )
        components = dict(components)

        mahalanobis_distances = pairwise_mahalanobis_centroid_distances(
            self,
            other,
            order=order,
            weighted=weighted_centroids,
            regularization=mahalanobis_regularization,
        )
        mahalanobis_cost = mahalanobis_distances**2
        total_cost = (
            np.asarray(base_cost, dtype=float) + mahalanobis_weight * mahalanobis_cost
        )
        gated = np.asarray(
            components.get("gated", np.zeros_like(total_cost, dtype=bool)), dtype=bool
        )
        total_cost = np.where(gated, large_cost, total_cost)
        total_cost = np.nan_to_num(
            total_cost, nan=large_cost, posinf=large_cost, neginf=large_cost
        )

        components["pairwise_cost_matrix"] = total_cost
        components["mahalanobis_centroid_distance"] = mahalanobis_distances
        components["mahalanobis_centroid_cost"] = mahalanobis_cost

        if return_components:
            return total_cost, components
        return total_cost

    calcium_plane_cls.pairwise_mahalanobis_centroid_distances = (
        pairwise_mahalanobis_centroid_distances
    )
    calcium_plane_cls.build_pairwise_cost_matrix = build_pairwise_cost_matrix
    setattr(calcium_plane_cls, _MAHALANOBIS_INSTALLED_ATTR, True)


def _validate_nonnegative_finite_scalar(name: str, raw_value: Any) -> float:
    if isinstance(raw_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative value")
    value = float(raw_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative value")
    return value
