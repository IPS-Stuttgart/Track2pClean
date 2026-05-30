"""Association helpers for BayesCaTrack."""

from .._exports import ASSOCIATION_PUBLIC_NAMES, reexport
from ..core import bridge as _bridge
from . import (
    _calibrated_mahalanobis_bundle_patch as _calibrated_mahalanobis_bundle_patch,
)
from . import _calibrated_roi_stat_feature_patch as _calibrated_roi_stat_feature_patch
from . import _global_assignment_input_validation as _global_assignment_input_validation

_global_assignment_input_validation.install_global_assignment_input_validation()

_PATCH_MODULES = (
    _calibrated_mahalanobis_bundle_patch,
    _calibrated_roi_stat_feature_patch,
    _global_assignment_input_validation,
)

__all__ = reexport(_bridge, globals(), ASSOCIATION_PUBLIC_NAMES)
