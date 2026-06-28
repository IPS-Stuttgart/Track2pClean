"""Association helpers for BayesCaTrack."""

from .._exports import ASSOCIATION_PUBLIC_NAMES, reexport
from ..core import bridge as _bridge
from . import _absence_config_scalar_validation as _absence_config_scalar_validation
from . import _absence_cue_shape_validation as _absence_cue_shape_validation
from . import _absence_roi_count_validation as _absence_roi_count_validation
from . import _absence_session_gap_validation as _absence_session_gap_validation
from . import (
    _activity_similarity_control_validation as _activity_similarity_control_validation,
)
from . import (
    _advanced_uncertainty_array_validation as _advanced_uncertainty_array_validation,
)
from . import (
    _calibrated_mahalanobis_bundle_patch as _calibrated_mahalanobis_bundle_patch,
)
from . import _calibrated_roi_stat_feature_patch as _calibrated_roi_stat_feature_patch
from . import _calibrated_session_gap_validation as _calibrated_session_gap_validation
from . import (
    _dynamic_activity_component_validation as _dynamic_activity_component_validation,
)
from . import _dynamic_edge_prior_validation as _dynamic_edge_prior_validation
from . import _global_assignment_input_validation as _global_assignment_input_validation
from . import _global_solver_track_validation as _global_solver_track_validation
from . import _growth_coordinate_validation as _growth_coordinate_validation
from . import _growth_prior_scalar_validation as _growth_prior_scalar_validation
from . import _monotone_ranker_feature_validation as _monotone_ranker_feature_validation
from . import _neuropil_ratio_shape_validation as _neuropil_ratio_shape_validation
from . import (
    _postsolve_relinking_input_validation as _postsolve_relinking_input_validation,
)
from . import (
    _registered_component_shape_validation as _registered_component_shape_validation,
)
from . import _roi_aware_local_validation as _roi_aware_local_validation
from . import _session_edge_pair_validation as _session_edge_pair_validation
from . import _shifted_iou_preset_validation as _shifted_iou_preset_validation
from . import (
    _track2p_policy_session_gap_validation as _track2p_policy_session_gap_validation,
)
from . import (
    _track_refinement_fill_value_validation as _track_refinement_fill_value_validation,
)
from . import (
    _track_refinement_numeric_control_validation as _track_refinement_numeric_control_validation,
)
from . import (
    _track_refinement_row_sentinel_validation as _track_refinement_row_sentinel_validation,
)
from . import _triplet_support_validation as _triplet_support_validation
from . import absence_model as _absence_model

_absence_config_scalar_validation.install_absence_config_scalar_validation(
    _absence_model
)
_absence_roi_count_validation.install_absence_roi_count_validation(_absence_model)
_absence_cue_shape_validation.install_absence_cue_shape_validation(_absence_model)
_absence_session_gap_validation.install_absence_session_gap_validation(_absence_model)
_activity_similarity_control_validation.install_activity_similarity_control_validation()
_advanced_uncertainty_array_validation.install_advanced_uncertainty_array_validation()
_neuropil_ratio_shape_validation.install_neuropil_ratio_shape_validation()
_session_edge_pair_validation.install_session_edge_pair_validation()
_triplet_support_validation.install_triplet_support_validation()
_global_assignment_input_validation.install_global_assignment_input_validation()
_global_solver_track_validation.install_global_solver_track_validation()
_monotone_ranker_feature_validation.install_monotone_ranker_feature_validation()
_roi_aware_local_validation.install_roi_aware_local_validation()
_registered_component_shape_validation.install_registered_component_shape_validation()
_shifted_iou_preset_validation.install_shifted_iou_preset_validation()
_dynamic_edge_prior_validation.install_dynamic_edge_prior_bool_validation()
_dynamic_activity_component_validation.install_dynamic_activity_component_shape_validation()
_track2p_policy_session_gap_validation.install_track2p_policy_session_gap_validation()
_track_refinement_fill_value_validation.install_track_refinement_fill_value_validation()
_track_refinement_numeric_control_validation.install_track_refinement_numeric_control_validation()
_track_refinement_row_sentinel_validation.install_track_refinement_row_sentinel_validation()
_growth_coordinate_validation.install_growth_coordinate_validation()
_growth_prior_scalar_validation.install_growth_prior_scalar_validation()
_postsolve_relinking_input_validation.install_postsolve_relinking_input_validation()

_PATCH_MODULES = (
    _absence_config_scalar_validation,
    _absence_cue_shape_validation,
    _absence_roi_count_validation,
    _absence_session_gap_validation,
    _activity_similarity_control_validation,
    _advanced_uncertainty_array_validation,
    _calibrated_mahalanobis_bundle_patch,
    _calibrated_roi_stat_feature_patch,
    _calibrated_session_gap_validation,
    _dynamic_activity_component_validation,
    _dynamic_edge_prior_validation,
    _global_assignment_input_validation,
    _global_solver_track_validation,
    _growth_coordinate_validation,
    _growth_prior_scalar_validation,
    _monotone_ranker_feature_validation,
    _neuropil_ratio_shape_validation,
    _postsolve_relinking_input_validation,
    _registered_component_shape_validation,
    _roi_aware_local_validation,
    _session_edge_pair_validation,
    _shifted_iou_preset_validation,
    _track2p_policy_session_gap_validation,
    _track_refinement_fill_value_validation,
    _track_refinement_numeric_control_validation,
    _track_refinement_row_sentinel_validation,
    _triplet_support_validation,
)

__all__ = reexport(_bridge, globals(), ASSOCIATION_PUBLIC_NAMES)
