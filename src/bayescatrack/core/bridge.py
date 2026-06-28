"""Public bridge exports for BayesCaTrack core."""

# pylint: disable=duplicate-code

from .._exports import BRIDGE_PUBLIC_NAMES
from . import _association_bundle_bool_validation as _association_bundle_bool_validation
from . import (
    _bridge_impl,
)
from . import _cell_probability_validation as _cell_probability_validation
from . import _core_scalar_validation as _core_scalar_validation
from . import _core_string_scalar_validation as _core_string_scalar_validation
from . import _export_bool_validation as _export_bool_validation
from . import _feature_name_validation as _feature_name_validation
from . import _iscell_value_validation as _iscell_value_validation
from . import _loader_bool_validation as _loader_bool_validation
from . import _loader_probability_validation as _loader_probability_validation
from . import _loader_validation as _loader_validation
from . import _local_evidence as _local_evidence
from . import _mahalanobis as _mahalanobis
from . import _roi_index_validation as _roi_index_validation
from . import _roi_stat_features as _roi_stat_features
from . import (
    _suite2p_coordinate_value_validation as _suite2p_coordinate_value_validation,
)
from . import _suite2p_lam_value_validation as _suite2p_lam_value_validation
from . import _suite2p_overlap_value_validation as _suite2p_overlap_value_validation
from . import _with_replaced_masks_fov_validation as _with_replaced_masks_fov_validation

_loader_probability_validation.install_loader_probability_validation(_loader_validation)
_association_bundle_bool_validation.install_association_bundle_bool_validation(
    _bridge_impl
)
_cell_probability_validation.install_cell_probability_cost_patch(_bridge_impl)
_export_bool_validation.install_subject_export_bool_validation(_bridge_impl)
_feature_name_validation.install_feature_name_string_normalization(_bridge_impl)
_loader_bool_validation.install_numpy_bool_loader_validation()
_loader_validation.install_loader_validation_patches(_bridge_impl)
_suite2p_coordinate_value_validation.install_suite2p_coordinate_value_validation(
    _bridge_impl
)
_suite2p_overlap_value_validation.install_suite2p_overlap_value_validation(_bridge_impl)
_iscell_value_validation.install_suite2p_iscell_value_validation(_bridge_impl)
_suite2p_lam_value_validation.install_suite2p_lam_value_validation(_bridge_impl)
_roi_index_validation.install_calcium_plane_roi_index_validation(
    _bridge_impl.CalciumPlaneData
)
_mahalanobis.install_mahalanobis_pairwise_features(_bridge_impl.CalciumPlaneData)
_roi_stat_features.install_split_roi_stat_pairwise_features(
    _bridge_impl.CalciumPlaneData
)
_local_evidence.install_local_evidence_pairwise_features(_bridge_impl.CalciumPlaneData)
_core_scalar_validation.install_core_scalar_validation_patches(
    _bridge_impl.CalciumPlaneData
)
_core_string_scalar_validation.install_core_string_scalar_validation(_core_scalar_validation)
_with_replaced_masks_fov_validation.install_with_replaced_masks_fov_validation(
    _bridge_impl.CalciumPlaneData
)

CalciumPlaneData = _bridge_impl.CalciumPlaneData
SessionAssociationBundle = _bridge_impl.SessionAssociationBundle
Track2pSession = _bridge_impl.Track2pSession
build_consecutive_session_association_bundles = (
    _bridge_impl.build_consecutive_session_association_bundles
)
build_session_pair_association_bundle = (
    _bridge_impl.build_session_pair_association_bundle
)
export_subject_to_npz = _bridge_impl.export_subject_to_npz
find_track2p_session_dirs = _bridge_impl.find_track2p_session_dirs
load_raw_npy_plane = _bridge_impl.load_raw_npy_plane
load_suite2p_plane = _bridge_impl.load_suite2p_plane
load_track2p_subject = _bridge_impl.load_track2p_subject
main = _bridge_impl.main
summarize_subject = _bridge_impl.summarize_subject

__all__ = tuple(name for name in BRIDGE_PUBLIC_NAMES if name in globals())
