"""Runtime loader validation patches for the Track2p/Suite2p bridge.

The core bridge implementation is intentionally kept compact.  These hooks add
input validation and safer auto-format dispatch without changing the public API.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_TRANSIENT_LOAD_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
)


def install_loader_validation_patches(bridge_impl: ModuleType) -> None:
    """Install idempotent validation wrappers on the bridge implementation."""

    original_suite2p_loader = bridge_impl.load_suite2p_plane
    if not getattr(
        original_suite2p_loader, "_bayescatrack_loader_validation_patch", False
    ):

        def _load_suite2p_plane_with_validation(
            plane_dir: str | Path,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            _validate_suite2p_stat_coordinates(
                Path(plane_dir),
                bridge_impl,
                include_non_cells=_strict_bool(
                    kwargs.get("include_non_cells", False),
                    name="include_non_cells",
                ),
                cell_probability_threshold=_finite_probability(
                    kwargs.get("cell_probability_threshold", 0.5),
                    name="cell_probability_threshold",
                ),
                exclude_overlapping_pixels=_strict_bool(
                    kwargs.get("exclude_overlapping_pixels", True),
                    name="exclude_overlapping_pixels",
                ),
            )
            return original_suite2p_loader(plane_dir, *args, **kwargs)

        setattr(
            _load_suite2p_plane_with_validation,
            "_bayescatrack_loader_validation_patch",
            True,
        )
        setattr(
            _load_suite2p_plane_with_validation,
            "_bayescatrack_original",
            original_suite2p_loader,
        )
        bridge_impl.load_suite2p_plane = _load_suite2p_plane_with_validation

    original_subject_loader = bridge_impl.load_track2p_subject
    if not getattr(original_subject_loader, "_bayescatrack_auto_fallback_patch", False):

        def _load_track2p_subject_with_auto_fallback(
            subject_dir: str | Path,
            *,
            plane_name: str = "plane0",
            input_format: str = "auto",
            include_behavior: bool = True,
            strict: bool = False,
            **suite2p_kwargs: Any,
        ) -> list[Any]:
            return _load_track2p_subject_with_auto_fallback_impl(
                bridge_impl,
                subject_dir,
                plane_name=plane_name,
                input_format=input_format,
                include_behavior=include_behavior,
                strict=strict,
                **suite2p_kwargs,
            )

        setattr(
            _load_track2p_subject_with_auto_fallback,
            "_bayescatrack_auto_fallback_patch",
            True,
        )
        setattr(
            _load_track2p_subject_with_auto_fallback,
            "_bayescatrack_original",
            original_subject_loader,
        )
        bridge_impl.load_track2p_subject = _load_track2p_subject_with_auto_fallback


def _validate_suite2p_stat_coordinates(
    plane_dir: Path,
    bridge_impl: ModuleType,
    *,
    include_non_cells: bool,
    cell_probability_threshold: float,
    exclude_overlapping_pixels: bool,
) -> None:
    stat_path = plane_dir / "stat.npy"
    if not stat_path.exists():
        return

    stat = np.load(stat_path, allow_pickle=True)
    if stat.ndim != 1:
        return

    ops = None
    ops_path = plane_dir / "ops.npy"
    if ops_path.exists():
        ops = np.load(ops_path, allow_pickle=True).item()
    image_shape = bridge_impl._infer_image_shape(
        stat, ops
    )  # pylint: disable=protected-access
    height, width = int(image_shape[0]), int(image_shape[1])
    keep_for_validation = _suite2p_validation_keep_mask(
        plane_dir,
        stat,
        include_non_cells=include_non_cells,
        cell_probability_threshold=cell_probability_threshold,
    )

    for roi_index, roi_stat in enumerate(stat):
        if not keep_for_validation[roi_index]:
            continue
        ypix = np.asarray(roi_stat["ypix"], dtype=int)
        xpix = np.asarray(roi_stat["xpix"], dtype=int)
        lam = np.asarray(roi_stat.get("lam", np.ones_like(ypix)), dtype=float)
        if ypix.shape != xpix.shape:
            raise ValueError(
                f"Suite2p ROI {roi_index} has mismatched ypix/xpix shapes: "
                f"{ypix.shape} vs {xpix.shape}"
            )
        if lam.shape != ypix.shape:
            raise ValueError(
                f"Suite2p ROI {roi_index} has lam shape {lam.shape}, "
                f"but expected {ypix.shape}"
            )
        if exclude_overlapping_pixels and "overlap" in roi_stat:
            overlap = np.asarray(roi_stat["overlap"], dtype=bool)
            if overlap.shape != ypix.shape:
                raise ValueError(
                    f"Suite2p ROI {roi_index} has overlap shape {overlap.shape}, "
                    f"but expected {ypix.shape}"
                )
            valid = ~overlap
            ypix = ypix[valid]
            xpix = xpix[valid]
            lam = lam[valid]
        if ypix.size == 0:
            continue
        invalid = (ypix < 0) | (ypix >= height) | (xpix < 0) | (xpix >= width)
        if np.any(invalid):
            bad_y = ypix[invalid]
            bad_x = xpix[invalid]
            raise ValueError(
                f"Suite2p ROI {roi_index} pixel coordinates are out of bounds "
                f"for image shape {image_shape}: "
                f"y range [{int(np.min(bad_y))}, {int(np.max(bad_y))}], "
                f"x range [{int(np.min(bad_x))}, {int(np.max(bad_x))}]"
            )


def _suite2p_validation_keep_mask(
    plane_dir: Path,
    stat: np.ndarray,
    *,
    include_non_cells: bool,
    cell_probability_threshold: float,
) -> np.ndarray:
    if include_non_cells:
        return np.ones((int(stat.shape[0]),), dtype=bool)

    iscell_path = plane_dir / "iscell.npy"
    if not iscell_path.exists():
        return np.ones((int(stat.shape[0]),), dtype=bool)

    iscell = np.asarray(np.load(iscell_path, allow_pickle=True))
    if iscell.ndim not in {1, 2} or iscell.shape[0] != stat.shape[0]:
        return np.ones((int(stat.shape[0]),), dtype=bool)
    if iscell.ndim == 2 and iscell.shape[1] < 1:
        return np.ones((int(stat.shape[0]),), dtype=bool)

    keep = np.zeros((int(stat.shape[0]),), dtype=bool)
    for roi_index in range(int(stat.shape[0])):
        if iscell.ndim == 2:
            is_cell = bool(iscell[roi_index, 0])
            probability = (
                float(iscell[roi_index, 1])
                if iscell.shape[1] > 1
                else float(iscell[roi_index, 0])
            )
        else:
            is_cell = bool(iscell[roi_index])
            probability = float(iscell[roi_index])
        keep[roi_index] = bool(is_cell and probability >= cell_probability_threshold)
    return keep


def _load_track2p_subject_with_auto_fallback_impl(
    bridge_impl: ModuleType,
    subject_dir: str | Path,
    *,
    plane_name: str,
    input_format: str,
    include_behavior: bool,
    strict: bool,
    **suite2p_kwargs: Any,
) -> list[Any]:
    include_behavior = _strict_bool(include_behavior, name="include_behavior")
    strict = _strict_bool(strict, name="strict")
    if input_format not in {"auto", "suite2p", "npy"}:
        raise ValueError("input_format must be 'auto', 'suite2p', or 'npy'")

    subject_path = Path(subject_dir)
    sessions: list[Any] = []
    for session_dir in bridge_impl.find_track2p_session_dirs(subject_path):
        suite2p_plane_dir = session_dir / "suite2p" / plane_name
        npy_plane_dir = session_dir / "data_npy" / plane_name

        plane_data = None
        if input_format == "auto":
            plane_data = _load_auto_plane_with_fallback(
                bridge_impl,
                session_dir=session_dir,
                plane_name=plane_name,
                suite2p_plane_dir=suite2p_plane_dir,
                npy_plane_dir=npy_plane_dir,
                strict=strict,
                suite2p_kwargs=suite2p_kwargs,
            )
            if plane_data is None:
                continue
        elif input_format == "suite2p":
            if not suite2p_plane_dir.exists():
                raise FileNotFoundError(
                    f"Could not find suite2p data for session '{session_dir.name}' "
                    f"and plane '{plane_name}'"
                )
            plane_data = bridge_impl.load_suite2p_plane(
                suite2p_plane_dir,
                **suite2p_kwargs,
            )
        else:
            if not npy_plane_dir.exists():
                raise FileNotFoundError(
                    f"Could not find npy data for session '{session_dir.name}' "
                    f"and plane '{plane_name}'"
                )
            plane_data = bridge_impl.load_raw_npy_plane(npy_plane_dir)

        motion_energy = None
        if include_behavior:
            motion_energy_path = session_dir / "move_deve" / "motion_energy_glob.npy"
            if motion_energy_path.exists():
                motion_energy = np.load(motion_energy_path)

        match = bridge_impl._SESSION_NAME_PATTERN.match(
            session_dir.name
        )  # pylint: disable=protected-access
        session_date = (
            date.fromisoformat(match.group("session_date"))
            if match is not None
            else None
        )
        sessions.append(
            bridge_impl.Track2pSession(
                session_dir=session_dir,
                session_name=session_dir.name,
                session_date=session_date,
                plane_data=plane_data,
                motion_energy=motion_energy,
            )
        )
    return sessions


def _load_auto_plane_with_fallback(
    bridge_impl: ModuleType,
    *,
    session_dir: Path,
    plane_name: str,
    suite2p_plane_dir: Path,
    npy_plane_dir: Path,
    strict: bool,
    suite2p_kwargs: dict[str, Any],
) -> Any | None:
    errors: list[tuple[str, BaseException]] = []

    if suite2p_plane_dir.exists():
        try:
            return bridge_impl.load_suite2p_plane(suite2p_plane_dir, **suite2p_kwargs)
        except _TRANSIENT_LOAD_EXCEPTIONS as exc:
            errors.append(("suite2p", exc))

    if npy_plane_dir.exists():
        try:
            return bridge_impl.load_raw_npy_plane(npy_plane_dir)
        except _TRANSIENT_LOAD_EXCEPTIONS as exc:
            errors.append(("npy", exc))

    if errors:
        detail = "; ".join(
            f"{format_name}: {type(exc).__name__}: {exc}" for format_name, exc in errors
        )
        raise RuntimeError(
            "Could not load any auto input format for recognized Track2p session "
            f"'{session_dir.name}' and plane '{plane_name}'. Tried existing "
            f"candidate directories. Errors: {detail}"
        ) from errors[-1][1]

    message = (
        "Skipping recognized Track2p session "
        f"'{session_dir.name}' for plane '{plane_name}' because neither "
        f"'{suite2p_plane_dir}' nor '{npy_plane_dir}' exists"
    )
    if not strict:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None
    raise FileNotFoundError(message)


def _strict_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_probability(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite probability")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return numeric
