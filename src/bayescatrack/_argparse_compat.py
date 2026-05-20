"""CLI-only argparse compatibility helpers.

This module deliberately does not run at package import time. The console entry
point uses it to keep legacy experiment parsers accepting newer registration
transforms and soft-overlap costs until those parsers are migrated to shared
explicit choice constants.
"""

from __future__ import annotations


def install_registration_transform_argparse_patch() -> None:
    """Expand legacy BayesCaTrack CLI choices without mutating argparse on import."""

    import argparse as _argparse

    current_add_argument = _argparse.ArgumentParser.add_argument
    patch_flag = "bayescatrack_registration_transform_patch"
    if getattr(current_add_argument, patch_flag, False):
        return

    growth_choices = (
        "fov-affine",
        "bspline",
        "b-spline",
        "thin-plate-spline",
        "tps",
        "landmark-tps",
        "local-affine-grid",
        "optical-flow",
    )
    soft_cost_choices = (
        "registered-soft-iou",
        "registered-shifted-iou",
        "roi-aware-shifted",
    )

    def _expanded_choices(choices, extra_choices):
        try:
            choices_tuple = tuple(choices) if choices is not None else ()
        except TypeError:
            return choices
        if not choices_tuple:
            return choices
        expanded = []
        for value in choices_tuple:
            if value == "none":
                for extra_choice in extra_choices:
                    if extra_choice not in expanded:
                        expanded.append(extra_choice)
            expanded.append(value)
        if "none" not in choices_tuple:
            for extra_choice in extra_choices:
                if extra_choice not in expanded:
                    expanded.append(extra_choice)
        return tuple(dict.fromkeys(expanded))

    def _bayescatrack_add_argument(self, *name_or_flags, **kwargs):
        if "--transform-type" in name_or_flags:
            choices = kwargs.get("choices")
            kwargs = {**kwargs, "choices": _expanded_choices(choices, growth_choices)}
            help_text = kwargs.get("help")
            if isinstance(help_text, str) and "bspline" not in help_text:
                kwargs["help"] = (
                    f"{help_text}; supports fov-affine and growth-aware transforms "
                    "bspline, tps, local-affine-grid, and optical-flow"
                )
        if "--cost" in name_or_flags:
            choices = kwargs.get("choices")
            try:
                choices_tuple = tuple(choices) if choices is not None else ()
            except TypeError:
                choices_tuple = ()
            if "registered-iou" in choices_tuple:
                kwargs = {
                    **kwargs,
                    "choices": _expanded_choices(choices, soft_cost_choices),
                }
            help_text = kwargs.get("help")
            if isinstance(help_text, str) and "registered-soft-iou" not in help_text:
                kwargs["help"] = (
                    f"{help_text}; supports registered-soft-iou and "
                    "registered-shifted-iou/roi-aware-shifted for near-miss "
                    "registered ROI overlap"
                )
        return current_add_argument(self, *name_or_flags, **kwargs)

    setattr(_bayescatrack_add_argument, patch_flag, True)
    setattr(_argparse.ArgumentParser, "add_argument", _bayescatrack_add_argument)
